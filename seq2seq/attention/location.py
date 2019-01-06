"""
Positioning attention.

TO DO:
- remove all the dependencies on `additional`. By removing the number of possible
    hyperparameters it will be a lot simple to simply write down the functions
    with specific inputs / outputs without dependencies on `additional`.
- many parameters that I keep for dev mode / comparasion here: you definitely
    don't have to accept all of these when refactoring. In case you are not sure
    which ones to keep : just ask me (but the ones that I know we should remove
    for sure are noted.

Contact: Yann Dubois
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

from seq2seq.util.helpers import (renormalize_input_length, get_rnn, get_extra_repr,
                                  clamp, format_source_lengths, leaky_noisy_clamp,
                                  clamp_regularize, HyperparameterInterpolator,
                                  HyperparameterCurriculumInterpolator, get_indices,
                                  regularization_loss, batch_reduction_f)
from seq2seq.util.torchextend import get_rounder, L0Gates, get_gate, PlateauAct
from seq2seq.util.initialization import replicate_hidden0
from seq2seq.util.base import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_regularizers_location(total_training_calls, n_steps_prepare_pos):
    def _initialize_regularizer(name, curriculum, **kwargs):
        max_p_interpolators[name] = HyperparameterCurriculumInterpolator(curriculum, **kwargs)

    max_p_interpolators = dict()

    _initialize_regularizer("pos_l0_weights",
                            [dict(step=int(n_steps_prepare_pos / 10), value=0),
                             dict(step=n_steps_prepare_pos, value=5e-3)])

    _initialize_regularizer("pos_clamp_mu",
                            [dict(step=0, value=5e-2)])

    _initialize_regularizer("pos_const_weights",
                            [dict(step=int(n_steps_prepare_pos / 10), value=5e-2),
                             dict(step=n_steps_prepare_pos, value=1e-3)])

    return max_p_interpolators


def _discrete_truncated_gaussian(x, mu, sigma):
    """Return normalized Gaussian_pdf(x)."""
    x = torch.exp(-(x - mu)**2 / (2 * sigma**2))
    x = F.normalize(x, p=1, dim=0)
    return x


def _discrete_truncated_laplace(x, mu, sigma):
    """Return normalized Laplacian_pdf(x)."""
    b = sigma / (2)**0.5
    x = torch.exp(-1 * torch.abs((x - mu) / b))
    x = F.normalize(x, p=1, dim=0)
    return x


def get_loc_pdf(name):
    """Get the correct positioner method.

    Args:
        pdf ({"gaussian", "laplace"}, optional): name of the pdf to use to
            generate the location attention.
    """
    if name == "gaussian":
        return _discrete_truncated_gaussian
    elif name == "laplace":
        return _discrete_truncated_laplace
    else:
        raise ValueError("Unkown pdf method {}".format(name))


class LocationAttender(Module):
    """Location Sub-Attender.

    Args:
        query_size (int): size of the query.
        max_len (int): a maximum allowed length for the sequence to be processed
        n_steps_prepare_pos (int): number of steps during which to consider
            the positioning as in a preparation mode. During preparation mode,
            the model have less parameters to tweak, it will focus on what I thought
            were the most crucial bits. For example it will have a fix
            sigma and won't have many of the regularization term, this is to
            help it start at a decent place in a lower dimensional space, before
            going to the hard task of tweaking all at the same time.
        pdf ({"gaussian", "laplace"}, optional): name of the pdf to use to generate
            the location attention.
        Generator (Module, optional): module to generate various values. It
            should be callable using `Generator(input_size, output_size)(x)`.
            By default `nn.Linear`.
        hidden_size (int, optional): number of neurones to use in the hidden layer
            of the weighter.
        gating ({None, "residual", "highway", "custom"}, optional): Gating
            mechanism for generated values. `None` no gating. `"residual"` adds
            the new value to the previous. `"highway"` gating using convex
            combination. `"custom"` gates the previous value and add the new one.
        is_sample_attn (bool, optional): whether to sample from the attention the
            word to attend to. This will force sigma to be smaller, and might give
            better estimates of position confidence.
        sigma_kwargs (dictionary, optional): additional arguments to the
            `SigmaGenerator`.
        mu_kwargs (dictionary, optional): additional arguments to the
            `MuGenerator`.
    """

    def __init__(self, query_size,
                 max_len=50,
                 n_steps_prepare_pos=100,
                 pdf="gaussian",
                 Generator=nn.Linear,
                 hidden_size=64,
                 gating="gated_res",
                 is_sample_attn=False,  # DEV MODE
                 sigma_kwargs={},
                 mu_kwargs={}):
        super().__init__()

        self.query_size = query_size
        self.max_len = max_len
        self.n_steps_prepare_pos = n_steps_prepare_pos
        self.pdf = pdf
        self.gating = gating

        self.rel_counter = torch.arange(0, self.max_len,
                                        dtype=torch.float,
                                        device=device).unsqueeze(1) / (self.max_len - 1)

        self.weighter, self.hidden0 = get_rnn("gru", self.query_size, hidden_size,
                                              batch_first=True,
                                              is_get_hidden0=True)

        self.mu_generator = MuGenerator(hidden_size,
                                        max_len=max_len,
                                        Generator=Generator,
                                        gating=self.gating,
                                        n_steps_prepare_pos=n_steps_prepare_pos,
                                        is_add_content=True,
                                        is_add_old_attn=True,
                                        **mu_kwargs)

        self.sigma_generator = SigmaGenerator(hidden_size,
                                              n_steps_const_sigma=n_steps_prepare_pos,
                                              Generator=Generator,
                                              gating=self.gating,
                                              **sigma_kwargs)

        self.pdf = get_loc_pdf(pdf)

        if self.is_sample_attn:
            self.temperature = 0.999

        self.reset_parameters()

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["pdf"],
                              conditional_shows=["gating"])

    def forward(self, query, source_lengths, step, content_attn, attn_old):
        """Compute and return the location attention, confidence.

        Args:
            query (torch.tensor): query. Shape: (batch_size, n_queries, kq_size).
            source_lengths (tuple(list of int, torch.FloatTesnor), optional): A
                list that contains the lengths of sequences in the mini-batch. The
                Tensor has the same information but is preinitialized on teh
                correct device.
            step (int): current decoding step.
            content_attn (torch.tensor): content attention. Shape: (batch_size,
                n_queries, n_keys).
            attn_old (torch.tensor): previous general attention. Shape: (batch_size,
                n_queries, n_keys).

        Return:
            loc_attn (torch.tensor): location attention. Shape: (batch_size,
                n_queries, n_keys).
            confidence (torch.tensor): confidence of location attenton. Shape:
                (batch_size, n_queries).
        """
        batch_size, n_queries, _ = query.size()
        if n_queries != 1:
            txt = "`n_queries = {}` but only single query supported for now."
            raise NotImplementedError(txt.format(n_queries))

        source_lengths_list, source_lengths_tensor = format_source_lengths(source_lengths)

        mu, sigma = self._compute_parameters(query, step, source_lengths_tensor)

        confidence = self._sigma_to_conf(sigma)

        to_store = [x.squeeze(1) for x in [mu, sigma, confidence]]
        labels_to_store = ["mu", "sigma", "loc_confidence"]
        self.add_to_visualize(to_store, labels_to_store)
        self.add_to_test(to_store[:-1], labels_to_store[:-1])

        sigma = renormalize_input_length(sigma, source_lengths_tensor, 1)

        loc_attn = self._compute_attn(mu, sigma, source_lengths)

        if self.training and self.is_sample_attn and self.n_training_calls > self.n_steps_prepare_pos * 2:
            self.temperature = max(0.5, self.temperature**1.005)
            soft_onehot = torch.distributions.RelaxedOneHotCategorical(self.temperature,
                                                                       probs=loc_attn)
            loc_attn = soft_onehot.rsample()

        return loc_attn, confidence

    def _compute_parameters(self, weighter_inputs, step, source_lengths_tensor):
        """Compute the parameters of the positioning function.

        Return:
            mu (torch.FloatTensor): mean location of size. Shape:
                (batch_size, n_queries, 1).
            sigma (torch.FloatTensor): standard deviation of location. Shape:
                (batch_size, n_queries, 1)
        """
        if step == 0:
            batch_size = weighter_inputs.size(0)
            self.storer["weighter_hidden"] = replicate_hidden0(self.hidden0, batch_size)

        (weighter_out,
         self.storer["weighter_hidden"]) = self.weighter(weighter_inputs,
                                                         self.storer["weighter_hidden"])

        mu = self.mu_generator(weighter_out, step, source_lengths_tensor)

        sigma = self.sigma_generator(weighter_out, mu, step)

        return mu, sigma

    def _sigma_to_conf(self, sigma, min_p=0.001):
        """
        Compute the confidence given sigma.

        Note:
            - was hesitating between using the maximum of position_attention
            as confidence, and using a linear mapping from sigma to confidence.
            The former never went to 0, even though we sometimes want that.
            The latter went abrubtly to 0 very quickly and it was hard to get
            out of 0 confidence region. Decided to go with a middle ground.
            This is an approximation of maximum of position_attention, for small
            sigma but goes to 0 when large sigma.

        Shape:
            pos_confidence: `(batch_size, n_queries, 1)`
        """
        pos_confidence = torch.exp(-sigma**2 + self.sigma_generator.hard_min_sigma**2
                                   ) * (1 - min_p)

        return pos_confidence

    def _compute_attn(self, mu, sigma, source_lengths):
        """
        Compute the attention given the sufficient parameeters of the probability
        density function.

        Shape:
            pos_confidence: `(batch_size, n_queries, n_keys)`
        """
        batch_size, _, _ = mu.size()
        source_lengths_list, source_lengths_tensor = format_source_lengths(source_lengths)

        rel_counter = self.rel_counter.expand(batch_size, self.max_len, 1)
        rel_counter = renormalize_input_length(rel_counter,
                                               source_lengths_tensor - 1,
                                               self.max_len - 1)

        # slow because list comprehension : should optimize
        loc_attn = pad_sequence([self.pdf(rel_counter[i_batch, :length, :],
                                          mu[i_batch, ...].squeeze(),
                                          sigma[i_batch, ...].squeeze())
                                 for i_batch, length in enumerate(source_lengths_list)],
                                batch_first=True)

        loc_attn = loc_attn.transpose(2, 1)

        return loc_attn


class SigmaGenerator(Module):
    """
    hidden_size (int): number of neurones to use in the hidden layer
        of the weighter.
    n_steps_const_sigma(int): number of steps during which to have a constant sigma.
    Generator (Module, optional): module to generate various values. It
        should be callable using `Generator(input_size, output_size)(x)`.
    gating ({None, "residual", "highway", "custom"}, optional): Gating
        mechanism for generated values. `None` no gating. `"residual"` adds
        the new value to the previous. `"highway"` gating using convex
        combination. `"custom"` gates the previous value and add the new one.
    min_sigma (float, optional): minimum value of the standard deviation in
         order not to have division by 0. This is also important to let the network
         continue learning by always attending to multiple words. It should be
         in the range ~[0.2,1], 0.41 being a good general default. Indeed,
         the max attention you can have is 0.9073 when sigma=0.41 (i.e last
         prime value with max atttn > 0.9 ;) ).
    initial_sigma (float, optional): initial sigma the network should use. It
         should be in the range ~[2,6], 5 being a good general default. Indeed,
         it's a high sigma but up to length 50, can still have different
         attention => can learn (min when len = 50 : 1.2548e-21).
    """

    def __init__(self,
                 hidden_size,
                 n_steps_const_sigma=100,
                 Generator=nn.Linear,
                 gating="gated_res",
                 min_sigma=0.41,
                 initial_sigma=5.0):

        super().__init__()
        self.sigma_generator = Generator(hidden_size, 1)
        # start close to final annealing value => no big difference in values
        self.gate = get_gate(gating, hidden_size, 1,
                             initial_gate=0.1, save_name="sigma_gate")

        self.min_sigma = min_sigma
        self.initial_sigma = initial_sigma
        self.hard_min_sigma = self.min_sigma / 1.5  # Max attn will be 0.9975
        self.get_sigma = HyperparameterInterpolator(self.initial_sigma,
                                                    self.min_sigma * 2,
                                                    n_steps_const_sigma,
                                                    mode="linear")

        self.reset_parameters()

    def reset_parameters(self):
        """Reset and initialize the module parameters."""
        super().reset_parameters()
        self.get_sigma.reset_parameters()
        self.sigma0 = Parameter(torch.tensor(self.get_sigma.final_value))

    def extra_repr(self):
        txt = self.get_sigma.extra_repr(value_name="sigma")
        return txt + ", " + get_extra_repr(self,
                                           conditional_shows=["min_sigma",
                                                              "initial_sigma"])

    def forward(self, weighter_out, mu, step):
        """Generate the standard deviation of the location attention.

        Args:
            weighter_out (torch.FloatTensor): output of the Weighter. Shape:
                (batch_size, n_queries, hidden_size).
            step (int): decoding step.
            mu (torch.FloatTensor): mean location attention. Shape:
                (batch_size, n_queries, 1).

        Return:
            sigma (torch.FloatTensor): standard deviation of location attention.
                Shape: (batch_size, n_queries, 1).
        """
        is_update_sigma = self.training and step == 0

        current_min_sigma = self.get_sigma(is_update_sigma)
        if self.get_sigma.is_annealing:
            # if still annealing min sigma don't backprop to sigma generator
            sigma = current_min_sigma + torch.zeros_like(mu)
        else:
            sigma_old = (self.sigma0.expand_as(mu) if step == 0
                         else self.storer["sigma_old"])
            unclamped_sigma = self.gate(self.sigma_generator(weighter_out),
                                        sigma_old, weighter_out)
            sigma = clamp(unclamped_sigma,
                          minimum=self.min_sigma,
                          is_leaky=True,
                          negative_slope=0.1,
                          hard_min=self.hard_min_sigma)
            self.storer["sigma_old"] = sigma

        return sigma


class MuGenerator(Module):
    """
    hidden_size (int, optional): number of neurones to use in the hidden layer
        of the weighter.
    max_len (int, optional): a maximum allowed length for the sequence to be
        processed
    Generator (Module, optional): module to generate various values. It
        should be callable using `Generator(input_size, output_size)(x)`.
        By default `nn.Linear`.
    gating ({None, "residual", "highway", "custom"}, optional): Gating
        mechanism for generated values. `None` no gating. `"residual"` adds
        the new value to the previous. `"highway"` gating using convex
        combination. `"custom"` gates the previous value and add the new one.
    n_steps_prepare_pos (int): number of steps during which to consider
        the positioning as in a preparation mode. During preparation mode,
        the model have less parameters to tweak, it will focus on what I thought
        were the most crucial bits. For example it will have a fix
        sigma and won't have many of the regularization term, this is to
        help it start at a decent place in a lower dimensional space, before
        going to the hard task of tweaking all at the same time.
    is_l0_bb_weights (bool, optional): whether to use l0 regularisation on
        the building block weights. This is achieved by reparametrizing the
        l0 loss as done in “Learning Sparse Neural Network through L_0
        Regularisation”.
    is_reg_clamp_mu (bool, optional): whether to regularise with lp norm the
        clamping of mu. I.e push the network to not overshoot and really
        generate the desired mu rather than the clamped one. This can be
        useful as if the mu completely overshoots it will be hard for it to
        come back to normal values if it needs to. It also makes sense to
        output what you want rather than relying on postpropressing.
    is_add_content (bool, optional): whether to add the current content
        attention in the building blocks.
    is_add_old_attn (bool, optional): whether to add the previous attention in
        the building blocks.
    rounder_mu_kwargs (dictionary, optional): additional arguments to the
        rounder mu. Rounding is desirable to make the position attention
        look at the correct position even for sentences longer than it have
        ever seen.
    rounder_weights_kwargs (dictionary, optional): additional arguments to the
        rounder weights. Rounding is desirable to make the output more
        interpretable and extrapolable (as the building blocks were designed
        such that integer wights could be used to solve most positonal patterns).
    """

    def __init__(self, hidden_size,
                 max_len=50,
                 Generator=nn.Linear,
                 gating="gated_res",
                 n_steps_prepare_pos=100,
                 is_l0_bb_weights=True,
                 is_reg_clamp_mu=True,
                 is_add_content=True,
                 is_add_old_attn=True,
                 rounder_mu_kwargs={},
                 rounder_weights_kwargs={},
                 ):

        super().__init__()

        self.max_len = max_len
        self.n_steps_prepare_pos = n_steps_prepare_pos
        self.is_l0_bb_weights = is_l0_bb_weights
        self.is_reg_clamp_mu = is_reg_clamp_mu
        self.is_add_content = is_add_content
        self.is_add_old_attn = is_add_old_attn

        # Building blocks
        self.single_step = torch.tensor(1. / (self.max_len - 1)).to(device)
        self.bias = torch.tensor(1.0).to(device)
        self.bb_labels = ["mu_old", "diagonal", "single_step", "bias"]
        if self.is_add_content:
            self.bb_labels = ["mean_content_attn"] + self.bb_labels
        if self.is_add_old_attn:
            self.bb_labels = ["mean_attn_old"] + self.bb_labels
        self.n_building_blocks = len(self.bb_labels)

        self.mu_weights_generator = Generator(hidden_size, self.n_building_blocks)

        self.gate = get_gate(gating, hidden_size, self.n_building_blocks,
                             is_single_gate=False, save_name="mu_gates")

        self.rounder_weights = get_rounder(**rounder_weights_kwargs)

        self.acti_plat_int = PlateauAct(plateaus="int")

        if self.rounder_weights is None:
            self.acti_plat_diag = PlateauAct(plateaus=[-1, 1]
                                             if self.is_l0_bb_weights else [-1, 0, 1],
                                             len_plateaus=5e-1)
            self.acti_plat_bias = PlateauAct(plateaus=[-0.5, 0.5]
                                             if self.is_l0_bb_weights else [-0.5, 0, 0.5],
                                             len_plateaus=3e-1)
            self.acti_plat = PlateauAct(plateaus=[0, 1], len_plateaus=3e-1)

        if self.is_l0_bb_weights:
            self.loss_weights = torch.ones(self.n_building_blocks, device=device)
            self.loss_weights[self.bb_labels.index("single_step")] = 2.
            self.loss_weights[self.bb_labels.index("bias")] = 2.

            rounding_kwargs = dict(n_steps_interpolate=self.n_steps_prepare_pos)
            self.linear_l0_weights = L0Gates(hidden_size, self.n_building_blocks,
                                             is_at_least_1=True,
                                             initial_gates=1.,
                                             rounding_kwargs=rounding_kwargs,
                                             is_gate_old=True)  # DEV MODE

        self.rounder_mu = get_rounder(**rounder_mu_kwargs)

        if self.is_reg_clamp_mu:
            self.get_clamping_eps = HyperparameterInterpolator(0.1,
                                                               0.01,
                                                               self.n_steps_prepare_pos,
                                                               mode="linear")

        self.rel_counter = torch.arange(0, self.max_len,
                                        dtype=torch.float,
                                        device=device).unsqueeze(1) / (self.max_len - 1)

        self.reset_parameters()

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["is_l0_bb_weights", "is_reg_clamp_mu"],
                              conditional_shows=["is_add_old_attn", "is_add_content"])

    def reset_parameters(self):
        """Reset and initialize the module parameters."""
        weights = [0.25, 0.25, 0, 0.]
        self.mu0 = Parameter(torch.tensor(0.))
        if self.is_add_old_attn:
            self.mean_attn_old0 = Parameter(torch.tensor(0.))
            weights = [0.25] + weights
        if self.is_add_content:
            self.mean_content_attn0 = Parameter(torch.tensor(0.))
            weights = [0.25] + weights

        self.weights = torch.tensor(weights)
        self.old_weights0 = Parameter(self.weights)

        if self.is_reg_clamp_mu:
            self.get_clamping_eps.reset_parameters()

        super().reset_parameters()

    def forward(self, weighter_out, step, source_lengths_tensor,
                content_attn=None, attn_old=None):
        """Compute the mean of the location attention.

        Args:
            weighter_out (torch.FloatTensor): output of the Weighter of size
                (batch_size, n_queries, hidden_size).
            step (int): decoding step.
            source_lengths_tensor (torch.FloatTensor): size of each source sentence
                (batch size, ).

        Return:
            mu (torch.FloatTensor): mean location attention
                (batch_size, n_queries, 1).
        """
        batch_size, n_queries, _ = weighter_out.size()

        raw_mu_weights = self.mu_weights_generator(weighter_out) + self.weights  # DEV MODE

        mu_weights = self._transform_weights(raw_mu_weights, step, weighter_out)

        building_blocks = self._get_building_blocks(weighter_out,
                                                    source_lengths_tensor,
                                                    step,
                                                    content_attn,
                                                    attn_old)

        mu = torch.bmm(mu_weights.view(batch_size * n_queries, 1, self.n_building_blocks),
                       building_blocks.view(batch_size * n_queries, self.n_building_blocks, 1))
        mu = mu.view(batch_size, n_queries, 1)

        mu = self._transform_mu(mu, source_lengths_tensor, step)

        self.storer["mu_old"] = mu
        self.add_to_test([mu_weights, raw_mu_weights], ['mu_weights', 'raw_mu_weights'])
        self.add_to_visualize([mu_weights], ['mu_weights'])

        # set back to mu in [0,1]
        mu = mu + 0.5

        return mu

    def _transform_weights(self, mu_weights, step, weighter_out):
        """Transforms the building block weights.

        Return:
            mu_weights (torch.FloatTensor): building blocks weights. Shape:
                (batch_size, n_queries, n_building_blocks).
        """
        # gate
        mu_weights_old = (self.old_weights0.expand_as(mu_weights)
                          if step == 0 else self.storer["mu_weights"])
        mu_weights = self.gate(mu_weights, mu_weights_old, weighter_out)

        # plateau activation or rounding
        dict_mu_weights = dict(zip(self.bb_labels, mu_weights.unbind(-1)))

        if self.rounder_weights is not None:  # DEV MODE
            assert self.is_l0_bb_weights
            for i, l in enumerate(self.bb_labels):
                is_update = (self.training and step == 0 and i == 0)
                if l == "diagonal":
                    dict_mu_weights[l] = clamp(dict_mu_weights[l], minimum=-1., maximum=1)
                    dict_mu_weights[l] = self.rounder_weights((dict_mu_weights[l] + 1) / 2,
                                                              is_update=is_update) * 2 - 1
                elif l == "single_step":
                    dict_mu_weights[l] = self.rounder_weights(dict_mu_weights[l],
                                                              is_update=is_update)
                elif l == "bias":
                    dict_mu_weights[l] = clamp(dict_mu_weights[l], minimum=-.5, maximum=.5)
                    dict_mu_weights[l] = self.rounder_weights(dict_mu_weights[l] + .5,
                                                              is_update=is_update) - .5
                elif l in ["mu_old", "mean_content_attn", "mean_attn_old"]:
                    dict_mu_weights[l] = clamp(dict_mu_weights[l], minimum=0., maximum=1.)
                    dict_mu_weights[l] = self.rounder_weights(dict_mu_weights[l],
                                                              is_update=is_update)
                else:
                    raise ValueError("Unkown label={}".format(l))
        else:
            for l in self.bb_labels:
                if l == "diagonal":
                    dict_mu_weights[l] = self.acti_plat_diag(dict_mu_weights[l])
                elif l == "single_step":
                    dict_mu_weights[l] = self.acti_plat_int(dict_mu_weights[l])
                elif l == "bias":
                    dict_mu_weights[l] = self.acti_plat_bias(dict_mu_weights[l])
                elif l in ["mu_old", "mean_content_attn", "mean_attn_old"]:
                    dict_mu_weights[l] = self.acti_plat(dict_mu_weights[l])
                else:
                    raise ValueError("Unkown label={}".format(l))

        # regularizes single step
        if self.is_regularize:
            loss = batch_reduction_f(regularization_loss(dict_mu_weights["single_step"],
                                                         p=1,
                                                         dim=-1,
                                                         min_x=1.),
                                     torch.mean)
            self.add_regularization_loss("pos_const_weights", loss)

        ordered_weights = [dict_mu_weights[l] for l in self.bb_labels]
        mu_weights = torch.stack(ordered_weights, dim=2)

        # l0
        if self.is_l0_bb_weights:
            # when using l0 gating no need of having weight for cont / total_attn/
            # pos because will be either 1 or 0 (gated)
            remove = torch.ones_like(mu_weights)
            remove[:, :, :-3] = 0.
            mu_weights = mu_weights * remove + 1 - remove

            gates, loss = self.linear_l0_weights(weighter_out,
                                                 is_reset=step == 0,
                                                 loss_weights=self.loss_weights)
            if self.is_regularize:
                self.add_regularization_loss("pos_l0_weights", loss)

            self.add_to_visualize(gates.sum(-1).mean(), "bb_gates")
            self.add_to_test([mu_weights, gates], ['ungated_weights', "bb_gates"])

            mu_weights = (mu_weights * gates)

        self.storer["mu_weights"] = mu_weights

        return mu_weights

    def _transform_mu(self, mu, source_lengths_tensor, step):
        """Postpreocessing of `mu`.

        Return:
            mu (torch.FloatTensor): mean location attention. Shape:
                (batch_size, n_queries, 1).
        """
        is_update = self.training and step == 0

        normalizer = (source_lengths_tensor - 1).unsqueeze(1).unsqueeze(1)
        if self.rounder_mu is not None:
            # rounding to words
            mu = self.rounder_mu((mu + 0.5) * normalizer,
                                 is_update=is_update
                                 ) / normalizer - 0.5
        else:  # DEV MODE
            # plateau activation to words
            mu = self.acti_plat_int((mu + 0.5) * normalizer) / normalizer - 0.5

        if self.is_regularize and self.is_reg_clamp_mu:
            eps = self.get_clamping_eps(is_update)
            delta = 0.01
            mu, loss = clamp_regularize(mu,
                                        negative_slope=0.1,
                                        minimum=-(0.5 + delta),
                                        maximum=0.5 + delta,
                                        is_leaky=True,
                                        reg_kwargs=dict(p=1, min_x=eps))
            self.add_regularization_loss("pos_clamp_mu", loss)
        else:
            mu = clamp(mu,
                       minimum=-0.5,
                       maximum=0.5,
                       is_leaky=True)

        return mu

    def _get_building_blocks(self, query, source_lengths_tensor, step,
                             content_attn=None, attn_old=None):
        """Get the inputs and the buillding blocks for location generation.

        Return:
            building_blocks (torch.FloatTensor): building blocks of size
                (batch_size, n_queries, n_building_blocks)
        """
        batch_size, n_queries, _ = query.size()

        unormalized_counter = self.rel_counter.expand(batch_size, self.max_len, 1)
        rel_counter = renormalize_input_length(unormalized_counter,
                                               source_lengths_tensor - 1,
                                               self.max_len - 1)

        mean_attn_old = None
        mean_content_attn = None
        if step == 0:
            mu_old = self.mu0.expand(batch_size, n_queries, 1)
            if self.is_add_old_attn:
                mean_attn_old = self.mean_attn_old0.expand(batch_size,
                                                           n_queries, 1)
            if self.is_add_content:
                mean_content_attn = self.mean_content_attn0.expand(batch_size,
                                                                   n_queries, 1)
        else:
            mu_old = self.storer["mu_old"]
            if self.is_add_old_attn:
                mean_attn_old = torch.bmm(attn_old,
                                          rel_counter[:, :attn_old.size(2), :]
                                          )
            if self.is_add_content:
                mean_content_attn = torch.bmm(content_attn,
                                              rel_counter[:, :content_attn.size(2), :]
                                              )

        # t/n
        diagonal = rel_counter[:, step:step + n_queries, :]

        single_step = self.single_step.expand(batch_size, n_queries, 1)
        single_step = renormalize_input_length(single_step,
                                               source_lengths_tensor - 1,
                                               self.max_len - 1)

        bias = self.bias.expand(batch_size, n_queries, 1)

        building_blocks = dict(mean_attn_old=mean_attn_old,
                               diagonal=diagonal,
                               mean_content_attn=mean_content_attn,
                               bias=bias,
                               mu_old=mu_old,
                               single_step=single_step)

        # set mu 0 to be middle
        # not mu_old as already correct
        for l in self.bb_labels:
            if l in ["diagonal", "mean_attn_old", "mean_content_attn"]:
                building_blocks[l] = building_blocks[l] - 0.5

        ordered_blocks = [building_blocks[l] for l in self.bb_labels]
        building_blocks = torch.cat(ordered_blocks, dim=2)

        self.add_to_visualize(building_blocks.squeeze(1), "building_blocks")
        self.add_to_test(building_blocks.squeeze(1), "building_blocks")

        return building_blocks
