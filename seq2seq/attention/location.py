"""
Positioning attention.

TO DO:
    - most important part of `spattn`

Contact: Yann Dubois
"""
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

from seq2seq.util.helpers import (renormalize_input_length, get_rnn, get_extra_repr,
                                  clamp, format_source_lengths, leaky_noisy_clamp,
                                  clamp_regularize, HyperparameterInterpolator,
                                  HyperparameterCurriculumInterpolator, get_indices,
                                  regularization_loss, batch_reduction_f, inv_sigmoid,
                                  identity)
from seq2seq.util.torchextend import (get_rounder, get_gate, PlateauAct,
                                      ProbabilityConverter, GaussianNoise)
from seq2seq.util.initialization import replicate_hidden0, init_param
from seq2seq.util.base import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_regularizers_location(total_training_calls, n_steps_prepare_pos):
    def _initialize_regularizer(name, curriculum, **kwargs):
        max_p_interpolators[name] = HyperparameterCurriculumInterpolator(curriculum, **kwargs)

    max_p_interpolators = dict()

    _initialize_regularizer("pos_clamp_mu", [dict(step=0, value=1e-2)])

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
        max_len (int, optional): a maximum allowed length for the sequence to be processed
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
        gating ({None, "residual", "highway", "gates_res"}, optional): Gating
            mechanism for generated values. `None` no gating. `"residual"` adds
            the new value to the previous. `"highway"` gating using convex
            combination. `"gates_res"` gates the previous value and add the new one.
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
                 pretrained_locator=None,  # DEV MODE
                 sigma_kwargs={},
                 mu_kwargs={},
                 is_recurrent=True,  # DEV MODE
                 ):
        super().__init__()

        self.query_size = query_size
        self.max_len = max_len
        self.n_steps_prepare_pos = n_steps_prepare_pos
        self.pdf = pdf
        self.gating = gating
        self.pretrained_locator = pretrained_locator

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
                                        **mu_kwargs)

        self.sigma_generator = SigmaGenerator(hidden_size,
                                              n_steps_const_sigma=n_steps_prepare_pos,
                                              Generator=Generator,
                                              gating=self.gating,
                                              **sigma_kwargs)

        self.pdf = get_loc_pdf(pdf)

        self.reset_parameters()

        if self.pretrained_locator is not None:
            self.load_locator(self.pretrained_locator)

    def reset_parameters(self):
        """Reset and initialize the module parameters."""
        if self.pretrained_locator is None:
            # only reset param if not pretrained
            super().reset_parameters()

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["pdf"],
                              conditional_shows=["gating", "pretrained_locator"])

    def load_locator(self, file):
        """
        Loads a pretrained locator (output from self.save_locator) for transfer
        learning.
        """
        locator = torch.load(file)
        self.weighter.load_state_dict(locator["weighter"])
        self.mu_generator.load_state_dict(locator["mu_generator"])
        self.sigma_generator.load_state_dict(locator["sigma_generator"])
        self.hidden0 = locator["hidden0"]
        self.pdf = locator["pdf"]

    def save_locator(self, file):
        """Save the pretrained locator to a file."""
        locator = dict(weighter=self.weighter.state_dict(),
                       hidden0=self.hidden0,
                       mu_generator=self.mu_generator.state_dict(),
                       sigma_generator=self.sigma_generator.state_dict(),
                       pdf=self.pdf)
        torch.save(locator, file)

    def forward(self, query, source_lengths, step, attn_old, provided_attention=None):
        """Compute and return the location attention, confidence.

        Args:
            query (torch.tensor): query. Shape: (batch_size, n_queries, kq_size).
            source_lengths (tuple(list of int, torch.FloatTesnor), optional): A
                list that contains the lengths of sequences in the mini-batch. The
                Tensor has the same information but is preinitialized on teh
                correct device.
            step (int): current decoding step.
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

        mu, sigma = self._compute_parameters(query, step, source_lengths_tensor,
                                             attn_old, provided_attention)

        confidence = self._sigma_to_conf(sigma).squeeze(-1)

        to_store = [x.squeeze(-1) for x in [mu, sigma]] + [confidence]
        labels_to_store = ["mu", "sigma", "loc_confidence"]
        self.add_to_visualize(to_store, labels_to_store)
        self.add_to_test(to_store[:-1], labels_to_store[:-1])

        sigma = renormalize_input_length(sigma, source_lengths_tensor, 1)

        loc_attn = self._compute_attn(mu, sigma, source_lengths)

        return loc_attn, confidence

    def _compute_parameters(self, weighter_inputs, step, source_lengths_tensor,
                            attn_old, provided_attention):
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

        mu = self.mu_generator(weighter_out, step, source_lengths_tensor,
                               attn_old, provided_attention)

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
    gating ({None, "residual", "highway", "gates_res"}, optional): Gating
        mechanism for generated values. `None` no gating. `"residual"` adds
        the new value to the previous. `"highway"` gating using convex
        combination. `"gates_res"` gates the previous value and add the new one.
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
    kwargs:
        Additional arguments to the Generator.
    """

    def __init__(self,
                 hidden_size,
                 n_steps_const_sigma=100,
                 Generator=nn.Linear,
                 gating="gated_res",
                 min_sigma=0.41,
                 initial_sigma=5.0,
                 max_sigma=7.0,
                 is_force_sigma=False,  # DEV MODE
                 **kwargs):

        super().__init__()
        self.sigma_generator = Generator(hidden_size, 1, **kwargs)
        # start close to final annealing value => no big difference in values
        self.gate = get_gate(gating, hidden_size, 1,
                             initial_gate=0.1, save_name="sigma_gate")

        self.is_force_sigma = is_force_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
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

        # run it before updating
        current_min_sigma = self.get_sigma(is_update_sigma)
        is_still_annealing = self.get_sigma.is_annealing
        if self.get_sigma.is_annealing or self.is_force_sigma:
            # if still annealing min sigma don't backprop to sigma generator
            sigma = current_min_sigma + torch.zeros_like(mu)
            if not is_still_annealing:
                sigma = sigma / 2
        else:
            # USE EXPONENTIAL TO MAKE SIGMA POSITIVE!!
            raw_sigma_old = (self.sigma0.expand_as(mu) if step == 0
                             else self.storer["raw_sigma_old"])
            raw_sigma = self.gate(self.sigma_generator(weighter_out),
                                  raw_sigma_old, weighter_out)
            self.storer["raw_sigma_old"] = raw_sigma
            sigma = clamp(raw_sigma,
                          minimum=self.min_sigma,
                          maximum=self.max_sigma,
                          is_leaky=True,
                          negative_slope=0.1,
                          hard_min=self.hard_min_sigma)

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
    gating ({None, "residual", "highway", "gates_res"}, optional): Gating
        mechanism for generated values. `None` no gating. `"residual"` adds
        the new value to the previous. `"highway"` gating using convex
        combination. `"gates_res"` gates the previous value and add the new one.
    n_steps_prepare_pos (int): number of steps during which to consider
        the positioning as in a preparation mode. During preparation mode,
        the model have less parameters to tweak, it will focus on what I thought
        were the most crucial bits. For example it will have a fix
        sigma and won't have many of the regularization term, this is to
        help it start at a decent place in a lower dimensional space, before
        going to the hard task of tweaking all at the same time.
    is_reg_clamp_mu (bool, optional): whether to regularise with lp norm the
        clamping of mu. I.e push the network to not overshoot and really
        generate the desired mu rather than the clamped one. This can be
        useful as if the mu completely overshoots it will be hard for it to
        come back to normal values if it needs to. It also makes sense to
        output what you want rather than relying on postpropressing.
    clipping_step (int, optional): maximum single step that can be taken by the
        network this has the same interpretation of k in "Self-Attention with
        Relative Position Representations" where they use a default of 2. If
        `None` then no clipping is used.
    weight_bias (dictionary, optional): initial building block weights.
    rounder_mu_kwargs (dictionary, optional): additional arguments to the
        rounder mu. Rounding is desirable to make the position attention
        look at the correct position even for sentences longer than it have
        ever seen.
    force_mu ({None, "bb", "bb-zeroW", "all"}, optional): If `"bb"` add a
        provided attention as building block, it still has to learn the
        correct weights. If `"bb-zeroW"` it zeros out the weights of all othet
        building blocks in addition to adding the provided attention as building
        block, the network still has to learn to use a weight of 1 for this bb.
        `"all"` sets both the correct building block and weights, the network
        will have access to the correct mu but still has to learn sigma.  `
        None` doesn't force mu.
    kwargs:
        Additional arguments to the Generator.
    """

    def __init__(self, hidden_size,
                 max_len=50,
                 Generator=nn.Linear,
                 gating="gated_res",
                 n_steps_prepare_pos=100,
                 is_reg_clamp_mu=True,
                 clipping_step=3,
                 perc_attn_old=0.7,
                 weight_bias=dict(perc_attn_old=0.,  # weight_bias["perc_attn_old"] = 0
                                  single_step=1.,  # start with half forward step
                                  bias=0.1,  # start looking at begining
                                  ),
                 weight_factor=dict(perc_attn_old=.5,
                                    single_step=.5,
                                    bias=.5),  # make slow changes
                 rounder_mu_kwargs={},
                 rounder_perc_kwargs={},
                 is_plateau_step=True,
                 is_plateau_bias=True,
                 mu_noise=0.,
                 **kwargs
                 ):

        super().__init__()

        self.max_len = max_len
        self.n_steps_prepare_pos = n_steps_prepare_pos
        self.is_reg_clamp_mu = is_reg_clamp_mu
        self.clipping_step = clipping_step
        self.is_plateau_step = is_plateau_step
        self.is_plateau_bias = is_plateau_bias
        self.perc_attn_old = perc_attn_old

        # Building blocks
        self.single_step = torch.tensor(1. / (self.max_len - 1)).to(device)
        self.bb_labels = ["single_step", "perc_attn_old", "bias"]
        # make the mu weights change relatively little at the begining
        # as small mu weight makes a big difference
        self.weight_bias = weight_bias
        self.weight_factor = weight_factor

        self.mu_weights_generator = Generator(hidden_size, len(self.bb_labels), **kwargs)
        self.gaussian_noise = GaussianNoise(is_relative_sigma=False, sigma=mu_noise)

        self.to_proba = ProbabilityConverter(initial_probability=perc_attn_old)
        self.rounder_perc = get_rounder(**rounder_perc_kwargs)

        #plateaus_step = list(range(-self.clipping_step, self.clipping_step + 1))
        #self.acti_plat_step = PlateauAct(plateaus_step) if self.is_plateau_step else identity
        self.acti_plat_step = PlateauAct([-1, 1]) if self.is_plateau_step else identity
        self.acti_plat_bias = PlateauAct([0, 1]) if self.is_plateau_bias else identity
        self.weight_bias = torch.tensor([self.weight_bias[l] for l in self.bb_labels],
                                        device=device, dtype=torch.float)
        self.weight_factor = torch.tensor([self.weight_factor[l] for l in self.bb_labels],
                                          device=device, dtype=torch.float)

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
                              always_shows=["is_reg_clamp_mu", "is_plateau_step", "is_plateau_bias"],
                              conditional_shows=[])

    def reset_parameters(self):
        """Reset and initialize the module parameters."""
        if self.is_reg_clamp_mu:
            self.get_clamping_eps.reset_parameters()

        super().reset_parameters()

    def forward(self, weighter_out, step, source_lengths_tensor, attn_old,
                provided_attention=None):
        """Compute the mean of the location attention.

        Args:
            weighter_out (torch.FloatTensor): output of the Weighter of size
                (batch_size, n_queries, hidden_size).
            step (int): decoding step.
            source_lengths_tensor (torch.FloatTensor): size of each source sentence
                (batch size, ).
            attn_old (torch.FloatTensor): location attention. Shape: (batch_size,
                n_queries, n_keys-1). -1 because don't attend to <sos>.

        Return:
            mu (torch.FloatTensor): mean location attention
                (batch_size, n_queries, 1).
        """
        batch_size, n_queries, _ = weighter_out.size()

        raw_mu_weights = (self.gaussian_noise(self.mu_weights_generator(weighter_out)) *
                          self.weight_factor) + self.weight_bias
        self.storer["raw_mu_weights"] = raw_mu_weights

        w_single_step, perc_attn_old, bias = raw_mu_weights.unbind(-1)
        perc_attn_old = self.to_proba(perc_attn_old)
        bias = self.acti_plat_bias(bias)

        if self.n_training_calls >= self.n_steps_prepare_pos:
            w_single_step = self.acti_plat_step(w_single_step)
            if self.rounder_perc is not None:
                perc_attn_old = self.rounder_perc(perc_attn_old)

        unormalized_counter = self.rel_counter.expand(batch_size, self.max_len, 1)
        rel_counter = renormalize_input_length(unormalized_counter,
                                               source_lengths_tensor - 1,
                                               self.max_len - 1)

        if step == 0:
            perc_attn_old = perc_attn_old * 0  # for plotting
            mean_attn_old = bias
            w_single_step = w_single_step * 0  # begining only look at bias
            start = bias  # no old attention at first step
        else:
            mean_attn_old = torch.bmm(attn_old,
                                      rel_counter[:, :attn_old.size(2), :]).squeeze(-1)
            start = perc_attn_old * mean_attn_old + (1 - perc_attn_old) * bias

        single_step = self.single_step.expand(batch_size, n_queries, 1)
        single_step = renormalize_input_length(single_step,
                                               source_lengths_tensor - 1,
                                               self.max_len - 1)

        mu = start.unsqueeze(-1) + w_single_step.unsqueeze(-1) * single_step

        self.add_to_visualize([single_step.squeeze(1), mean_attn_old, w_single_step,
                               bias, perc_attn_old],
                              ["single_step", "mean_attn_old", "w_single_step",
                               "bias", "perc_attn_old"])
        self.add_to_test([mean_attn_old, w_single_step,
                          bias, perc_attn_old],
                         ["mean_attn_old", "w_single_step",
                          "bias", "perc_attn_old"])

        mu = self._transform_mu(mu, source_lengths_tensor, step)

        return mu

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
            mu = self.rounder_mu(mu * normalizer,
                                 is_update=is_update
                                 ) / normalizer

        if self.is_regularize and self.is_reg_clamp_mu:
            eps = self.get_clamping_eps(is_update)
            delta = 0.01
            mu, loss = clamp_regularize(mu,
                                        negative_slope=0.1,
                                        minimum=-delta,
                                        maximum=1 + delta,
                                        is_leaky=True,
                                        reg_kwargs=dict(p=1, min_x=eps))
            self.add_regularization_loss("pos_clamp_mu", loss)
        else:
            mu = clamp(mu,
                       minimum=0.,
                       maximum=1.,
                       is_leaky=True)

        return mu
