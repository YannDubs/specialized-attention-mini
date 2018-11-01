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

from seq2seq.util.helpers import (renormalize_input_length, get_rnn,
                                  HyperparameterInterpolator, get_extra_repr,
                                  clamp, format_source_lengths, Rate2Steps,
                                  get_indices, regularization_loss, batch_reduction_f,
                                  clamp_regularize, HyperparameterCurriculumInterpolator,
                                  mean)
from seq2seq.util.torchextend import (MLP, ConcreteRounding, ProbabilityConverter,
                                      AnnealedGaussianNoise, L0Gates, StochasticRounding)
from seq2seq.util.initialization import replicate_hidden0, init_param, weights_init
from seq2seq.util.base import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_regularizers_positioner(total_training_calls, n_steps_prepare_pos):
    def _initialize_regularizer(name, curriculum, **kwargs):
        max_p_interpolators[name] = HyperparameterCurriculumInterpolator(curriculum, **kwargs)

    max_p_interpolators = dict()

    _initialize_regularizer("pos_const_weights",
                            [dict(step=n_steps_prepare_pos, value=5e-2, mode="geometric"),
                             dict(step=int(n_steps_prepare_pos * 3 / 2), value=1e-4, mode="geometric"),
                             dict(step=n_steps_prepare_pos * 2, value=1e-2)])

    _initialize_regularizer("pos_old_weights",
                            [dict(step=0, value=5e-2, mode="linear"),
                             dict(step=n_steps_prepare_pos, value=0)])

    _initialize_regularizer("pos_clamp_weights",
                            [dict(step=0, value=5e-3, mode="linear"),
                             dict(step=n_steps_prepare_pos, value=5e-2)])

    _initialize_regularizer("pos_l0_weights",
                            [dict(step=n_steps_prepare_pos * 3, value=0, mode="linear"),
                             dict(step=n_steps_prepare_pos * 6, value=3e-2, mode="geometric"),
                             dict(step=n_steps_prepare_pos * 9, value=1e-2)])

    _initialize_regularizer("pos_variance_weights",
                            [dict(step=n_steps_prepare_pos, value=5e-3, mode="linear"),
                             dict(step=n_steps_prepare_pos * 2, value=0)])

    _initialize_regularizer("pos_clamp_mu",
                            [dict(step=0, value=5e-3, mode="linear"),
                             dict(step=n_steps_prepare_pos, value=5e-2)])

    # don't use a name starting with `pos_%` because later I look for losses
    # starting with `pos_` and it means it only apply to positioing attention
    # while this ones applies to mixing the attention
    _initialize_regularizer("mix_%_pos",
                            [dict(step=0, value=0, mode="linear"),
                             dict(step=n_steps_prepare_pos, value=5e-2, mode="geometric"),
                             dict(step=n_steps_prepare_pos * 3, value=5e-3)])

    return max_p_interpolators


def _discrete_truncated_gaussian(x, mu, sigma):
    """Return normalized Gaussian_pdf(x)."""
    x = torch.exp(-(x - mu)**2 / (2 * sigma**2))
    x = F.normalize(x, p=1, dim=0)
    return x


def _discrete_truncated_laplace(x, mu, b):
    """Return normalized Laplacian_pdf(x)."""
    x = torch.exp(-1 * torch.abs((x - mu) / b))
    x = F.normalize(x, p=1, dim=0)
    return x


def _get_positioner(name):
    """Get the correct positioner method."""
    if name == "gaussian":
        return _discrete_truncated_gaussian
    elif name == "laplace":
        return _discrete_truncated_laplace
    else:
        raise ValueError("Unkown positioner method {}".format(name))


def _get_rounder(name=None, **kwargs):
    if name is None:
        return None
    elif name == "concrete":
        return ConcreteRounding(**kwargs)
    elif name == "stochastic":
        return StochasticRounding(**kwargs)
    else:
        raise ValueError("Unkown rounder method {}".format(name))


class PositionAttention(Module):
    """Position Attention Generator.

    Args:
        decoder_output_size (int): size of the hidden activations of the decoder.
        max_len (int): a maximum allowed length for the sequence to be processed
        n_steps_prepare_pos (int): number of steps during which to consider
            the positioning as in a preparation mode. During preparation mode,
            the model have less parameters to tweak, it will focus on what I thought
            were the most crucial bits. For example it will have a fix
            sigma and won't have many of the regularization term, this is to
            help it start at a decent place in a lower dimensional space, before
            going to the hard task of tweaking all at the same time.
        n_steps_init_help (int, optional): number of training steps for which to
            us an initializer helper for the position attention. Currently the helper
            consists of alternating between values of 0.5 and -0.5 for the
            "rel_counter_decoder" weights.
        positioning_method (str, optional): name of the psotioner function to use
        is_mlps (bool, optional): whether to use MLP's instead of linear function
            for the weight generators.
        hidden_size (int, optional): number of neurones to use in hidden layers.
        is_recurrent (bool, optional) whether to use a rnn.
        rnn_cell (str, optional): type of RNN cell
        is_bb_bias (bool, optional): adding a bias term to the building blocks.
            THis has the advantage of letting the network go to absolut positions
            (ex: middle, end, ...). THe disadvantage being that the model will often
            stop using other building blocks and thus be less general.
        is_content_attn (bool, optional): whether you are using content attention.
        is_reg_const_weights (bool, optional): whether to use a lp regularization
            on the constant position mu building block. This can be usefull in
            otrder to push the network to use non constant building blocks that are
            more extrapolable (i.e with constants, the network has to make varying
            weights which is not interpretable. If the blocks ae varying then
            the "hard" extrapolable output would already be done for the network).
        is_reg_old_weights (bool, optional): whether to use a lp norm regularisation
            on the building blocks that depend on previous positioning attention.
            This can be useful as these building blocks cannot be used correctly
            before positioning attention actually converged.
        is_reg_clamp_mu (bool, optional): whether to regularise with lp norm the
            clamping of mu. I.e push the network to not overshoot and really
            generate the desired mu rather than the clamped one. This can be
            useful as if the mu completely overshoots it will be hard for it to
            come back to normal values if it needs to. It also makes sense to
            output what you want rather than relying on postpropressing.
        is_reg_variance_weights (bool, optional): whether to use lp norm
            regularisation to force the building blocks to have low variance
            across time steps. This can be useful as it forces the model to use
            simpler weight patterns that are more extrapolable. For example it
            would prefer giving a weight of `1` to `block_j/n`than using a weight
            of `j` to `block_1/n`.
        is_l0_bb_weights (bool, optional): whether to use l0 regularisation on
            the building block weights. This is achieved by reparametrizing the
            l0 loss as done in “Learning Sparse Neural Network through L_0
            Regularisation”.
        is_clamp_weights (bool, optional): whether to clamp the building block
            weights on some meaningful intervals.
         rounder_weights_kwargs (dictionary, optional): additional arguments to the
            rounder weights. Rounding is desirable to make the output more
            interpretable and extrapolable (as the building blocks were designed
            such that integer wights could be used to solve most positonal patterns).
         rounder_mu_kwargs (dictionary, optional): additional arguments to the
            rounder mu. Rounding is desirable to make the position attention
            look at the correct position even for sentences longer than it have
            ever seen.
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
        n_steps_interpolate_min_sigma (int, optional): if not 0 , it will force
            the network to keep a higher sigma while it's learning. min_sigma will
            actually start at `initial_sigma` and will linearly decrease at each
            training calls, until it reaches the given `min_sigma`. This parameter
            defines the number of training calls before the mdoel should reach
            the final `min_sigma`. As a rule of thumb it should be a percentage
            (example 0.1*n_epochs) of the total number of training calls so that
            the network can have some time to train with the final min_sigma.
        lp_reg_weights (bool, optional): the p in the lp norm to use for all the
            regularisation above. p can be in [0,”inf”]. If `p=0` will use some
            approximation to the l0 norm.
    """

    def __init__(self, decoder_output_size, max_len, n_steps_prepare_pos,
                 n_steps_init_help=0,
                 positioning_method="gaussian",
                 is_mlps=True,
                 hidden_size=32,
                 is_recurrent=True,
                 rnn_cell="gru",
                 rnn_kwargs={},
                 is_bb_bias=True,  # TO DO: remove this parameter (i.e force True)
                 is_content_attn=True,
                 regularizations=["is_reg_clamp_mu", "is_l0_bb_weights"],
                 is_clamp_weights=True,  # TO DO - medium: chose best and remove parameter
                 rounder_weights_kwargs={},
                 rounder_mu_kwargs={},
                 n_steps_interpolate_min_sigma=0,
                 min_sigma=0.41,
                 initial_sigma=5.0,
                 lp_reg_weights=1,
                 is_sample_attn=False):  # TO DOC
        super(PositionAttention, self).__init__()

        self.n_steps_prepare_pos = n_steps_prepare_pos
        self.n_steps_init_help = n_steps_init_help
        self.positioning_method = positioning_method
        self.is_content_attn = is_content_attn
        self.regularizations = regularizations
        self.is_l0_bb_weights = "is_l0_bb_weights" in regularizations
        self.lp_reg_weights = lp_reg_weights
        self.is_clamp_weights = is_clamp_weights
        self.is_sample_attn = is_sample_attn

        n_additional_mu_input = 9 - int(not self.is_content_attn)

        input_size = decoder_output_size + n_additional_mu_input
        self.is_recurrent = is_recurrent
        self.positioner = _get_positioner(self.positioning_method)
        self.min_sigma = min_sigma
        self.hard_min_sigma = self.min_sigma / 1.5  # Max mu will be 0.9975
        self.initial_sigma = initial_sigma
        self.get_sigma = HyperparameterInterpolator(self.initial_sigma,
                                                    self.min_sigma * 2,
                                                    self.n_steps_prepare_pos,
                                                    mode="linear")

        self.max_len = max_len

        # Building blocks
        self.is_bb_bias = is_bb_bias
        self.single_step = torch.tensor(1. / (self.max_len - 1)).to(device)
        self.rel_counter = torch.arange(0, self.max_len,
                                        dtype=torch.float,
                                        device=device).unsqueeze(1) / (self.max_len - 1)

        self.bb_labels = ["mean_attn_old",
                          "rel_counter_decoder",
                          "single_step",
                          "mu_old"]  # TO DO: should use dictionnary instead

        # indicate what expected value the weights should be initalized to if it
        # they were on their own.
        self.expected_mu_weights_init = [.5, .5, .5, .5]

        if self.is_content_attn:
            self.bb_labels += ["mean_content_old"]
            self.expected_mu_weights_init += [.5]

        if self.is_bb_bias:
            self.bias = torch.tensor(1.0).to(device)
            self.bb_labels += ["bias"]
            self.expected_mu_weights_init += [.5]

        n_building_blocks_mu = len(self.bb_labels)

        self.expected_mu_weights_init = torch.tensor(self.expected_mu_weights_init,
                                                     dtype=torch.float,
                                                     device=device) / n_building_blocks_mu

        if self.is_recurrent:
            self.rnn, self.hidden0 = get_rnn(rnn_cell, input_size, hidden_size,
                                             batch_first=True,
                                             is_get_hidden0=True,
                                             **rnn_kwargs)
            self.mu_weights_generator = nn.Linear(hidden_size,
                                                  n_building_blocks_mu)

            self.sigma_generator = nn.Linear(hidden_size, 1)
        else:
            if is_mlps:
                self.mu_weights_generator = MLP(input_size,
                                                hidden_size,
                                                n_building_blocks_mu)

                self.sigma_generator = MLP(input_size,
                                           hidden_size // 2,
                                           1)
            else:
                self.mu_weights_generator = nn.Linear(input_size,
                                                      n_building_blocks_mu)

                self.sigma_generator = nn.Linear(input_size, 1)

        self.rounder_weights = _get_rounder(**rounder_weights_kwargs)
        self.rounder_mu = _get_rounder(**rounder_mu_kwargs)

        if self.is_l0_bb_weights:
            rounding_kwargs = dict(n_steps_interpolate=self.n_steps_prepare_pos)
            self.linear_l0_weights = L0Gates(hidden_size, n_building_blocks_mu,
                                             is_at_least_1=True,
                                             initial_gates=1,
                                             rounding_kwargs=rounding_kwargs)

        self.mean_attns_discounting_factor = Parameter(torch.tensor(0.0))

        if self.is_sample_attn:
            self.temperature = 0.999

        self.reset_parameters()

    def reset_parameters(self):
        """Reset and initialize the module parameters."""
        super().reset_parameters()

        self.get_sigma.reset_parameters()

        # could start at 0 if want to bias to start reading from the begining
        self.mu0 = Parameter(torch.tensor(0.5))

        self.sigma0 = Parameter(torch.tensor(self.get_sigma.final_value))

        init_param(self.mean_attns_discounting_factor, is_positive=True)

    def extra_repr(self):
        txt = self.get_sigma.extra_repr(value_name="sigma")
        return txt + ", " + get_extra_repr(self,
                                           always_shows=["regularizations"],
                                           conditional_shows=["positioning_method",
                                                              "is_bb_bias",
                                                              "min_sigma",
                                                              "is_clamp_weights",
                                                              "initial_sigma",
                                                              "is_sample_attn"])

    def forward(self,
                decoder_outputs,
                source_lengths,
                step,
                mu_old,
                sigma_old,
                mean_content_old,
                mean_attn_old,
                mean_attn_olds,
                additional):
        """Compute and return the positional attention, confidence, parameters.

        Args:
            decoder_outputs (torch.tensor): tensor of size (batch_size, n_steps,
                hidden_size) containing the hidden activations of the coder.
            source_lengths (tuple(list of int, torch.FloatTesnor), optional): A
                list that contains the lengths of sequences in the mini-batch. The
                Tensor has the same information but is preinitialized on teh
                correct device.
            step (int): current decoding step.
            mu_old (torch.tensor): tensor of size (batch_size, n_steps, 1)
                containing last means of the positional attention. `None` for
                step == 0.
            sigma_old (torch.tensor): tensor of size (batch_size, n_steps, 1)
                containing last standard deviations of the positional attention.
                `None` for step == 0.
            mean_content_old (torch.tensor): tensor of size (batch_size, n_steps)
                containing the mean position of the last attention.
            mean_attn_old (torch.tensor): tensor of size (batch_size, n_steps)
                containing the mean position of the last content attention.
            mean_attn_olds (torch.tensor): tensor of size (batch_size, n_steps)
                containing the (dscounted) mean mu across all previous time steps.
            additional (dictionary): dictionary containing additional variables
                that are necessary for some hyperparamets.
        """
        batch_size, max_source_lengths, _ = decoder_outputs.size()
        source_lengths_list, source_lengths_tensor = format_source_lengths(source_lengths)

        positioning_inputs, building_blocks, additional = self._get_features(decoder_outputs,
                                                                             source_lengths_tensor,
                                                                             step,
                                                                             mu_old,
                                                                             sigma_old,
                                                                             mean_content_old,
                                                                             mean_attn_old,
                                                                             additional)

        mu, sigma, additional = self._compute_parameters(positioning_inputs,
                                                         building_blocks,
                                                         step,
                                                         source_lengths_tensor,
                                                         additional)

        pos_confidence = self._sigma_to_conf(sigma)

        # need to take relative sigma after sigma_to_conf because not fair if
        # the confidence depends on the length of the sequence.
        # Should be in `_compute_parameters` but needed the raw sigma to compute
        # `pos_confidence`
        sigma = renormalize_input_length(sigma, source_lengths_tensor, 1)

        rel_counter_encoder = renormalize_input_length(self.rel_counter.expand(batch_size, -1, 1),
                                                       source_lengths_tensor - 1,
                                                       self.max_len - 1)

        # slow because list comprehension : should optimize
        pos_attn = pad_sequence([self.positioner(rel_counter_encoder[i_batch, :length, :],
                                                 mu[i_batch].squeeze(),
                                                 sigma[i_batch].squeeze())
                                 for i_batch, length in enumerate(source_lengths_list)],
                                batch_first=True)

        # new size = (batch, n_queries, n_keys)
        pos_attn = pos_attn.transpose(2, 1)

        if self.training and self.is_sample_attn and self.n_training_calls > self.n_steps_prepare_pos:
            self.temperature = max(0.5, self.temperature**1.005)
            soft_onehot = torch.distributions.RelaxedOneHotCategorical(self.temperature,
                                                                       probs=pos_attn)
            pos_attn = soft_onehot.rsample()

        self.add_to_visualize([mu, sigma, pos_confidence], ["mu", "sigma", "pos_confidence"])

        self.add_to_test([mu, sigma], ["mu", "sigma"])

        return pos_attn, pos_confidence, mu, sigma, additional

    def _get_features(self, decoder_outputs, source_lengths_tensor, step, mu_old,
                      sigma_old, mean_content_old, mean_attn_old, additional):
        """
        Gets the inputs and the buillding blocks for positioning. Together
        those will be used to compute the parameters of the positioning function.
        """

        batch_size, _, _ = decoder_outputs.size()

        # j/n
        rel_counter_decoder = renormalize_input_length(self.rel_counter[step:step + 1
                                                                        ].expand(batch_size, 1),
                                                       source_lengths_tensor - 1,
                                                       self.max_len - 1)
        # j
        abs_counter_decoder = (self.rel_counter[step:step + 1].expand(batch_size, 1) *
                               (self.max_len - 1))

        if step == 0:
            mu_old = self.mu0.expand(batch_size, 1)
            sigma_old = self.sigma0.expand(batch_size, 1)
            mean_attn_olds = mu_old
        else:
            mean_attn_olds = additional["mean_attn_olds"]
            mu_old = mu_old.squeeze(2)
            sigma_old = sigma_old.squeeze(2)
            discounting_factor = torch.relu(self.mean_attns_discounting_factor)
            mean_attn_olds = (mean_attn_olds.squeeze(2) * step * (1 - discounting_factor))
            mean_attn_olds = (mean_attn_olds + mean_attn_old * discounting_factor) / (step + 1)

        single_step = renormalize_input_length(self.single_step.expand(batch_size, 1),
                                               source_lengths_tensor - 1,
                                               self.max_len - 1)

        dict_features = dict(mean_attn_old=mean_attn_old,
                             rel_counter_decoder=rel_counter_decoder,
                             abs_counter_decoder=abs_counter_decoder,
                             sigma_old=sigma_old,
                             mu_old=mu_old,
                             single_step=single_step,
                             source_lengths=source_lengths_tensor.unsqueeze(-1),
                             mean_content_old=mean_content_old,
                             mean_attn_olds=mean_attn_olds)

        if self.is_bb_bias:
            dict_features["bias"] = self.bias.expand(batch_size, 1)

        # next line needed for python < 3.6 . for higher can use
        # list(dict_mu_weights.values())
        ordered_blocks = [dict_features[l] for l in self.bb_labels]
        building_blocks = torch.cat(ordered_blocks, dim=1)

        content_old_label = ["mean_content_old"] if self.is_content_attn else []
        pos_features_only_labels = ["sigma_old", "abs_counter_decoder", "source_lengths",
                                    "mu_old", "mean_attn_olds"] + content_old_label
        pos_features_labels = set(l for l in (self.bb_labels + pos_features_only_labels)
                                  if l != "bias")
        additional_pos_features = [dict_features[l] for l in pos_features_labels]

        positioning_inputs = torch.cat([decoder_outputs.squeeze(1)] + additional_pos_features,
                                       dim=1)

        additional["mean_attn_olds"] = mean_attn_olds.unsqueeze(2)

        return positioning_inputs, building_blocks, additional

    def _compute_parameters(self, positioning_inputs, building_blocks, step,
                            source_lengths_tensor, additional=None):
        """
        Compute the parameters of the positioning function.

        Note:
            - Additional is used here only when `is_reg_variance_weights` as it
            will contain the last `mu_weights` that are needed to compute the loss,
            or when using recurrent positioning attention as it needs to give the
            last `positioner_hidden`.
        """

        (positioning_outputs,
         additional) = self._compute_positioning_outputs(positioning_inputs,
                                                         step,
                                                         additional)

        raw_mu_weights = self.mu_weights_generator(positioning_outputs)
        raw_mu_weights = raw_mu_weights + self.expected_mu_weights_init

        gates = self._regularize_bb(positioning_outputs, raw_mu_weights, step)

        gates = self._initialization_helper_gates(gates)

        building_blocks = self._transform_bb(building_blocks, step)

        mu_weights, additional = self._transform_weights(raw_mu_weights, gates, step,
                                                         additional)

        mu = torch.bmm(mu_weights.unsqueeze(1), building_blocks.unsqueeze(2))

        mu = self._transform_mu(mu, source_lengths_tensor, step)

        sigma = self._generate_sigma(positioning_outputs, mu, step)
        sigma = self._initialization_helper_sigma(sigma)

        self.add_to_test([raw_mu_weights, mu_weights, building_blocks],
                         ['raw_mu_weights', 'mu_weights', 'building_blocks'])

        self.add_to_visualize([mu_weights, building_blocks],
                              ['mu_weights', 'building_blocks'])

        return mu, sigma, additional

    def _compute_positioning_outputs(self, positioning_inputs, step, additional):
        """Compute the intermediate to the parameters of the positioning function."""

        batch_size = positioning_inputs.size(0)

        if self.is_recurrent:
            if step == 0:
                additional["positioner_hidden"] = replicate_hidden0(self.hidden0, batch_size)
            positioning_outputs, positioner_hidden = self.rnn(positioning_inputs.unsqueeze(1),
                                                              additional["positioner_hidden"])
            additional["positioner_hidden"] = positioner_hidden
            positioning_outputs = positioning_outputs.squeeze(1)
        else:
            positioning_outputs = positioning_inputs

        return positioning_outputs, additional

    def _regularize_bb(self, positioning_outputs, mu_weights, step):
        """Regularize building block associated values."""

        if self.is_l0_bb_weights:
            gates, loss = self.linear_l0_weights(positioning_outputs)
            if self.is_regularize:
                self.add_regularization_loss("pos_l0_weights", loss)

            self.add_to_test(gates, "bb_gates")
        else:
            gates = None

        if not self.is_regularize:
            return gates

        if "is_reg_const_weights" in self.regularizations:
            # regularizes the constant values that could be used by the network
            # to bypass the other buidling blocks by having the weights = mu

            # no need of regularizing the bias weight when it's rounded
            add_bias = (self.is_bb_bias and self.rounder_weights is None)
            reg_labels_const = ["single_step"] + (["bias"] if add_bias else [])
            w_idcs_const = get_indices(self.bb_labels, reg_labels_const)

            loss = batch_reduction_f(regularization_loss(mu_weights[:, w_idcs_const],
                                                         p=self.lp_reg_weights,
                                                         dim=-1),
                                     torch.mean)
            self.add_regularization_loss("pos_const_weights", loss)

        if "is_reg_old_weights" in self.regularizations:
            # regularizes the weights of the building blocks that are not stable
            # yet (because they depend on positioning attention)
            idcs_pos_old = get_indices(self.bb_labels, ["mu_old", "mean_attn_old"])

            loss = batch_reduction_f(regularization_loss(mu_weights[:, idcs_pos_old],
                                                         p=self.lp_reg_weights,
                                                         dim=-1),
                                     torch.mean)
            self.add_regularization_loss("pos_old_weights", loss)

        return gates

    def _initialization_helper_gates(self, gates):
        """
        Transforms the weights accosicated with the building blocks
        during the initial training steps, in order to help it converging
        at the right position.
        """
        def interpolate_help(start, end):
            delta = (end - start) * interpolating_factor
            return start + delta

        if (self.is_l0_bb_weights and self.training and
                self.n_training_calls < self.n_steps_init_help):

            interpolating_factor = (self.n_training_calls - 1) / self.n_steps_init_help

            new_gates = torch.zeros_like(gates)
            new_gates[:, self.bb_labels.index("bias")] = 1.
            new_gates[:, self.bb_labels.index("rel_counter_decoder")] = 1.

            gates = (new_gates * interpolate_help(0.5, 0) +
                     gates * interpolate_help(0.5, 1))

        return gates

    def _initialization_helper_sigma(self, sigma):
        """
        Transforms the sigma during the initial training steps, in order to help
        it converging at the right position.
        """
        def interpolate_help(start, end):
            delta = (end - start) * interpolating_factor
            return start + delta

        if self.training and self.n_training_calls < self.n_steps_init_help:
            interpolating_factor = (self.n_training_calls - 1) / self.n_steps_init_help

            # use a very small sigma at the begining as we "are showing where to attend to"
            sigma = (self.min_sigma * interpolate_help(0.9, 0) +
                     sigma * interpolate_help(0.1, 1))

        return sigma

    def _initialization_helper_weights(self, dict_mu_weights):
        """
        Transforms the weights accosicated with the building blocks
        during the initial training steps, in order to help it converging
        at the right position.
        """
        def interpolate_help(start, end):
            delta = (end - start) * interpolating_factor
            return start + delta

        if self.training and self.n_training_calls < self.n_steps_init_help:
            interpolating_factor = (self.n_training_calls - 1) / self.n_steps_init_help

            # adds either 0.75 or -0.75
            oscilating075 = (0.5 - (self.n_training_calls % 2)) * 3 / 2
            dict_mu_weights["rel_counter_decoder"
                            ] = (dict_mu_weights["rel_counter_decoder"] *
                                 interpolate_help(0.25, 1) +
                                 interpolate_help(oscilating075, 0))
            if self.is_bb_bias:
                # adds either 0.125 or 0.875
                dict_mu_weights["bias"
                                ] = (dict_mu_weights["bias"] *
                                     interpolate_help(0.125, 1) +
                                     interpolate_help(0.5 + oscilating075 / 2, 0))

            for l in self.bb_labels:
                if l not in ["rel_counter_decoder", "bias"]:
                    # at the start use mostly bias and rel_counter_decoder
                    dict_mu_weights[l] = dict_mu_weights[l] * interpolate_help(0.1, 1)

        return dict_mu_weights

    def _transform_bb(self, building_blocks, step):
        """Transforms the building block values."""

        return building_blocks

    def _transform_weights(self, mu_weights, gates, step, additional):
        """Transforms the building block weights."""

        bb_labels_old = [l for l in ["mu_old", "mean_attn_old", "mean_content_old"]
                         if l in self.bb_labels]

        dict_mu_weights = dict(zip(self.bb_labels, mu_weights.unbind(-1)))

        # TO DO : REMOVE if not useful
        self._initialization_helper_weights(dict_mu_weights)

        # clamping
        if self.is_clamp_weights:
            clamp_weights_kwargs = dict(minimum=0., maximum=1., is_leaky=True)
            clamp_rel_counter_dec_kwargs = dict(minimum=-1., maximum=1., is_leaky=True)

            if self.is_regularize and "is_reg_clamp_weights" in self.regularizations:
                losses = []
                for l in bb_labels_old + (["bias"] if self.is_bb_bias else []):
                    dict_mu_weights[l], loss = clamp_regularize(dict_mu_weights[l],
                                                                **clamp_weights_kwargs)
                    losses.append(loss)

                (dict_mu_weights["rel_counter_decoder"],
                 loss_rel_counter_dec) = clamp_regularize(dict_mu_weights["rel_counter_decoder"],
                                                          **clamp_rel_counter_dec_kwargs)
                losses.append(loss_rel_counter_dec)

                self.add_regularization_loss("pos_clamp_weights", mean(losses))
            else:
                for l in bb_labels_old + (["bias"] if self.is_bb_bias else []):
                    dict_mu_weights[l] = clamp(dict_mu_weights[l], **clamp_weights_kwargs)

                dict_mu_weights["rel_counter_decoder"
                                ] = clamp(dict_mu_weights["rel_counter_decoder"],
                                          **clamp_rel_counter_dec_kwargs)

        # rounding
        if self.rounder_weights is not None:
            for i, l in enumerate(bb_labels_old + ["single_step",
                                                   "rel_counter_decoder"]):
                dict_mu_weights[l] = self.rounder_weights(dict_mu_weights[l],
                                                          is_update=(step == 0 and i == 0))

            if self.is_bb_bias:
                # rounds up to 0.5. IS IT NECESSARY ?????????????????
                dict_mu_weights["bias"] = self.rounder_weights(dict_mu_weights["bias"] * 2.,
                                                               is_update=False) / 2.

        # next line needed for python < 3.6 . for higher can use
        # list(dict_mu_weights.values())
        ordered_weights = [dict_mu_weights[l] for l in self.bb_labels]
        mu_weights = torch.stack(ordered_weights, dim=-1)

        # want this regularization between rounding and gating
        # if not shouldn't be in this method
        if self.is_regularize and "is_reg_variance_weights" in self.regularizations:
            # forces the weights to always be relatively similar after rounding
            if step != 0:
                loss = batch_reduction_f(regularization_loss(mu_weights -
                                                             additional["mu_weights"],
                                                             p=0.5,
                                                             dim=-1),
                                         torch.mean)
                self.add_regularization_loss("pos_variance_weights", loss)

            additional["mu_weights"] = mu_weights

        if self.is_l0_bb_weights:
            mu_weights = (mu_weights * gates)

        return mu_weights, additional

    def _transform_mu(self, mu, source_lengths_tensor, step):
        """Postpreocessing of `mu`."""
        if self.rounder_mu is not None:
            # rounding to words
            normalizer = (source_lengths_tensor - 1).unsqueeze(1).unsqueeze(1)
            mu = self.rounder_mu(mu * normalizer, is_update=(step == 0)
                                 ) / normalizer

        clamp_mu_kwargs = dict(minimum=0, maximum=1, is_leaky=True)
        if self.is_regularize and "is_reg_clamp_mu" in self.regularizations:
            mu, loss = clamp_regularize(mu, **clamp_mu_kwargs)
            self.add_regularization_loss("pos_clamp_mu", loss)
        else:
            mu = clamp(mu, **clamp_mu_kwargs)

        return mu

    def _generate_sigma(self, positioning_outputs, mu, step):
        """Generate sigma."""
        is_update_sigma = self.training and step == 0

        if self.get_sigma.is_annealing:
            current_min_sigma = self.get_sigma(is_update_sigma)

            # if you are still annealing min sigma then don't backprop
            # to sigma generator
            sigma = current_min_sigma + torch.zeros_like(mu)
        else:
            unclamped_sigma = (self.get_sigma.final_value +
                               self.sigma_generator(positioning_outputs))
            sigma = clamp(unclamped_sigma.unsqueeze(1),
                          minimum=self.min_sigma,
                          is_leaky=True,
                          negative_slope=0.1,
                          hard_min=self.hard_min_sigma)

        return sigma

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
        """
        pos_confidence = torch.exp(-sigma**2 + self.hard_min_sigma**2) * (1 - min_p)
        pos_confidence = pos_confidence.squeeze(-1)

        return pos_confidence
