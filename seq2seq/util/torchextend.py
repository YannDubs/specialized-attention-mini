"""
Pytorch extension modules.

To Do - medium:
    - add input / output size in docstrings of `forward` function.

Contact: Yann Dubois
"""
import ipdb
import logging

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from seq2seq.util.initialization import linear_init, init_param
from seq2seq.util.helpers import (get_extra_repr, identity, clamp, Clamper,
                                  HyperparameterInterpolator, batch_reduction_f,
                                  bound_probability)
from seq2seq.util.base import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NO_PLATEAU_EPS = False
logger = logging.getLogger(__name__)


class MLP(Module):
    """General MLP class.

    Args:
        input_size (int): size of the input.
        output_size (int): output size.
        hidden_size (int, optional): number of hidden neurones. Forces it to be between
            [input_size, output_size].
        activation (torch.nn.modules.activation, optional): unitialized activation class.
        dropout_input (float, optional): dropout probability to apply on the
            input of the generator.
        dropout_hidden (float, optional): dropout probability to apply on the
            hidden layer of the generator.
    """

    def __init__(self, input_size, output_size,
                 hidden_size=32,
                 activation=nn.LeakyReLU,
                 bias=True,
                 dropout_input=0,
                 dropout_hidden=0):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = min(self.input_size, max(hidden_size, self.output_size))

        self.dropout_input = (nn.Dropout(p=dropout_input)
                              if dropout_input > 0 else identity)
        self.mlp = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        self.dropout_hidden = (nn.Dropout(p=dropout_hidden)
                               if dropout_hidden > 0 else identity)
        self.activation = activation()  # cannot be a function from Functional but class
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        x = self.dropout_input(x)
        y = self.mlp(x)
        y = self.dropout_hidden(y)
        y = self.activation(y)
        y = self.out(y)
        return y

    def reset_parameters(self):
        linear_init(self.mlp, activation=self.activation)
        linear_init(self.out)

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["input_size", "hidden_size", "output_size"])


class ProbabilityConverter(Module):
    """Maps floats to probabilites (between 0 and 1), element-wise.

    Args:
        min_p (int, optional): minimum probability, can be useful to set greater
            than 0 in order to keep gradient flowing if the probability is used
            for convex combinations of different parts of the model. Note that
            maximum probability is `1-min_p`.
        activation ({"sigmoid", "hard-sigmoid"}, optional): name of the activation
            to use to generate the probabilities. `sigmoid` has the advantage of
            being smooth and never exactly 0 or 1, which helps gradient flows.
            `hard-sigmoid` has the advantage of making all values between min_p
            and max_p equiprobable.
        temperature (bool, optional): whether to add a paremeter controling the
            steapness of the activation. This is useful when x is used for multiple
            tasks, and you don't want to constraint its magnitude.
        bias (bool, optional): bias used to shift the activation. This is useful
            when x is used for multiple tasks, and you don't want to constraint
            it's scale.
        initial_temperature (int, optional): initial temperature, a higher
            temperature makes the activation steaper.
        initial_probability (float, optional): initial probability you want to
            start with.
        initial_x (float, optional): first value that will be given to the function,
            important to make initial_probability work correctly.
        bias_transformer (callable, optional): transformer function of the bias.
            This function should only take care of the boundaries (ex: leaky relu
            or relu). Note: cannot be a lambda function because of pickle.
            (default: identity)
        temperature_transformer (callable, optional): transformer function of the
            temperature. This function should only take care of the boundaries
            (ex: leaky relu  or relu), if not the initial_probability might not
            work correctly (as `_porbability_to_bias`) doesn't take into account
            the transformers. By default leakyclamp to [.1,10] then hardclamp to
            [0.01, inf[. Note: cannot be a lambda function because
            of pickle.
        fix_point (tuple, optional): tuple (x,y) which defines a fix point of
            the probability converter. With a fix_point, can only use a temperature
            parameter but not a bias. This is only possible with `activation="hard-sigmoid"`.
        plateau_eps (float, optional): if proba is less than `plateau_eps`
            next to 0 or 1 then round the forward pass (but keep back pass).
            Note that this is done after bounding the proba with `min_p` and
            `plateau_eps` should thus be larger than `min_p`.
    """

    def __init__(self,
                 min_p=0.,
                 activation="sigmoid",
                 is_temperature=False,
                 is_bias=False,
                 initial_temperature=1.0,
                 initial_probability=0.5,
                 initial_x=0,
                 bias_transformer=identity,
                 temperature_transformer=Clamper(minimum=0.1, maximum=10., is_leaky=True,
                                                 hard_min=0.01),
                 fix_point=None,
                 plateau_eps=1e-2  # DEV MODE # to DOC
                 ):

        super(ProbabilityConverter, self).__init__()
        self.min_p = min_p
        self.activation = activation
        self.is_temperature = is_temperature
        self.is_bias = is_bias
        self.initial_temperature = initial_temperature
        self.initial_probability = initial_probability
        self.initial_x = initial_x
        self.bias_transformer = bias_transformer
        self.temperature_transformer = temperature_transformer
        self.fix_point = fix_point
        self.plateau_eps = plateau_eps

        if self.fix_point is not None:
            if self.activation != "hard-sigmoid":
                logger.warning("Can only use `fix_point` if activation=hard-sigmoid. Replace {} by 'hard-sigmoid' ".format(self.activation))
                self.activation = 'hard-sigmoid'

            if self.is_bias:
                logger.warning("Cannot use bias when using `fix_point`. Setting to False and using temperature instead. ".format(self.activation))
                self.is_bias = False
                self.is_temperature = True

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.fix_point is None:
            if self.is_temperature:
                self.temperature = Parameter(torch.tensor(self.initial_temperature)).to(device)
            else:
                self.temperature = torch.tensor(self.initial_temperature).to(device)

            initial_bias = self._probability_to_bias(self.initial_probability,
                                                     initial_x=self.initial_x)
            if self.is_bias:
                self.bias = Parameter(torch.tensor(initial_bias)).to(device)
            else:
                self.bias = torch.tensor(initial_bias).to(device)
        else:
            self.initial_temperature = self._fix_point_temperature_init(self.initial_probability,
                                                                        initial_x=self.initial_x)
            if self.is_temperature:
                self.temperature = Parameter(torch.tensor(self.initial_temperature)).to(device)
            else:
                self.temperature = torch.tensor(self.initial_temperature).to(device)

    def forward(self, x):
        temperature = self.temperature_transformer(self.temperature)
        if self.fix_point is None:
            bias = self.bias_transformer(self.bias)

        if self.activation == "sigmoid":
            full_p = torch.sigmoid((x + bias) * temperature)
        elif self.activation == "hard-sigmoid":
            if self.fix_point is not None:
                y = 0.2 * ((x - self.fix_point[0]) * temperature) + self.fix_point[1]
            else:
                y = 0.2 * ((x + bias) * temperature) + 0.5
            full_p = clamp(y, minimum=0., maximum=1., is_leaky=False)

        elif self.activation == "leaky-hard-sigmoid":
            y = 0.2 * ((x + bias) * temperature) + 0.5
            full_p = clamp(y, minimum=0.1, maximum=.9,
                           is_leaky=True, negative_slope=0.01,
                           hard_min=0, hard_max=0)
        else:
            raise ValueError("Unkown activation : {}".format(self.activation))

        p = bound_probability(full_p, self.min_p)

        if self.plateau_eps is not None and not NO_PLATEAU_EPS:  # DEV Mode
            p = replace_sim(p, [0, 1], [self.plateau_eps, self.plateau_eps])

        return p

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["min_p", "activation"],
                              conditional_shows=["is_temperature", "is_bias",
                                                 "initial_temperature",
                                                 "initial_probability",
                                                 "initial_x",
                                                 "fix_point"])

    def _probability_to_bias(self, p, initial_x=0):
        """Compute the bias to use given an inital `point(initial_x, p)`"""
        assert p > self.min_p and p < 1 - self.min_p
        range_p = 1 - self.min_p * 2
        p = (p - self.min_p) / range_p
        p = torch.tensor(p, dtype=torch.float)
        if self.activation == "sigmoid":
            bias = -(torch.log((1 - p) / p) / self.initial_temperature + initial_x)

        elif self.activation == "hard-sigmoid" or self.activation == "leaky-hard-sigmoid":
            bias = ((p - 0.5) / 0.2) / self.initial_temperature - initial_x
        else:
            raise ValueError("Unkown activation : {}".format(self.activation))

        return bias

    def _fix_point_temperature_init(self, p, initial_x=0):
        """
        Compute the temperature to use based on a given inital `point(initial_x, p)`
            and a fix_point.
        """
        assert p > self.min_p and p < 1 - self.min_p
        range_p = 1 - self.min_p * 2
        p = (p - self.min_p) / range_p
        p = torch.tensor(p, dtype=torch.float)

        temperature = 5 * (p - self.fix_point[1]) / (initial_x - self.fix_point[0])

        return temperature


class GaussianNoise(Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): standard deviation used to generate the noise.
        is_relative_sigma (bool, optional): whether to use relative standard
            deviation instead of absolute. Relative means that it will be
            multiplied by the magnitude of the value your are adding the noise
            to. This means that sigma can be the same regardless of the scale of
            the vector. This should be `True` unless you have a good reason not
            to, indeed the model could bypass the noise if `False` by increasing
            all by a scale factor.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise if `is_relative_sigma=True`. If
            `False` then the scale of the noise won't be seen as a constant but
            something to optimize: this will bias the network to generate vectors
            with smaller values.

    Note: This can be though of as emulating a variational autoencoder. Indeed,
        a variational autoencoder simply adds noise (to produce a distribution
        and not points) and constrains the mean such that the model cannot increase
        the mean to effectively remove the constant noise. By adding relative noise
        you effectively remove that possibility.
    """

    def __init__(self, sigma=0.1, is_relative_sigma=True, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_sigma = is_relative_sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0.0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma
            if self.is_relative_sigma:
                scale = scale * abs((x.detach() if self.is_relative_detach else x))
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["sigma"],
                              conditional_shows=["is_relative_sigma",
                                                 "is_relative_detach"])


class AnnealedGaussianNoise(GaussianNoise):
    """Gaussian noise regularizer with annealing.

    Args:
        initial_sigma (float, optional): initial sigma.
        final_sigma (float, optional): final standard deviation used to generate
            the noise.
        n_steps_interpolate (int, optional): number of training steps before
            reaching the `final_sigma`.
        mode ({"linear", "geometric"}, optional): interpolation mode.
        is_relative_sigma (bool, optional): whether to use relative standard
            deviation instead of absolute. Relative means that it will be
            multiplied by the magnitude of the value your are adding the noise
            to. This means that sigma can be the same regardless of the scale of
            the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise if `is_relative_sigma=True` . If
            `False` then the scale of the noise won't be seen as a constant but
            something to optimize: this will bias the network to generate vectors
            with smaller values.
        kwargs: additional arguments to `HyperparameterInterpolator`.
    """

    def __init__(self,
                 initial_sigma=0.2,
                 final_sigma=0,
                 n_steps_interpolate=0,
                 mode="linear",
                 is_relative_sigma=True,
                 is_relative_detach=True,
                 **kwargs):
        super().__init__(sigma=initial_sigma,
                         is_relative_sigma=is_relative_sigma,
                         is_relative_detach=is_relative_detach)

        self.get_sigma = HyperparameterInterpolator(initial_sigma,
                                                    final_sigma,
                                                    n_steps_interpolate,
                                                    mode=mode,
                                                    **kwargs)

    def reset_parameters(self):
        self.get_sigma.reset_parameters()
        super().reset_parameters()

    def extra_repr(self):
        detached_str = '' if self.is_relative_sigma else ', not_relative'
        detached_str += '' if self.is_relative_detach else ', not_detached'
        txt = self.get_sigma.extra_repr(value_name="sigma")
        return txt + detached_str

    def forward(self, x, is_update=True):
        self.sigma = self.get_sigma(is_update and self.training)
        return super().forward(x)


class AnnealedDropout(nn.Dropout):
    """Dropout regularizer with annealing.

    Args:
        initial_dropout (float, optional): initial dropout probability.
        final_dropout (float, optional): final dropout probability. Default is 0
            if no interpolate and 0.1 if interpolating.
        n_steps_interpolate (int, optional): number of training steps before
            reaching the `final_dropout`.
        mode ({"linear", "geometric"}, optional): interpolation mode.
        kwargs: additional arguments to `HyperparameterInterpolator`.
    """

    def __init__(self,
                 initial_dropout=0.7,
                 final_dropout=None,
                 n_steps_interpolate=0,
                 mode="geometric",
                 **kwargs):
        super().__init__(p=initial_dropout)

        if final_dropout is None:
            final_dropout = 0 if n_steps_interpolate == 0 else 0.3

        self.get_dropout_p = HyperparameterInterpolator(initial_dropout,
                                                        final_dropout,
                                                        n_steps_interpolate,
                                                        mode=mode,
                                                        **kwargs)

    def reset_parameters(self):
        self.get_dropout_p.reset_parameters()
        super().reset_parameters()

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        txt = self.get_dropout_p.extra_repr(value_name="dropout")
        return txt + inplace_str

    def forward(self, x, is_update=True):
        self.p = self.get_dropout_p(is_update and self.training)
        if self.p == 0:
            return x
        return super().forward(x)


class StochasticRounder(Module):
    """Applies differentiable stochastic rounding.

    Notes:
        - I thought that the gradient were biased but now I'm starting to think
        that they are actually unbiased as if you average over multiple rounding
        steps the average will be an identity function (i.e the expectation is to
        map each point to itself). In which case this would be unbiased. But
        empirically I find better results with `ConcreteRounder` which makes me think
        that it is actually biased. HAVE TO CHECK
        - approximatevly 1.5x speedup compared to concrete rounding.

    Args:
        min_p (float, optional): minimum probability of rounding to the "wrong"
            number. Useful to keep exploring.
        start_step (int, optional): number of steps to wait for before starting rounding.
    """

    def __init__(self, min_p=0.001, start_step=0):
        super().__init__()
        self.min_p = min_p
        self.start_step = start_step

    def extra_repr(self):
        return get_extra_repr(self, always_shows=["start_step"])

    def forward(self, x,
                is_update=False,  # is_update just to use same syntax as concrete
                idcs_to_round=None):

        if not self.training:
            return x.round()

        if self.start_step > self.n_training_calls:
            return x

        x_detached = x.detach()
        x_floored = x_detached.floor()
        decimals = (x_detached - x_floored)
        p = bound_probability(decimals, self.min_p)
        x_hard = x_floored + torch.bernoulli(p)
        x_delta = x_hard - x_detached
        x_rounded = x_delta + x

        if idcs_to_round is not None:
            idcs_to_round = idcs_to_round.float()
            x_rounded = x_rounded * idcs_to_round + (1 - idcs_to_round) * x

        return x_rounded


class ConcreteRounder(Module):
    """Applies rounding through gumbel/concrete softmax.

    Notes:
        - Approximatively 1.5x slower than stochastic rounding.
        - The temperature variable follows the implementation in the paper,
            so it is the inverse of the temperature in `ProbabilityConverter`.
            I.e lower temperature means higher slope.
        - The gradients with respect to `x` look like multiple waves, but the peaks
            are higher for higher absolute values. The peaks are at the integres.

    Args:
        start_step (int, optional): number of steps to wait for before starting rounding.
        min_p (float, optional): minimum probability of rounding to the "wrong"
            number. Useful to keep exploring.
        initial_temperature (float, optional): initial softmax temperature.
        final_temperature (float, optional): final softmax temperature. Default:
            `2/3 if n_steps_interpolate==0 else 0.5`. If temperature -> infty,
            the realex bernouilli distributions start looking like a constant
            distribution equal to 0.5 (i.e whatever the decimal, the probability
            of getting rounded above is 0.5). If temperature -> 0, the relaxed
            bernouilli becomes a real bernoulli (i.e probabiliry of getting
            rounded above is equal to the decimal).
        n_steps_interpolate (int, optional): number of training steps before
            reaching the `final_temperature`.
        mode (str, optional): interpolation mode. One of {"linear", "geometric"}.
        kwargs: additional arguments to `HyperparameterInterpolator`.
    """

    def __init__(self,
                 start_step=0,
                 min_p=0.001,
                 initial_temperature=1,
                 final_temperature=None,
                 n_steps_interpolate=0,
                 mode="linear",
                 is_hard=True,  # DEV MODE
                 **kwargs):
        super().__init__()

        if final_temperature is None:
            final_temperature = 2 / 3 if n_steps_interpolate == 0 else 0.5

        self.start_step = start_step
        self.min_p = min_p
        self.is_hard = is_hard
        self.get_temperature = HyperparameterInterpolator(initial_temperature,
                                                          final_temperature,
                                                          n_steps_interpolate,
                                                          mode=mode,
                                                          **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        self.get_temperature.reset_parameters()

    def extra_repr(self):
        txt = get_extra_repr(self, always_shows=["start_step"])
        interpolator_txt = self.get_temperature.extra_repr(value_name="temperature")

        txt += ", " + interpolator_txt
        return txt

    def forward(self, x, is_update=True, idcs_to_round=None):
        if not self.training:
            return x.round()

        if self.start_step > self.n_training_calls:
            return x

        temperature = self.get_temperature(is_update)

        x_detached = x.detach()
        x_floored = x_detached.floor()

        decimals = x - x_floored
        p = bound_probability(decimals, self.min_p)
        softBernouilli = torch.distributions.RelaxedBernoulli(temperature, p)
        soft_sample = softBernouilli.rsample()

        if self.is_hard:
            new_d_detached = soft_sample.detach()
            # removes a detached version of the soft X and adds the real X
            # to emulate the fact that we add some non differentaible noise which just
            # hapens to make the variable rounded. I.e the total is still differentiable
            new_decimals = new_d_detached.round() - new_d_detached + soft_sample
            x_rounded = x_floored + new_decimals - x_detached + x
        else:
            x_rounded = x_floored + soft_sample - x_detached + x

        if idcs_to_round is not None:
            idcs_to_round = idcs_to_round.float()
            x_rounded = x_rounded * idcs_to_round + (1 - idcs_to_round) * x

        return x_rounded


def get_rounder(name=None, **kwargs):
    """Return the correct  rounding module.

    Args:
        name ({"concrete", "stochastic"}, optional): name of the rounding method.
            `"concrete"` Applies rounding through gumbel/concrete softmax.
            `"stocchastic"` differentiable stochastic rounding.
        kwargs:
            Additional arguments to the *Rounding
    """
    if name is None:
        return None
    elif name == "concrete":
        return ConcreteRounder(**kwargs)
    elif name == "stochastic":
        return StochasticRounder(**kwargs)
    elif name == "softConcrete":
        return ConcreteRounder(**kwargs, is_hard=False)
    elif name == "plateau":
        return PlateauAct([0, 1], **kwargs)
    else:
        raise ValueError("Unkown rounder method {}".format(name))


class L0Gates(Module):
    """Return gates for L0 regularization.

    Notes:
        Main idea taken from `Learning Sparse Neural Networks through L_0
        Regularization`, but modified using plateau activation.

    Args:
        input_size (int): size of the input to the gate generator.
        output_size (int): length of the vectors to dot product.
        is_at_least_1 (bool, optional): only start regularizing if more than one
            gate is 1.
        bias (bool, optional): whether to use a bias for the gate generation.
        Generator (Module, optional): module to generate various values. It
            should be callable using `Generator(input_size, output_size)(x)`.
            By default `nn.Linear`.
        initial_gates (float or list, optional): initial expected sum of gates
            to use. If scalar then simply adds `initial_gate/n_gates` to each.
            If vector then specify the inital expected gate for every gate.
        gating ({None, "residual", "highway", "gates_res"}, optional): Gating
            mechanism for generated values. `None` no gating. `"residual"` adds
            the new value to the previous. `"highway"` gating using convex
            combination. `"gates_res"` gates the previous value and add the new one.
        kwargs:
            Additional arguments to the Generator.
    """

    def __init__(self,
                 input_size, output_size,
                 is_at_least_1=False,
                 Generator=nn.Linear,
                 initial_gates=0.,
                 gating="gated_res",
                 rounder=None,  # DEV MODE
                 **kwargs):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.is_at_least_1 = is_at_least_1
        self.gating = gating
        self.gate_generator = Generator(self.input_size, self.output_size,
                                        **kwargs)
        if not isinstance(initial_gates, list):
            initial_gates = [initial_gates / output_size] * output_size
        self.initial_gates = torch.tensor(initial_gates, dtype=torch.float,
                                          device=device)

        self.gate_to_prob = ProbabilityConverter()
        self.rounder = get_rounder(rounder)

        if self.gating is not None:
            self.old_gater = get_gate(self.gating, input_size, self.output_size,
                                      save_name="l0_old_gate")

        self.reset_parameters()

    def reset_parameters(self):
        if self.gating is not None:
            self.gates0 = init_param(Parameter(torch.ones_like(self.initial_gates))
                                     ) + self.initial_gates
        super().reset_parameters()

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["input_size", "output_size"],
                              conditional_shows=["is_at_least_1", "gating"])

    def forward(self, x, loss_weights=None, is_reset=True):
        gates = self.gate_generator(x)
        gates = gates + self.initial_gates

        if self.gating is not None:
            gates_old = (self.gates0.expand_as(gates)
                         if is_reset else self.storer["raw_gates_old"])
            gates = self.old_gater(gates, gates_old, x)
            # gates old is actually before being clamped to [0,1]
            # such that using residual gates still make sense
            self.storer["raw_gates_old"] = gates

        gates = self.gate_to_prob(gates)
        if self.rounder is not None:
            gates = self.rounder(gates)

        to_reg = gates if loss_weights is None else gates * loss_weights
        loss = batch_reduction_f(to_reg, torch.mean)

        if self.is_at_least_1:
            # correct only if no loss weight
            loss = loss - 1. / gates.size(-1)
            # penalize also if less than 1 but less
            loss = torch.relu(loss) + torch.relu(-loss / 3)

        return gates, loss


class Highway(Module):
    """
    Highway module, which does a weighted average of an old and new vetor based
    on a weight generated from a controller.

    Args:
        controller_size (int): size of the controller that will be used
            to generate the carry weight.
        n_gates (int): Dimension of the vectors to weight. unused if
            `is_single_carry=True`.
        initial_gate (float, optional): initial percentage of the new value
            to let through.
        is_single_gate (bool, optional): whether to use a one dimension gate
            instead of n dimensional.
        is_additive_highway (bool, optional): if `True` the carry weight will
            only be applied to the residue and will not scale the new value with
            `1-carry`.
        generator (Module, optional): module to generate various values. It
            should be callable using `generator(input_size, output_size)(x)`.
            By default `nn.Linear`.
        kwargs:
            Additional arguments to `ProbabilityConverter`
    """

    def __init__(self, controller_size, n_gates,
                 initial_gate=0.5,
                 is_single_gate=False,
                 is_additive_highway=False,
                 Generator=nn.Linear,
                 save_name=None,
                 is_reg=False,  # DEV MODE : regularizes how much of the new goes through
                 rounder=None,  # DEV MODE
                 **kwargs):
        super().__init__()

        self.controller_size = controller_size
        self.initial_gate = initial_gate
        self.is_single_gate = is_single_gate
        self.is_additive_highway = is_additive_highway
        self.save_name = save_name
        self.is_reg = is_reg

        self.n_gates = 1 if self.is_single_gate else n_gates
        self.gate_generator = Generator(self.controller_size, self.n_gates)

        self.gate_to_prob = ProbabilityConverter(initial_probability=self.initial_gate,
                                                 **kwargs)
        self.rounder = get_rounder(rounder)

        self.reset_parameters()

    def extra_repr(self):
        return get_extra_repr(self,
                              conditional_shows=["initial_gate", "is_single_gate",
                                                 "is_additive_highway"])

    def forward(self, x_new, x_old, controller):
        """Weighted average of old and new vector.

        Args:
            x_new (torch.Tensor): new variable of size (batch_size, *, n_gates).
            x_old (torch.Tensor): previous variable of size (batch_size, *, n_gates).
            controller (torch.Tensor): tensor of size (batch_size, *,
                controller_size) used to generate the gates.

        Returns:
            x_new (torch.Tensor): new variable of size (batch_size, *, n_gates).
        """
        gates = self.gate_generator(controller)
        gates = self.gate_to_prob(gates)
        if self.rounder is not None:
            gates = self.rounder(gates)

        if self.save_name is not None:
            self.add_to_test(gates, self.save_name)
            self.add_to_visualize(gates.mean(-1).mean(-1), self.save_name)

        try:
            if self.is_additive_highway:
                x_new = x_new + (1 - gates) * x_old
            else:
                x_new = gates * x_new + (1 - gates) * x_old
        except:
            ipdb.set_trace()

        if self.is_reg:
            loss = batch_reduction_f(gates, torch.mean)
            return x_new, loss
        else:
            return x_new


def no_gate(new, old, controller):
    """No gate helper function as lambda cannot be pickled."""
    return new


def force_gate(new, old, controller):
    """Force gate helper function as lambda cannot be pickled."""
    return old


def res_gate(new, old, controller):
    """Res gate helper function as lambda cannot be pickled."""
    return new + old


def get_gate(gating, *args, **kwargs):
    """Return a callable gate.

    gating ({None, "residual", "highway", "gates_res"}, optional): Gating
        mechanism for generated values. `None` no gating. `"residual"` adds
        the new value to the previous. `"highway"` gating using convex
        combination. `"gates_res"` gates the previous value and add the new one.
    """
    if gating is None:
        return no_gate
    elif gating == "force":
        return force_gate
    elif gating == "residual":
        return res_gate
    elif gating == "highway":
        return Highway(*args, is_additive_highway=False, **kwargs)
    elif gating == "r-highway":
        return Highway(*args, is_additive_highway=False, rounder="stochastic", **kwargs)
    elif gating == "p-highway":
        return Highway(*args, is_additive_highway=False, rounder="plateau", **kwargs)
    elif gating == "c-highway":
        return Highway(*args, is_additive_highway=False, rounder="softConcrete", **kwargs)
    elif gating == "gated_res":
        return Highway(*args, is_additive_highway=True, **kwargs)
    else:
        raise ValueError("Unkown `gating={}`".format(gating))


class PlateauAct(Module):
    """Plateau Activation. The backward pass is a sum of sigmoids that looks like a
    soft staircase. The forward pass uses a constat value for the plateaus of the
    stairs.

    Args:
        plateaus (list or "int"): list of y values for which to have plateaus.
            If `"int"` then uses for all integers.
        is_leaky_bounds (bool, optional): whether to use leaky at the end of
            the last plateaus. It will also extend the first and last plateau
            such that the outside bound part is twice the size of the inside part.
        len_plateaus (float or list, optional): length of each plateau. If list
            has to specify length of each plateau.
        eps (float or list, optional): minimum difference to the final
            value to be considered "in the plateau". If list, specify epsilon
            for each plateau. Has to be a float with `plateaus="int"`.
        negative_slope (float, optional): slope to use if `is_regularize_bounds`.
        len_factor_final (float, optional): by how much to multiply `len_plateaus`
            at the end of training. In between the initial and final `len_plateaus`
            the value will be linearly interpolated. This is useful to force the
            values to be closer to the plateaus at the end of training. If `None`,
            keep `len_plateaus` all the time. Not used when `plateaus == "int"`.
        n_steps_interpolate (float, optional): number of interpolation steps for
            `len_plateaus`. Only used if `len_factor_final is not None`.
    """

    def __init__(self, plateaus,
                 is_leaky_bounds=True,
                 len_plateaus=2e-1,
                 eps=1e-4,
                 negative_slope=1e-2,
                 len_factor_final=None,
                 n_steps_interpolate=100):

        super(PlateauAct, self).__init__()

        self.plateaus = plateaus
        self.len_factor_final = len_factor_final

        if self.len_factor_final is not None:
            if self.plateaus == "int":
                txt = "`len_factor_final` cannot be used with `self.plateaus == 'int'`"
                raise ValueError(txt)
            self.get_len_factor = HyperparameterInterpolator(1,
                                                             self.len_factor_final,
                                                             n_steps_interpolate)

        if self.plateaus == "int":
            self.eps = eps
        else:
            self.is_leaky_bounds = is_leaky_bounds
            self.negative_slope = negative_slope

            if isinstance(len_plateaus, float):
                self.len_plateaus = [len_plateaus] * len(self.plateaus)
            else:
                self.len_plateaus = len_plateaus

            if isinstance(eps, float):
                self.eps = [eps] * len(self.plateaus)
            else:
                self.eps = eps

            self.eps = torch.tensor(self.eps)
            self.len_plateaus = torch.tensor(self.len_plateaus)
            plateaus = torch.tensor(self.plateaus, dtype=torch.float, device=device)
            self.mipoints = (plateaus[1:] + plateaus[:-1]) / 2
            self.heights = plateaus[1:] - plateaus[:-1]
            self.out = plateaus[0]

        self.reset_parameters()

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["plateaus"],
                              conditional_shows=["len_factor_final"])

    def forward(self, x, is_update=True):
        if self.plateaus == "int":
            x = x - 0.5
            rounded = x.detach().round()
            out = rounded + torch.sigmoid(20 * (x - rounded))

            x = replace_int(out, self.eps)
        else:
            if len(self.plateaus) < 2:
                raise ValueError("`len(plateaus) = {}`< 2.".format(len(self.plateaus)))

            if self.len_factor_final is not None:
                len_factor = self.get_len_factor(is_update)
            else:
                len_factor = 1.

            out = self.out
            for mid, h, l, eps in zip(self.mipoints, self.heights, self.len_plateaus, self.eps):
                l = l * len_factor
                t = self.get_temp(h, h, l / 2, eps=eps)
                out = out + self.sigmoid(x, shiftx=mid, temperature=t, height=h)

            if not NO_PLATEAU_EPS:  # DEV MODE
                out = replace_sim(out, self.plateaus, self.eps)

            if self.is_leaky_bounds:
                minimum = min(self.plateaus)
                maximum = max(self.plateaus)
                inside = (x > (minimum - self.len_plateaus[0])
                          ) & (x < (maximum + self.len_plateaus[-1]))
                clamped = clamp(x,
                                minimum=minimum,
                                maximum=maximum,
                                is_leaky=True,
                                negative_slope=self.negative_slope)
                inside = inside.float()
                out = inside * out + (1 - inside) * clamped

        return out

    def get_temp(self, height, length, delta, eps):
        return - torch.log(eps / (height - eps)) / (length / 2 - delta)

    def sigmoid(self, x, shiftx, temperature, height, shifty=0):
        x = temperature * (x - shiftx)
        out = torch.sigmoid(x)
        out = height * out + shifty
        return out


class ReplaceInt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        """
        Replace all elements that are at least `eps` close to an integer .
        """
        rounded = x.round()
        i = (x - rounded).abs() < eps
        x[i] = rounded[i]

        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator for x, others are not tensors
        return grad_output, None


class ReplaceSimilar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, values, eps):
        """
        Replace all elements in a tensor that are at least `eps` close to
        `values` .
        """
        for v, e in zip(values, eps):
            i = (x - v).abs() < e
            x[i] = v

        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator for x, others are not tensors
        return grad_output, None, None


replace_sim = ReplaceSimilar.apply
replace_int = ReplaceInt.apply
