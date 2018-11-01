"""
Pytorch extension modules.

To Do - medium:
    - add input / output size in docstrings of `forward` function.

Contact: Yann Dubois
"""

import warnings

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from seq2seq.util.initialization import linear_init
from seq2seq.util.helpers import (get_extra_repr, identity, clamp, Clamper,
                                  HyperparameterInterpolator, batch_reduction_f,
                                  bound_probability)
from seq2seq.util.base import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(Module):
    """General MLP class.

    Args:
        input_size (int): size of the input.
        hidden_size (int): number of hidden neurones. Forces it to be between
            [input_size, output_size].
        output_size (int): output size.
        activation (torch.nn.modules.activation, optional): unitialized activation class.
        dropout_input (float, optional): dropout probability to apply on the
            input of the generator.
        dropout_hidden (float, optional): dropout probability to apply on the
            hidden layer of the generator.
        noise_sigma_input (float, optional): standard deviation of the noise to
            apply on the input of the generator.
        noise_sigma_hidden (float, optional): standard deviation of the noise to
            apply on the hidden layer of the generator.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 activation=nn.LeakyReLU,
                 bias=True,
                 dropout_input=0,
                 dropout_hidden=0,
                 noise_sigma_input=0,
                 noise_sigma_hidden=0):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = min(self.input_size, max(hidden_size, self.output_size))

        self.dropout_input = (nn.Dropout(p=dropout_input)
                              if dropout_input > 0 else identity)
        self.noise_sigma_input = (GaussianNoise(noise_sigma_input)
                                  if noise_sigma_input > 0 else identity)
        self.mlp = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        self.dropout_hidden = (nn.Dropout(p=dropout_hidden)
                               if dropout_hidden > 0 else identity)
        self.noise_sigma_hidden = (GaussianNoise(noise_sigma_hidden)
                                   if noise_sigma_hidden > 0 else identity)
        self.activation = activation()  # cannot be a function from Functional but class
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        x = self.dropout_input(x)
        x = self.noise_sigma_input(x)
        y = self.mlp(x)
        y = self.dropout_hidden(y)
        y = self.noise_sigma_hidden(y)
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
    """

    def __init__(self,
                 min_p=0.001,
                 activation="sigmoid",
                 is_temperature=False,
                 is_bias=False,
                 initial_temperature=1.0,
                 initial_probability=0.5,
                 initial_x=0,
                 bias_transformer=identity,
                 temperature_transformer=Clamper(minimum=0.1, maximum=10., is_leaky=True,
                                                 hard_min=0.01),
                 fix_point=None):

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

        if self.fix_point is not None:
            if self.activation != "hard-sigmoid":
                warnings.warn("Can only use `fix_point` if activation=hard-sigmoid. Replace {} by 'hard-sigmoid' ".format(self.activation))
                self.activation = 'hard-sigmoid'

            if self.is_bias:
                warnings.warn("Cannot use bias when using `fix_point`. Setting to False and using temperature instead. ".format(self.activation))
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
            final_dropout = 0 if n_steps_interpolate == 0 else 0.1

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


class StochasticRounding(Module):
    """Applies differentiable stochastic rounding.

    Notes:
        - I thought that the gradient were biased but now I'm starting to think
        that they are actually unbiased as if you average over multiple rounding
        steps the average will be an identity function (i.e the expectation is to
        map each point to itself). In which case this would be unbiased. But
        empirically I find better results with `ConcreteRounding` which makes me think
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

    def forward(self, x, is_update): # is_update just to use same syntax as concrete
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
        return x_rounded


class ConcreteRounding(Module):
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
                 **kwargs):
        super().__init__()

        if final_temperature is None:
            final_temperature = 2 / 3 if n_steps_interpolate == 0 else 0.5

        self.start_step = start_step
        self.min_p = min_p
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

    def forward(self, x, is_update=True):
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
        new_d_detached = soft_sample.detach()
        # removes a detached version of the soft X and adds the real X
        # to emulate the fact that we add some non differentaible noise which just
        # hapens to make the variable rounded. I.e the total is still differentiable
        new_decimals = new_d_detached.round() - new_d_detached + soft_sample
        x_rounded = x_floored + new_decimals - x_detached + x
        return x_rounded


class L0Gates(Module):
    """Return gates for L0 regularization.

    Notes:
        Main idea taken from `Learning Sparse Neural Networks through L_0
        Regularization`, but modified using straight through Gumbel
        softmax estimator.

    Args:
        input_size (int): size of the input to the gate generator.
        output_size (int): length of the vectors to dot product.
        is_at_least_1 (bool, optional): only start regularizing if more than one
            gate is 1.
        bias (bool, optional): whether to use a bias for the gate generation.
        is_mlp (bool, optional): whether to use a MLP for the gate generation.
        initial_gates (float or list, optional): initial expected sum of gates
            to use. If scalar then simply adds `initial_gate/n_gates` to each.
            If vector then specify the inital expected gate for every gate.
        rounding_kwargs (dictionary, optional): additional arguments to the
            `ConcreteRounding`.
        kwargs:
            Additional arguments to the gate generator.
    """

    def __init__(self,
                 input_size, output_size,
                 is_at_least_1=False,
                 is_mlp=False,
                 initial_gates=0.,
                 rounding_kwargs={},
                 **kwargs):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.is_at_least_1 = is_at_least_1
        self.is_mlp = is_mlp
        if self.is_mlp:
            self.gate_generator = MLP(self.input_size, self.output_size, self.output_size,
                                      **kwargs)
        else:
            self.gate_generator = nn.Linear(self.input_size, self.output_size,
                                            **kwargs)

        if not isinstance(initial_gates, list):
            initial_gates = [initial_gates / output_size] * output_size
        self.initial_gates = torch.tensor(initial_gates, dtype=torch.float, device=device)
        self.rounder = ConcreteRounding(**rounding_kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if not self.is_mlp:
            linear_init(self.gate_generator, "sigmoid")

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["input_size", "output_size"],
                              conditional_shows=["is_mlp", "is_at_least_1"])

    def forward(self, x):
        gates = self.gate_generator(x)
        gates = gates + self.initial_gates
        gates = torch.sigmoid(gates)
        gates = self.rounder(gates)

        loss = batch_reduction_f(gates, torch.mean)
        if self.is_at_least_1:
            loss = torch.relu(loss - 1. / gates.size(-1))

        return gates, loss


class Highway(Module):
    """
    Highway module, which does a weighted average of 2 vectors based on a weight
    generated from a third vector.

    Args:
        input_size (int, optional): input size of the vectors that will be used
            to generate the carry weight.
        output_size (int, optional): output size of the vectors for which to do
            a weighted average.
        initial_highway (float, optional): initial highway carry rate. This can
            be useful to make the network learn the attention even before the
            decoder converged.
        is_single_carry (bool, optional): whetehr to use a one dimension carry
            weight instead of n dimensional. If a n dimension then the network
            can learn to carry some dimensions but not others. The downside is that
            the number of parameters would be larger.
        is_additive_highway (bool, optional): whether to use a residual connection
            with a carry weight got th residue. I.e if `True` the carry weight will
            only be applied to the residue and will not scale the new value with
            `1-carry`.
        is_mlps (bool, optional): whether to use MLPs for the generators instead
            of a linear layer.
        min_hidden (int, optional): minimum number fof hidden neurons
            to use if using a MLP.
    """

    def __init__(self, input_size, output_size,
                 initial_highway=0.5,
                 is_single_carry=True,
                 is_additive_highway=False,
                 is_mlps=True,
                 min_hidden=16):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.initial_highway = initial_highway
        self.is_single_carry = is_single_carry
        self.is_additive_highway = is_additive_highway
        self.is_mlps = is_mlps
        self.min_hidden = min_hidden

        self.carry_size = 1 if self.is_single_carry else self.output_size

        if self.is_mlps:
            self.carrier = MLP(self.input_size,
                               self.min_hidden,
                               self.carry_size)
        else:
            self.carrier = nn.Linear(self.input_size, self.carry_size)

        self.carry_to_prob = ProbabilityConverter(initial_probability=self.initial_highway)

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["output_size"],
                              conditional_shows=["initial_highway", "is_single_carry",
                                                 "is_additive_highway", "is_mlps"])

    def forward(self, carry_input, x_old, x_new):

        carry_rates = self.carrier(carry_input)
        carry_rates = self.carry_to_prob(carry_rates)

        if self.is_additive_highway:
            x_new = x_new + carry_rates * x_old
        else:
            x_new = (1 - carry_rates) * x_new + carry_rates * x_old

        self.add_to_test(carry_rates, "carry_rates")
        self.add_to_visualize(carry_rates.mean(-1).mean(-1), "carry_rates")

        return x_new
