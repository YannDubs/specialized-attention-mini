"""
Sets of useful helper objects.

Nota Bene:
    - I'm not sure I'm still using all of these helper functions.
    - Should order them / split them by groups.

Contact: Yann Dubois
"""

import os
import glob
import sys
import inspect
import math
import collections

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from seq2seq.util.initialization import get_hidden0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clamp(x,
          minimum=-float("Inf"),
          maximum=float("Inf"),
          is_leaky=False,
          negative_slope=0.01,
          hard_min=None,
          hard_max=None):
    """
    Clamps a tensor to the given [minimum, maximum] (leaky) bound, with
    an optional hard clamping.
    """
    lower_bound = ((minimum + negative_slope * (x - minimum))
                   if is_leaky else torch.zeros_like(x) + minimum)
    upper_bound = ((maximum + negative_slope * (x - maximum))
                   if is_leaky else torch.zeros_like(x) + maximum)
    clamped = torch.max(lower_bound, torch.min(x, upper_bound))

    if hard_min is not None or hard_max is not None:
        if hard_min is None:
            hard_min = -float("Inf")
        elif hard_max is None:
            hard_max = float("Inf")
        clamped = clamp(x, minimum=hard_min, maximum=hard_max, is_leaky=False)

    return clamped


class Clamper:
    """Clamp wrapper class. To bypass the lambda pickling issue."""

    def __init__(self,
                 minimum=-float("Inf"),
                 maximum=float("Inf"),
                 is_leaky=False,
                 negative_slope=0.01,
                 hard_min=None,
                 hard_max=None):
        self.minimum = minimum
        self.maximum = maximum
        self.is_leaky = is_leaky
        self.negative_slope = negative_slope
        self.hard_min = hard_min
        self.hard_max = hard_max

    def __call__(self, x):
        return clamp(x, minimum=self.minimum, maximum=self.maximum,
                     is_leaky=self.is_leaky, negative_slope=self.negative_slope,
                     hard_min=self.hard_min, hard_max=self.hard_max)


def clamp_regularize(x, is_leaky=False, reg_kwargs={}, **kwargs):
    """
    Clamps a tensor to the given [minimum, maximum] (leaky) bound, with
    an optional hard clamping. And computes a loss proportional to the clamping.
    """
    x_clamped = clamp(x, is_leaky=is_leaky, **kwargs)

    if is_leaky:
        x_clamped_noleak = clamp(x, is_leaky=False, **kwargs)
    else:
        x_clamped_noleak = x_clamped

    loss = batch_reduction_f(regularization_loss(x - x_clamped_noleak.detach(),
                                                 is_no_mean=True,
                                                 **reg_kwargs),
                             torch.mean)

    return x_clamped, loss


def leaky_noisy_clamp(x, minimum, maximum, **kwargs):
    """
    Clamps a tensor to the given [minimum, maximum] (leaky) bound, with
    an optional hard clamping. And generates noise when outside of bounds.
    """
    outside = (x < minimum) | (x > maximum)
    if outside.sum() > 0:
        outside = outside.float()
        x = clamp(x, minimum=minimum, maximum=maximum, is_leaky=True, **kwargs)
        x = x * (1 - outside) + outside * outside.normal_() * x
    return x


def identity(x):
    """simple identity function"""
    return x


def mean(l):
    """Return mean of list."""
    return sum(l) / len(l)


def recursive_update(dic, update):
    """Recursively update a dictionary `dic` with a new dictionary `update`."""
    for k, v in update.items():
        if isinstance(v, collections.Mapping):
            dic[k] = recursive_update(dic.get(k, {}), v)
        else:
            dic[k] = v
    return dic


def check_import(module, to_use=None):
    """Check whether the given module is imported."""
    if module not in sys.modules:
        if to_use is None:
            error = '{} module not imported. Try "pip install {}".'.format(module, module)
            raise ImportError(error)
        else:
            error = 'You need {} to use {}. Try "pip install {}".'.format(module, to_use, module)
            raise ImportError(error)


def rm_prefix(s, prefix):
    """Removes the prefix of a string if it exists."""
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s


def rm_dict_keys(dic, keys_to_rm):
    """remove a set of keys from a dictionary not in place."""
    return {k: v for k, v in dic.items() if k not in keys_to_rm}


def get_default_args(func):
    """Get the default arguments of a function as a dictionary."""
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def renormalize_input_length(x, input_lengths, max_len=1):
    """Given a tensor that was normalized by a constant value across the whole
        batch, normalizes it by a diferent value for each example in the batch.

    Note:
        - Should preallocate the lengths only once on GPU to speed up.

    Args:
        x (torch.tensor) tensor to normalize of any dimension and size as long
            as the batch dimension is the first one.
        input_lengths (list or torch.tensor) values used for normalizing the
            input, length should be `batch_size`.
        max_len (float, optional) previous constant value that was used to
            normalize the input.
    """
    if input_lengths is None:
        return x
    else:
        if not isinstance(input_lengths, torch.Tensor):
            input_lengths = torch.FloatTensor(input_lengths).to(device)
        input_lengths = atleast_nd(input_lengths, x.dim())
        return (x * max_len) / input_lengths


def atleast_nd(x, n):
    """Adds dimensions to x until reaches n."""
    while x.dim() < n:
        x = x.unsqueeze(-1)
    return x


def get_extra_repr(module, always_shows=[], conditional_shows=dict()):
    """Gets the `extra_repr` for a module.

    Note:
        All variables that you want to show have to be attributes of `module` with
        the same name. The name of the param in the function definition is not enough.

    Args:
        module (nn.Module): Module for which to get `extra_repr`.
        always_show (list of str): list of variables to always show.
        conditional_show (dictionary or list): variables to show depending on
            their values. Keys are the names, and variables the values
            they should not take to be shown. If a list then the condition
            is that their value is different from the default one in the constructor.
    """
    extra_repr = ""
    for show in always_shows:
        extra_repr += ", {0}={{{0}}}".format(show)

    if isinstance(conditional_shows, list):
        default_args = get_default_args(module.__class__)
        conditional_shows = {show: default_args[show] for show in conditional_shows}

    for show, condition in conditional_shows.items():
        if condition is None:
            if module.__dict__[show] is not None:
                extra_repr += ", {0}={{{0}}}".format(show)
        else:
            if module.__dict__[show] != condition:
                extra_repr += ", {0}={{{0}}}".format(show)

    extra_repr = rm_prefix(extra_repr, ", ")
    return extra_repr.format(**module.__dict__)


def get_rnn_cell(rnn_name):
    """Return the correct rnn cell."""
    if rnn_name.lower() == 'lstm':
        return nn.LSTM
    elif rnn_name.lower() == 'gru':
        return nn.GRU
    else:
        raise ValueError("Unsupported RNN Cell: {0}".format(rnn_name))


def apply_weight_norm(module):
    """Recursively apply weight norm to children of given module

    copied from : https://github.com/j-min/Adversarial_Video_Summary/blob/master/layers/weight_norm.py
    """
    if isinstance(module, nn.Linear):
        weight_norm(module, 'weight')
    if isinstance(module, (nn.RNNCell, nn.GRUCell, nn.LSTMCell)):
        weight_norm(module, 'weight_ih')
        weight_norm(module, 'weight_hh')
    if isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
        for i in range(module.num_layers):
            weight_norm(module, 'weight_ih_l{}'.format(i))
            weight_norm(module, 'weight_hh_l{}'.format(i))
            if module.bidirectional:
                weight_norm(module, 'weight_ih_l{}_reverse'.format(i))
                weight_norm(module, 'weight_hh_l{}_reverse'.format(i))


def get_rnn(rnn_name, input_size, hidden_size,
            is_weight_norm=False,
            is_get_hidden0=True,
            **kwargs):
    """Return an initialized rnn (and the initializized first hidden state)."""
    Rnn = get_rnn_cell(rnn_name)
    rnn = Rnn(input_size, hidden_size, **kwargs)
    if is_weight_norm:
        apply_weight_norm(rnn)
    if is_get_hidden0:
        return rnn, get_hidden0(rnn)
    return rnn


def format_source_lengths(source_lengths):
    if isinstance(source_lengths, tuple):
        source_lengths_list, source_lengths_tensor = source_lengths
    else:
        source_lengths_list, source_lengths_tensor = None, None

    return source_lengths_list, source_lengths_tensor


def apply_along_dim(f, X, dim=0, **kwargs):
    """
    Applies a function along the given dimension.
    Might be slow because list comprehension.
    """
    tensors = [f(x, **kwargs) for x in torch.unbind(X, dim=dim)]
    out = torch.stack(tensors, dim=dim)
    return out


def get_indices(l, keys):
    """
    Returns a list of the indices associated with each `keys.
    SLow as O(K*N).
    """
    out = []
    for k in keys:
        try:
            out.append(l.index(k))
        except ValueError:
            pass
    return out


### CLASSES ###

class HyperparameterInterpolator:
    """Helper class to compute the value of a hyperparameter at each training step.

    Args:
        initial_value (float): initial value of the hyperparameter.
        final_value (float): final value of the hyperparameter.
        n_steps_interpolate (int): number of training steps before reaching the
            `final_value`.
        start_step (int, optional): number of steps to wait for before starting annealing.
            During the waiting time, the hyperparameter will be `default`.
        default (float, optional): default hyperparameter value that will be used
            for the first `start_step`s. If `None` uses `initial_value`.
        mode (str, optional): interpolation mode. One of {"linear", "geometric"}.
    """

    def __init__(self, initial_value, final_value, n_steps_interpolate,
                 start_step=0,
                 default=None,
                 mode="linear"):
        if n_steps_interpolate < 0:
            # quick trick to swith final / initial
            n_steps_interpolate *= -1
            initial_value, final_value = final_value, initial_value

        self.initial_value = initial_value
        self.final_value = final_value
        self.n_steps_interpolate = n_steps_interpolate
        self.start_step = start_step
        self.default = default if default is not None else self.initial_value
        self.mode = mode.lower()
        self.is_interpolate = not (self.initial_value == self.final_value or
                                   self.n_steps_interpolate == 0)

        self.n_training_calls = 0

        if self.is_interpolate:
            if self.mode == "linear":
                delta = (self.final_value - self.initial_value)
                self.factor = delta / self.n_steps_interpolate
            elif self.mode == "geometric":
                delta = (self.final_value / self.initial_value)
                self.factor = delta ** (1 / self.n_steps_interpolate)
            else:
                raise ValueError("Unkown mode : {}.".format(mode))

    def reset_parameters(self):
        """Reset the interpolator."""
        self.n_training_calls = 0

    def extra_repr(self, value_name="value"):
        """
        Return a a string that can be used by `extra_repr` of a parent `nn.Module`.
        """
        if self.is_interpolate:
            txt = 'initial_{0}={1}, final_{0}={2}, n_steps_interpolate={3}, {4}'
            txt = txt.format(value_name,
                             self.initial_value,
                             self.final_value,
                             self.n_steps_interpolate, self.mode)
        else:
            txt = "{}={}".format(value_name, self.final_value)
        return txt

    @property
    def is_annealing(self):
        return (self.is_interpolate) and (
            self.start_step <= self.n_training_calls) and (
            self.n_training_calls <= (self.n_steps_interpolate + self.start_step))

    def __call__(self, is_update):
        """Return the current value of the hyperparameter.

        Args:
            is_update (bool): whether to update the hyperparameter.
        """
        if not self.is_interpolate:
            return self.final_value

        if is_update:
            self.n_training_calls += 1

        if self.start_step >= self.n_training_calls:
            return self.default

        n_actual_training_calls = self.n_training_calls - self.start_step

        if self.is_annealing:
            current = self.initial_value
            if self.mode == "geometric":
                current *= (self.factor ** n_actual_training_calls)
            elif self.mode == "linear":
                current += self.factor * n_actual_training_calls
        else:
            current = self.final_value

        return current


class HyperparameterCurriculumInterpolator:
    """Helper class to compute the value of a hyperparameter at each training step
    given a curriculum.

    Args:
        curriculum (list of tuple): list of dict("step", "value", ["mode"]),
            defining the the points in betweeen which to use interpolation. If the
            first element doesn't start at step 0 then will be constant with value
            equal to the value of the first point in the curriculum.
        default_mode (str, optional): default interpolation mode when mode not
            given in `curriculum`. One of {"linear", "geometric"}.
    """

    def __init__(self, curriculum, default_mode="linear"):
        if curriculum[0]["step"] > 0:
            curriculum = [dict(step=0, value=curriculum[0]["value"])] + curriculum
        self.curriculum = curriculum
        self.default_mode = default_mode

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the interpolator."""
        self.n_training_calls = 0
        self.curriculum = self.curriculum
        self.current_interpolator = None
        self.future_curriculums = self.curriculum

    @property
    def next_step(self):
        """Return the next curriculum step."""
        if self.future_curriculums == []:
            return float("inf")
        return self.future_curriculums[0]["step"]

    def new_interpolator(self):
        """
        Sets the new sub-interpolator for the current sub-curriculum.
        """
        current_curriculum = self.future_curriculums.pop(0)
        if len(self.future_curriculums) != 0:
            next_curriculum = self.future_curriculums[0]
        else:
            next_curriculum = current_curriculum
        n_steps_interpolate = next_curriculum["step"] - current_curriculum["step"]

        mode = current_curriculum.get("mode", self.default_mode)

        self.current_interpolator = HyperparameterInterpolator(current_curriculum["value"],
                                                               next_curriculum["value"],
                                                               n_steps_interpolate,
                                                               mode=mode)

    def extra_repr(self, value_name="value"):
        """
        Return a a string that can be used by `extra_repr` of a parent `nn.Module`.
        """
        txt = '{}_curriculum={}, {}'
        txt = txt.format(value_name, self.curriculum, self.default_mode)
        return txt

    @property
    def is_annealing(self):
        """Whether you are currently annealing the hyperparameters."""
        return self.current_interpolator.is_annealing

    def __call__(self, is_update):
        """Return the current value of the hyperparameter.

        Args:
            is_update (bool): whether to update the hyperparameter.
        """
        if self.n_training_calls == self.next_step:
            is_new_interpolator = True
            self.new_interpolator()
        else:
            is_new_interpolator = False

        if is_update:
            self.n_training_calls += 1

        return self.current_interpolator(is_update and not is_new_interpolator)


class Rate2Steps:
    """Convert interpolating rates to steps useful for annealing.

    Args:
        total_training_calls (int): total number of training steps.
    """

    def __init__(self, total_training_calls):
        self.total_training_calls = total_training_calls

    def __call__(self, rate):
        """
        Convert a rate to a number of steps.

        Args:
            rate (float): rate to convert in [0,1]. If larger than one then considered
                as already being a number of steps, i.e returns the raw input.
        """
        if rate > 1:
            return rate
        return math.ceil(rate * self.total_training_calls)


def l0_loss(x, temperature=10, is_leaky=True, negative_slope=0.01, dim=None,
            keepdim=False, is_no_mean=False):
    """Compute the approximate differentiable l0 loss of a matrix.

    Note:
        Uses an absolute value of a tanh which is 0 when x == 0, and 1 for larger
        values of `x`.

    Args:
        temperature (float, optional): controls the spikeness of the differentiable
            l0_loss. When teperature -> infinty, we recover the rel l0_loss.
        is_leaky (bool, optional): whether to use a leaky l0-loss, i.e penalizing
            a bit more larger values. This is useful for gradient propagations.
        negative_slope (float, optional): negative slope of the leakiness.
        dim (int, optional): the dimension to reduce. By default flattens `x`
            then reduces to a scalar.
        keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not.
    """
    norm = torch.abs(torch.tanh(temperature * x))
    if is_leaky:
        norm = norm + torch.abs(negative_slope * x)

    if is_no_mean:
        return norm
    if dim is None:
        return norm.mean(keepdim=keepdim)
    else:
        return norm.mean(dim=dim, keepdim=keepdim)


def regularization_loss(x,
                        min_x=0.,
                        p=2.,
                        dim=None,
                        lower_bound=1e-4,
                        is_no_mean=False,
                        **kwargs):
    """Compute the regularization loss.

    Args:
        p (float, optional): element wise power to apply. All of those have been
            made differentiable (even `p=0`).
        min_x (float, optional): if `abs(x)<min_x` then don't regularize.
            Always positive.
        dim (int, optional): the dimension to reduce. By default flattens `x`
            then reduces to a scalar.
        lower_bound (float, optional): lower bounds the absolute value of a entry
            of x when p<1 to avoid exploding gradients. Note that by lowerbounding
            the gradients will be 0 for these values (as tehy were clamped).
        is_no_mean (bool, optional): does not compute the mean over any dimension.
        kwargs:
            Additional parameters to `l0_norm`.
    """
    if min_x > 0:
        to_reg = (torch.abs(x) >= min_x).float()
        x = x * to_reg

    if p < 1:
        x = abs_clamp(x, lower_bound)

    if p == 0:
        return l0_loss(x, dim=None, is_no_mean=is_no_mean, **kwargs)

    loss = (torch.abs(x)**p)

    if not is_no_mean:
        if dim is None:
            loss = loss.mean()
        else:
            loss = loss.mean(dim=dim)

    return loss


def abs_clamp(x, lower_bound):
    """Lowerbounds the absolute value of a tensor."""
    sign = x.sign()
    lower_bounded = torch.max(x * sign, torch.ones_like(x) * lower_bound
                              ) * sign
    return lower_bounded


def batch_reduction_f(x, f, batch_first=True, **kwargs):
    """Applies a reduction function `fun` batchwise, i.e output will be of len=batch."""
    if x.dim() <= 1:
        return x
    if not batch_first:
        x = x.transpose(1, 0).contiguous()
    return f(x.view(x.size(0), -1), dim=1, **kwargs)


# TO DO : temporary trick -> this should be a callback
def add_to_visualize(values, keys, to_visualize, is_training, training_step,
                     save_every_n_batches=5):
    """Every `save_every_n_batches` batch, adds a certain variable to the `visualization`
    sub-dictionary of additional. Such variables should be the ones that are
    interpretable, and for which the size is independant of the source length.
    I.e avaregae over the source length if it is dependant.

    The variables will then be averaged over decoding step and over batch_size.

    Note : the function simpyl sets values. Doesn't append values.
    """
    if is_training:
        if training_step % save_every_n_batches == 0:
            if isinstance(keys, list):
                for k, v in zip(keys, values):
                    add_to_visualize(v, k, to_visualize, is_training, training_step,
                                     save_every_n_batches=save_every_n_batches)
            else:
                # averages over the batch size
                if isinstance(values, torch.Tensor):
                    values = values.mean(0).detach().cpu()
                to_visualize[keys] = values


# TO DO : temporary trick -> this should be a callback
def add_to_test(values, keys, to_test, is_dev_mode):
    """
    Save a variable to additional["test"] only if dev mode is on. The
    variables saved should be the interpretable ones for which you want to
    know the value of during test time.

    Batch size should always be 1 when predicting with dev mode !
    """
    if is_dev_mode:
        if isinstance(keys, list):
            for k, v in zip(keys, values):
                add_to_test(v, k, to_test, is_dev_mode)
        else:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu()
            to_test[keys] = values


class SummaryStatistics:
    """Computes the summary statistics of a vector.

    Args:
        statistics_name ({list,"all","pos"}, optional): name of the statistics to
            use. Use "all" instead of a list to use all functions that are defined
            for any values of xi's. Use "pos" to use all functions plus the ones
            that are only defined for stricly positive xi's.
    """

    def __init__(self, statistics_name=["min", "max", "mean", "median"]):
        all_stats = ["min", "max", "mean", "median", "std", "mad", "skew",
                     "kurtosis", "rms", "logsumexp", "absmean"]
        pos_stats = ["min", "max", "mean", "median", "std", "mad", "skew",
                     "kurtosis", "rms", "logsumexp", "absmean", "gmean", "hmean"]
        if statistics_name == "all":
            statistics_name = all_stats
        elif statistics_name == "positive":
            statistics_name = pos_stats

        self.statistics_name = statistics_name
        self.n_statistics = len(self.statistics_name)

        self.std = ExtendedStd()
        self.skew = ExtendedSkewness()
        self.kurtosis = ExtendedKurtosis()

    def __call__(self, x):
        stats = []

        n = x.size(-1)
        median = torch.median(x, dim=-1, keepdim=True)[0]

        if "min" in self.statistics_name:
            stats.append(torch.min(x, dim=-1, keepdim=True)[0])

        if "max" in self.statistics_name:
            stats.append(torch.max(x, dim=-1, keepdim=True)[0])

        if "mean" in self.statistics_name:
            stats.append(torch.mean(x, dim=-1, keepdim=True))

        if "median" in self.statistics_name:
            stats.append(median)

        if "std" in self.statistics_name:
            stats.append(self.std(x, dim=-1, keepdim=True))

        if "mad" in self.statistics_name:
            stats.append(torch.median(torch.abs(x - median), dim=-1, keepdim=True)[0])

        if "rms" in self.statistics_name:
            rms = torch.norm(x, p=2, dim=-1, keepdim=True) / (n**0.5)
            stats.append(rms)

        if "absmean" in self.statistics_name:
            absolute_mean = torch.norm(x, p=1, dim=-1, keepdim=True) / n
            stats.append(absolute_mean)

        if "skew" in self.statistics_name:
            stats.append(self.skew(x, dim=-1, keepdim=True))

        if "kurtosis" in self.statistics_name:
            stats.append(self.kurtosis(x, dim=-1, keepdim=True))

        if "logsumexp" in self.statistics_name:
            stats.append(torch.logsumexp(x, dim=-1, keepdim=True))

        if "gmean" in self.statistics_name:
            if torch.any(x <= 0):
                raise ValueError("Geometric mean only implemented if all elements greater than zero")
            geometric_mean = torch.exp(torch.log(x).mean(dim=-1, keepdim=True))
            stats.append(geometric_mean)

        if "hmean" in self.statistics_name:
            if torch.any(x <= 0):
                raise ValueError("Harmonic mean only defined if all elements greater than zero")
            harmonic_mean = n / (1.0 / x).sum(dim=-1, keepdim=True)
            stats.append(harmonic_mean)

        return torch.cat(stats, dim=-1)


def is_constant(x):
    """Whether a tensor has all th3 same values."""
    return torch.any(x == x[0])


class ExtendedStd(nn.Module):
    """
    Generalizes the standard deviation function by extending it through limits
    at the point x = 0, with dx=0.
    """

    def __init__(self):
        super(ExtendedStd, self).__init__()
        self.register_backward_hook(mask_infinite_backward_hook)

    def forward(self, x, dim=-1, unbiased=True, **kwargs):
        if x.size(dim) == 1:
            # division by 0 if unbiaised with sinle sample
            unbiased = False
        return torch.std(x, dim=dim, unbiased=unbiased, **kwargs)


class ExtendedSkewness(nn.Module):
    """
    Generalizes the skewness function by extending it through limits
    at the point x = 0, with dx=0.
    """

    def __init__(self):
        super(ExtendedSkewness, self).__init__()
        self.std = ExtendedStd()
        self.register_backward_hook(mask_infinite_backward_hook)

    def forward(self, x, dim=-1, keepdim=False, **kwargs):
        n = x.size(dim)
        biased_std = self.std(x, dim=dim, unbiased=True, keepdim=keepdim)
        mu = torch.mean(x, dim=dim, keepdim=keepdim)
        mask_not_zero = (~(biased_std == 0)).float()
        biased_std = (1 - mask_not_zero) + biased_std
        fisher_coef_skew = ((x - mu)**3).mean(dim=-1, keepdim=True) / biased_std**3
        adjust = ((n * (n - 1))**0.5) / (n - 2) if n > 2 else 1
        return adjust * fisher_coef_skew * mask_not_zero


class ExtendedKurtosis(nn.Module):
    """
    Generalizes the skewness function by extending it through limits
    at the point x = 0, with dx=0.
    """

    def __init__(self):
        super(ExtendedKurtosis, self).__init__()
        self.std = ExtendedStd()
        self.register_backward_hook(mask_infinite_backward_hook)

    def forward(self, x, dim=-1, keepdim=False, **kwargs):
        n = x.size(dim)
        biased_std = self.std(x, dim=dim, unbiased=True, keepdim=keepdim)
        mu = torch.mean(x, dim=dim, keepdim=keepdim)
        mask_not_zero = (~(biased_std == 0)).float()
        biased_std = (1 - mask_not_zero) + biased_std
        kurtosis = ((x - mu)**4).mean(dim=-1, keepdim=True) / biased_std**4
        if n > 3:
            kurtosis = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * kurtosis + 6)
        return kurtosis * mask_not_zero


def mask_infinite_backward_hook(self, grad_input, grad_output):
    """Masks nan gradients during the backward pass."""
    mask_new_infinite = ~torch.isfinite(grad_input[0]) & torch.isfinite(grad_output[0])
    grad_input0 = grad_input[0].masked_fill(mask_new_infinite, 0.0)
    return (grad_input0, ) + tuple(gi for gi in grad_input[1:])


def get_latest_file(path):
    """Return the latest modified/added file in a path."""
    list_of_files = glob.glob(os.path.join(path, "*"))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def bound_probability(x, min_p):
    """Bound a probability from [0,1] to [min_p, 1-min_p]."""
    if (x < 0).any() or (x > 1).any():
        x_outside = x[x < 0] if (x < 0).any() else x[x > 1]
        raise ValueError("x={} is not in [0,1]. Value outside bounds : {}.".format(x, x_outside))
    if min_p < 0 or min_p > 1:
        raise ValueError("min_p={} is not in [0,1].".format(min_p))

    range_p = 1 - min_p * 2
    new_p = x * range_p + min_p
    return new_p


def inv_sigmoid(p):
    """Return the inverse sigmoid."""
    return torch.log(p/(1-p))
