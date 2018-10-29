"""
Initialization helper functions

Contact: Yann Dubois
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {nn.LeakyReLU: "leaky_relu", nn.ReLU: "relu", nn.Tanh: "tanh", nn.Sigmoid: "sigmoid", nn.Softmax: "sigmoid"}
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def init_param(param, activation=None, is_positive=False, bound=0.05):
    """Initialize some parameters of the model that are not part of a children module.

    Args:
        param (nn.Parameters): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `param`.
        is_positive (bool, optional): whether to initilize only with positive values.
        bound (float, optional): maximum absolute value of the initealized values.
            By default `0.05` which is keras default uniform bound.
    """
    gain = get_gain(activation)
    if is_positive:
        return nn.init.uniform_(param, 1e-5, bound * gain)

    return nn.init.uniform_(param, -bound * gain, bound * gain)


def linear_init(layer, activation=None):
    """Initialize a linear layer.

    Args:
        layer (nn.Linear): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `layer`.
    """
    x = layer.weight

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def init_basernn(cell):
    """Initialize a simple RNN."""
    cell.reset_parameters()

    # orthogonal initialization of all recurrent weights
    for ih, hh, _, _ in cell.all_weights:
        # loops over all wight matrix even though concatenated in single matrix
        for i in range(0, hh.size(0), cell.hidden_size):
            nn.init.xavier_uniform_(ih[i:i + cell.hidden_size])
            nn.init.orthogonal_(hh[i:i + cell.hidden_size])


def init_gru(cell):
    """Initialize a GRU."""
    init_basernn(cell)

    # -1 reset gate biais of GRU
    for _, _, ih_b, hh_b in cell.all_weights:
        n_row = ih_b.size(0)
        ih_b[: n_row // 3].data.fill_(-1.0)
        hh_b[: n_row // 3].data.fill_(-1.0)


def init_lstm(cell):
    """Initialize a LSTM."""
    init_basernn(cell)

    # +1 forget gate bias (Jozefowicz et al., 2015)
    for _, _, ih_b, hh_b in cell.all_weights:
        n_row = ih_b.size(0)
        ih_b[n_row // 4: n_row // 2].data.fill_(1.0)
        hh_b[n_row // 4: n_row // 2].data.fill_(1.0)


def get_hidden0(rnn):
    """Return an initial state of the hidden actiavtion of a RNN."""

    n_stacked = rnn.num_layers
    if rnn.bidirectional:
        n_stacked *= 2

    hidden = Parameter(torch.randn(n_stacked, 1, rnn.hidden_size))
    init_param(hidden, activation="sigmoid")

    if isinstance(rnn, nn.LSTM):
        cell = init_param(hidden.clone(), activation="tanh")
        return (hidden, cell)
    else:
        return hidden


def replicate_hidden0(hidden, batch_size):
    """Replicate the initial hidden state for batch work."""

    # Note : hidden is not transposed even when using batch_first
    if isinstance(hidden, tuple):
        return replicate_hidden0(hidden[0], batch_size), replicate_hidden0(hidden[1], batch_size)

    return hidden.expand(hidden.size(0), batch_size, hidden.size(2)).contiguous()


def weights_init(module):
    """Initialize the weights of a module."""
    try:
        module.reset_parameters()
    except AttributeError:
        pass

    if isinstance(module, nn.Embedding):
        init_param(module.weight)
    elif isinstance(module, nn.LSTM):
        init_lstm(module)
    elif isinstance(module, nn.GRU):
        init_gru(module)
    elif isinstance(module, nn.Linear):
        linear_init(module)
