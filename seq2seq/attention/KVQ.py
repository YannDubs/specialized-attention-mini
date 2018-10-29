"""
Key Value Query Generator Classes.

Contact: Yann Dubois
"""
import math
import warnings

import torch
import torch.nn as nn

from seq2seq.util.initialization import get_hidden0, weights_init, replicate_hidden0
from seq2seq.util.helpers import renormalize_input_length, get_rnn, get_extra_repr
from seq2seq.util.base import Module
from seq2seq.util.torchextend import (MLP, ProbabilityConverter, AnnealedDropout,
                                      AnnealedGaussianNoise, Highway)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_size(size, hidden_size, name=None):
    if size == -1:
        return hidden_size
    elif 0 < size < 1:
        return math.ceil(size * hidden_size)
    elif 0 < size <= hidden_size:
        return size
    else:
        raise ValueError("Invalid size for {} : {}".format(name, size))


class BaseKeyValueQuery(Module):
    """Base class for quey query value generators.

    Args:
        input_size (int): size of the hidden activations of the controller,
            which will be given as input to the generator.
        output_size (int, optional): output size of the generator.
        is_contained_kv (bool, optional): whether or not to use different parts
            of the controller output as input for key and value generation.
        min_input_size (int, optional): minimum input size for the generator.
        is_mlps (bool, optional): whether to use MLPs for the generators instead
            of a linear layer.
        min_generator_hidden (int, optional): minimum number fof hidden neurons
            to use if using a MLP.
    """

    def __init__(self, input_size,
                 output_size=-1,
                 is_contained_kv=False,
                 min_input_size=32,
                 is_mlps=True,
                 min_generator_hidden=16):

        super(BaseKeyValueQuery, self).__init__()

        self.input_size = input_size
        self.output_size = _compute_size(output_size, self.input_size,
                                         name="output_size - {}".format(type(self).__name__))
        self.is_contained_kv = is_contained_kv
        self.min_input_size = min_input_size
        self.is_mlps = is_mlps
        self.min_generator_hidden = min_generator_hidden
        self.used_input_size = self._compute_used_input_size()

    def _compute_used_input_size(self):
        return (max(self.min_input_size, self.output_size)
                if self.is_contained_kv else self.input_size)

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["input_size", "output_size"],
                              conditional_shows=dict(is_contained_kv=False, is_mlps=True))


class KQGenerator(BaseKeyValueQuery):
    """Base class for quey query generators.

    Args:
        input_size (int): size of the hidden activations of the controller,
            which will be given as input to the generator.
        annealed_dropout_kwargs (float, optional): additional arguments to the
            annealed output dropout.
        annealed_noise_output_kwargs (float, optional): additional arguments to
            the annealed output noise.
        kwargs:
            Additional arguments for the `BaseKeyValueQuery` parent class.
    """

    def __init__(self, input_size,
                 annealed_dropout_output_kwargs={},
                 annealed_noise_output_kwargs={},
                 **kwargs):

        super(KQGenerator, self).__init__(input_size, **kwargs)

        if self.is_mlps:
            self.generator = MLP(self.used_input_size,
                                 self.min_generator_hidden,
                                 self.output_size)
        else:
            self.generator = nn.Linear(self.used_input_size,
                                       self.output_size)

        self.dropout_output = AnnealedDropout(**annealed_dropout_output_kwargs)
        self.noise_output = AnnealedGaussianNoise(**annealed_noise_output_kwargs)

        self.reset_parameters()

    def forward(self, controller_out, step):
        """Generates the key or query.

        Args:
            controller_out (torch.tensor): tensor of size (batch_size, input_length,
                hidden_size) containing the outputs of the controller (encoder
                or decoder for key and query respectively).
        """

        if self.is_contained_kv:
            input_generator = controller_out[:, :, :self.used_input_size]
        else:
            input_generator = controller_out

        kq = self.generator(input_generator)
        kq = self.dropout_output(kq, is_update=(step == 0))
        kq = self.noise_output(kq, is_update=(step == 0))

        return kq


class ValueGenerator(BaseKeyValueQuery):
    """Value generator class.

    Args:
        input_size (int): size of the hidden activations of the controller,
            which will be given as input to the generator.
        embedding_size (int): size of the embeddings.
        is_highway (bool, optional): whether to use a highway between the
            embedding an the output.
        highway_kwargs (dictionary, optional): additional arguments to highway.
        kwargs:
            Additional arguments for the `BaseKeyValueQuery` parent class.
    """

    def __init__(self, input_size, embedding_size,
                 is_highway=False,
                 highway_kwargs={},
                 **kwargs):

        super(ValueGenerator, self).__init__(input_size, **kwargs)

        if is_highway and embedding_size != self.output_size:
            warnings.warn("Using value_size == {} instead of {} bcause highway.".format(embedding_size, self.output_size))
            self.output_size = embedding_size
            self.used_input_size = self._compute_used_input_size()

        self.is_highway = is_highway

        if self.is_mlps:
            self.generator = MLP(self.used_input_size,
                                 self.min_generator_hidden,
                                 self.output_size)
        else:
            self.generator = nn.Linear(self.used_input_size, self.output_size)

        if self.is_highway:
            self.highway = Highway(self.used_input_size, self.output_size,
                                   min_hidden=self.min_generator_hidden,
                                   **highway_kwargs)

        self.reset_parameters()

    def extra_repr(self):
        parrent_repr = super().extra_repr()
        # must use dict in conditional_shows because this is a base class
        new_repr = get_extra_repr(self,
                                  conditional_shows=["is_highway"])
        if new_repr != "":
            parrent_repr += ", " + new_repr

        return parrent_repr

    def forward(self, encoder_out, embedded):
        """Generate the value.

        Args:
            encoder_out (torch.tensor): tensor of size (batch_size, input_length,
                hidden_size) containing the hidden activations of the encoder.
            embedded (torch.tensor): tensor of size (batch_size, input_length,
                embedding_size) containing the input embeddings.
        """

        if self.is_contained_kv:
            input_generator = encoder_out[:, :, -self.used_input_size:]
        else:
            input_generator = encoder_out

        values = self.generator(input_generator)

        if self.is_highway:
            values = self.highway(input_generator, embedded, values)

        return values
