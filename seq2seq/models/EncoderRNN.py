"""
Encoder class for a seq2seq.

NOTA BENE:
     - Major difference is the value generator

Contact: Yann Dubois
"""
import math
import logging

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

from .baseRNN import BaseRNN

from seq2seq.util.initialization import replicate_hidden0, weights_init
from seq2seq.util.helpers import (get_rnn, get_extra_repr, format_source_lengths)
from seq2seq.util.base import Module
from seq2seq.util.torchextend import ProbabilityConverter, Highway

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def _compute_size(size, hidden_size, name=None):
    if size == -1:
        return hidden_size
    elif 0 < size < 1:
        return math.ceil(size * hidden_size)
    elif 0 < size <= hidden_size:
        return size
    else:
        raise ValueError("Invalid size for {} : {}".format(name, size))


class EncoderRNN(BaseRNN):
    """
    Applies a multi-layer KV-RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        embedding_size (int): the size of the embedding of input variables
        value_kwargs (dict, optional): additional arguments to the value generator.
        kwargs:
            Additional arguments to `BaseRNN`
    """

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size,
                 value_kwargs={},
                 **kwargs):
        super().__init__(vocab_size, max_len, hidden_size,
                         **kwargs)
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.controller, hidden0 = get_rnn(self.rnn_cell, self.embedding_size,
                                           self.hidden_size,
                                           batch_first=True,
                                           is_get_hidden0=True)
        if isinstance(hidden0, tuple):
            self.hidden0 = hidden0[0]
            self.cell0 = hidden0[1]
        else:
            self.hidden0 = hidden0
            self.cell0 = None

        self.value_generator = ValueGenerator(self.hidden_size,
                                              embedding_size,
                                              **value_kwargs)
        self.value_size = self.value_generator.output_size

        self.reset_parameters()

    def extra_repr(self):
        return ""

    def forward(self, input_var, input_lengths):
        """
        Applies a multi-layer KV-RNN to an input sequence.

        Args:
            input_var (tensor): tensor of shape (batch, seq_len) containing the
                features of the input sequence.
            input_lengths (tuple(list of int, torch.FloatTesnor)): A
                list that contains the lengths of sequences in the mini-batch. The
                Tensor has the same information but is preinitialized on the
                correct device.
        """

        input_lengths_list, input_lengths_tensor = format_source_lengths(input_lengths)

        batch_size = input_var.size(0)

        hidden0 = self.hidden0 if self.cell0 is None else (self.hidden0, self.cell0)
        hidden = replicate_hidden0(hidden0, batch_size)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        embedded_unpacked = embedded
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths_list,
                                                     batch_first=True)

        # output is all hidden layers n last layer
        # hidden is all layers at last time step
        output, hidden = self.controller(embedded, hidden)

        embedded = embedded_unpacked
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        keys = output

        values = self.value_generator(output, embedded)

        last_control_out = output[:, -1:, :]

        return keys, values, hidden, last_control_out


class ValueGenerator(Module):
    """Value generator class.

    Args:
        input_size (int): size of the hidden activations of the controller,
            which will be given as input to the generator.
        embedding_size (int): size of the embeddings.
        output_size (int, optional): output size of the generator.
        is_highway (bool, optional): whether to use a highway between the
            embedding an the output.
        highway_kwargs (dictionary, optional): additional arguments to highway.
        Generator (Module, optional): module to generate various values. It
            should be callable using `Generator(input_size, output_size)(x)`.
            By default `nn.Linear`.
        kwargs:
            Additional arguments for the `BaseKeyValueQuery` parent class.
    """

    def __init__(self, input_size, embedding_size,
                 output_size=-1,
                 is_highway=False,
                 highway_kwargs={},
                 Generator=nn.Linear,
                 is_force_highway=False,  # DEV MODE
                 **kwargs):

        super(ValueGenerator, self).__init__()

        self.input_size = input_size
        self.is_force_highway = is_force_highway
        self.output_size = _compute_size(output_size, self.input_size,
                                         name="output_size - {}".format(type(self).__name__))

        if (is_highway or self.is_force_highway) and embedding_size != self.output_size:
            logger.warning("Using value_size == {} instead of {} because highway.".format(embedding_size, self.output_size))
            self.output_size = embedding_size

        self.is_highway = is_highway

        self.generator = Generator(self.input_size, self.output_size)

        if self.is_highway:
            self.highway = Highway(self.input_size, self.output_size,
                                   save_name="value_gates", **highway_kwargs)

        self.reset_parameters()

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["input_size", "output_size"],
                              conditional_shows=["is_highway"])

    def forward(self, input_generator, embedded):
        """Generate the value.

        Args:
            input_generator (torch.tensor): tensor of size (batch_size, input_length,
                hidden_size) containing the hidden activations of the encoder.
            embedded (torch.tensor): tensor of size (batch_size, input_length,
                embedding_size) containing the input embeddings.
        """

        values = self.generator(input_generator)

        if self.is_force_highway:
            values = embedded

        elif self.is_highway:
            values = self.highway(values, embedded, input_generator)

        return values
