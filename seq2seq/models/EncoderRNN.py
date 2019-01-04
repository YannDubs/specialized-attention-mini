"""
Encoder class for a seq2seq.

NOTA BENE:
     - Modified substantially from `Machine`.

Contact: Yann Dubois
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

from .baseRNN import BaseRNN

from seq2seq.util.initialization import replicate_hidden0, init_param, weights_init
from seq2seq.util.helpers import get_rnn, get_extra_repr, format_source_lengths
from seq2seq.util.torchextend import ProbabilityConverter
from seq2seq.attention.KVQ import ValueGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            Additional arguments to `get_rnn` and `BaseRNN`
    """

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size,
                 value_kwargs={},
                 **kwargs):
        super().__init__(vocab_size, max_len, hidden_size,
                         **kwargs)
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.controller, self.hidden0 = get_rnn(self.rnn_cell, self.embedding_size,
                                                self.hidden_size,
                                                batch_first=True,
                                                is_get_hidden0=True)

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

        hidden = replicate_hidden0(self.hidden0, batch_size)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        embedded_unpacked = embedded
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths_list,
                                                     batch_first=True)

        output, hidden = self.controller(embedded, hidden)

        embedded = embedded_unpacked
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        keys = output

        values = self.value_generator(output, embedded)

        last_control_out = output[:, -1:, :]

        return keys, values, hidden, last_control_out
