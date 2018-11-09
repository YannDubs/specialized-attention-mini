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
from seq2seq.util.confuser import confuse_keys_queries
from seq2seq.attention import KQGenerator, ValueGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(BaseRNN):
    """
    Applies a multi-layer KV-RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        embedding_size (int): the size of the embedding of input variables
        key_kwargs (dict, optional): additional arguments to the key generator.
        value_kwargs (dict, optional): additional arguments to the value generator.
        kwargs:
            Additional arguments to `get_rnn` and `BaseRNN`
    """

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size,
                 is_content_attn=True,
                 is_key=True,
                 is_value=True,
                 key_kwargs={},
                 value_kwargs={},
                 **kwargs):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         **kwargs)
        self.embedding_size = embedding_size
        self.is_content_attn = is_content_attn
        
        self.is_value = is_value
        self.is_key = is_key

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.controller, self.hidden0 = get_rnn(self.rnn_cell, self.embedding_size,
                                                self.hidden_size,
                                                batch_first=True,
                                                is_get_hidden0=True,
                                                **self.rnn_kwargs)

        if not self.is_content_attn and (self.is_key or self.is_query):
            self.is_key = False
            self.is_query = False

        if self.is_key:
            self.key_generator = KQGenerator(self.hidden_size, **key_kwargs)
            self.key_size = self.key_generator.output_size
        else:
            self.key_size = self.hidden_size

        if self.is_value:
            self.value_generator = ValueGenerator(self.hidden_size,
                                                  embedding_size,
                                                  **value_kwargs)
            self.value_size = self.value_generator.output_size
        else:
            self.value_size = self.hidden_size

        self.enc_counter = torch.arange(1, self.max_len + 1,
                                        dtype=torch.float,
                                        device=device)

        self.reset_parameters()

    def extra_repr(self):
        return ""

    def forward(self, input_var, input_lengths, confusers=dict()):
        """
        Applies a multi-layer KV-RNN to an input sequence.

        Args:
            input_var (tensor): tensor of shape (batch, seq_len) containing the
                features of the input sequence.
            input_lengths (tuple(list of int, torch.FloatTesnor)): A
                list that contains the lengths of sequences in the mini-batch. The
                Tensor has the same information but is preinitialized on the
                correct device.
            confusers (dictionary, optional): dictionary of confusers to use.
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
    
        if self.is_key:
            keys = self.key_generator(output, step=0)
        else:
            keys = output

        if "key_confuser" in confusers:
            confuse_keys_queries(input_lengths_tensor,
                                 keys,
                                 confusers["key_confuser"],
                                 self.enc_counter,
                                 self.training)

        self.add_to_test(keys, "keys")
       
        if self.is_value:
            values = self.value_generator(output, embedded)
        else:
            values = output

        last_control_out = output[:, -1:, :]

        return keys, values, hidden, last_control_out
