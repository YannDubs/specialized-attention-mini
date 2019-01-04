"""
Decoder class for a seq2seq.

TO DO:
    - Lot of refactoring here. I build step by step on the machine library, so
    a lot of the work feels (and is) donc in an ad-hoc way. everything should
    be made with the idea of having Key Value Query Position by default (although
    we should still keep the possibility to remove those in order to compare results
    although the vanilla one doesn't need to be otpimized: i.e we can use tricks
    to make it work in postioning framework.)
    - The attention classes should be initialized in the decoder but rather given
    the constructor of the decoder. Indeed we want the decoder to be general and not
    specific to our type of attention.

Contact : Yann Dubois
"""
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from seq2seq.attention import Attender
from seq2seq.util.helpers import (renormalize_input_length, get_rnn,
                                  get_extra_repr, format_source_lengths,
                                  recursive_update, add_to_test)
from seq2seq.util.torchextend import AnnealedGaussianNoise
from seq2seq.util.initialization import weights_init, init_param
from .baseRNN import BaseRNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoderRNN(BaseRNN):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        values_size (int, optional): size of the generated value. -1 means same
            as hidden size. Can also give percentage of hidden size betwen 0 and 1.
        _additional_to_store (list): keys from `additional` that should be
            stored in `ret_dict` if present.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size, sos_id, eos_id,
                 value_size=None,
                 attender=Attender,
                 attender_kwargs={},
                 _additional_to_store=["test", "visualize", "losses", "pos_perc"],
                 **kwargs):

        super().__init__(vocab_size, max_len, hidden_size, **kwargs)
        self.additional_to_store = _additional_to_store

        self.embedding_size = embedding_size
        self.output_size = vocab_size
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.value_size = value_size

        input_rnn_size = self.embedding_size + self.value_size

        self.rel_counter = torch.arange(0, self.max_len,
                                        dtype=torch.float,
                                        device=device).unsqueeze(1) / (self.max_len - 1)

        self.dec_counter = torch.arange(1, self.max_len + 1,
                                        dtype=torch.float,
                                        device=device)

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.controller = get_rnn("gru",
                                  input_rnn_size,
                                  self.hidden_size,
                                  batch_first=True,
                                  is_get_hidden0=False)

        self.attender = attender(self.hidden_size, self.max_len, **attender_kwargs)

        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.reset_parameters()

    def extra_repr(self):
        pass

    def forward(self,
                encoder_hidden,
                keys,
                values,
                last_enc_control_out,
                inputs=None,
                teacher_forcing_ratio=0,
                source_lengths=None,
                additional=None):

        ret_dict = dict()
        ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden,
                                                             teacher_forcing_ratio)

        additional, ret_dict = self._store_additional(additional, ret_dict)
        additional = self._initialize_additional(additional)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)
        controller_output = last_enc_control_out

        symbols = None
        for di in range(max_length):
            if di == 0 or use_teacher_forcing:
                # We always start with the SOS symbol as input. We need to add
                # extra dimension of length 1 for the number of decoder steps
                # (1 in this case) When we use teacher forcing, we always use the
                # target input.
                decoder_input = inputs[:, di].unsqueeze(1)

            else:
                # If we don't use teacher forcing (and we are beyond the first
                # SOS step), we use the last output as new input
                decoder_input = symbols

            (decoder_output,
             decoder_hidden,
             step_attn,
             controller_output) = self.forward_step(decoder_input,
                                                    decoder_hidden,
                                                    keys,
                                                    values,
                                                    controller_output,
                                                    di,
                                                    source_lengths=source_lengths)

            step_output = decoder_output.squeeze(1)

            # TO DO : don't rely on decoding to save the follwing variables
            # this should be done in `add_regularization_loss`
            additional["test"] = self.get_to_test()
            additional["visualize"] = self.get_to_visualize()
            additional["losses"] = self.get_regularization_losses()

            (symbols,
             decoder_outputs,
             ret_dict,
             sequence_symbols,
             lengths) = self._decode(decoder_outputs, ret_dict,
                                     sequence_symbols,
                                     lengths, di, step_output, step_attn,
                                     additional=additional)

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _store_additional(self, additional, ret_dict, is_multiple_call=False):
        """
        Store in `ret_dict` the values from `additional` associated with the
        keys of `additional_to_store`.

        Args:
            additional (dictionary): dictionary containing additional
                variables that are necessary for some hyperparamets.
            ret_dict (dictionary): dictionary that will be returned, in which to
                store the new values. The new values will be appended to a list
                at each step.
            is_multiple_call (bool, optional): whether will call the function
                multiple time with th same `additional_to_store`.
        """
        def append_to_list(dictionary, k, v):
            dictionary[k] = dictionary.get(k, list())
            dictionary[k].append(v)

        if additional is None:
            return None, ret_dict

        filtered_additional = {k: v for k, v in additional.items()
                               if k in self.additional_to_store}

        if not is_multiple_call:
            ret_dict = recursive_update(ret_dict, filtered_additional)
            # removes so that doesn't add them multiple time uselessly
            for k in filtered_additional.keys():
                additional.pop(k, None)
        else:
            for k, v in filtered_additional.items():
                if v is not None:
                    if isinstance(v, dict):
                        ret_dict[k] = ret_dict.get(k, dict())
                        for sub_k, sub_v in v.items():
                            append_to_list(ret_dict[k], sub_k, sub_v)
                    else:
                        append_to_list(ret_dict, k, v)

        return additional, ret_dict

    def _decode(self, decoder_outputs, ret_dict, sequence_symbols,
                lengths, step, step_output, step_attn, additional=None):
        """
        Args:
            decoder_outputs (list): list containing the output softmax distribution
                from all previous decoding steps.
            ret_dict (dictionary): dictionary that will be returned, in which to
                store the new values. The new values will be appended to a list
                at each step.
            sequence_symbols (list): list of sequences, where each sequence is a
                list of predicted token IDs
            lengths (list): list of integers representing lengths of output sequences.
            step (int): current decoding step.
            step_output (FloatTensor): current outputed softmax distribution.
            step_attn (list): current attention distribution of the decoder RNN.
            additional (dictionary): dictionary containing additional variables
                that are necessary for some hyperparamets.
        """

        decoder_outputs.append(step_output)
        ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

        additional, ret_dict = self._store_additional(additional, ret_dict,
                                                      is_multiple_call=True)

        symbols = decoder_outputs[-1].topk(1)[1]
        sequence_symbols.append(symbols)

        eos_batches = symbols.data.eq(self.eos_id)
        if eos_batches.dim() > 0:
            eos_batches = eos_batches.cpu().view(-1).numpy()
            update_idx = ((lengths > step) & eos_batches) != 0
            lengths[update_idx] = len(sequence_symbols)

        return symbols, decoder_outputs, ret_dict, sequence_symbols, lengths

    def forward_step(self, input_var, hidden, keys, values,
                     controller_output, step,
                     source_lengths=None):
        """
        Performs one or multiple forward decoder steps.

        Args:
            input_var (torch.LongTensor): Variable containing the input(s) to the
                decoder RNN. Shape : (batch_size, output_len).
            hidden (torch.FloatTensor): Variable containing the previous decoder
                hidden state.
            keys (torch.FloatTensor): Keys generated by the encoder.
            values (torch.FloatTensor): Values generated by the encoder.

        Returns:
            predicted_softmax (torch.FloatTensor): The output softmax distribution
                at every time step of the decoder RNN. Shape:
                (batch_size, output_len, input_len)
            hidden (torch.FloatTensor): The hidden state at every time step of
                the decoder RNN
            attn (torch.FloatTensor): The attention distribution at every time
                step of the decoder RNN
        """
        source_lengths_list, source_lengths_tensor = format_source_lengths(source_lengths)

        batch_size, output_len = input_var.size()

        embedded = self.embedding(input_var)

        embedded = self.input_dropout(embedded)

        context, attn = self._compute_context(controller_output,
                                              keys,
                                              values,
                                              source_lengths,
                                              step)

        controller_input = self._combine_context(embedded, context)

        controller_output, hidden = self.controller(controller_input, hidden)

        prediction_input = controller_output.contiguous().view(-1, self.out.in_features)

        predicted_softmax = F.log_softmax(self.out(prediction_input),
                                          dim=1).view(batch_size, output_len, -1)

        return predicted_softmax, hidden, attn, controller_output

    def _validate_args(self, inputs, encoder_hidden, teacher_forcing_ratio):
        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                hidden = encoder_hidden

                if self.rnn_cell == "lstm":
                    batch_size = hidden[0].size(1)
                elif self.rnn_cell == "gru":
                    batch_size = hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_len
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length

    def _compute_context(self, controller_output, keys, values, source_lengths,
                         step):
        source_lengths_list, source_lengths_tensor = format_source_lengths(source_lengths)

        batch_size = values.size(0)

        unormalized_counter = self.rel_counter.expand(batch_size, -1, 1)
        rel_counter_encoder = renormalize_input_length(unormalized_counter,
                                                       source_lengths_tensor - 1,
                                                       self.max_len - 1)

        attn = self.attender(keys, controller_output, source_lengths, step,
                             controller=controller_output)

        mean_attn = torch.bmm(attn,
                              rel_counter_encoder[:, :attn.size(2), :]
                              ).squeeze(2)

        context = torch.bmm(attn, values)

        self.add_to_visualize([mean_attn, step],
                              ["mean_attn", "step"])

        return context, attn

    def _initialize_additional(self, additional):
        if additional is None:
            additional = dict()

        return additional

    def _combine_context(self, input_var, context):
        combined_input = torch.cat((context, input_var), dim=2)

        return combined_input
