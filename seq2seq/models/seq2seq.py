"""
Seq2seq class.

NOTA BENE:
    - I changed only a few things here.
    - input_lengths should not use this tuple trick
"""
import torch

from seq2seq.util.helpers import get_extra_repr
from seq2seq.util.base import Module
from seq2seq import
from seq2seq.util.torchextend import GaussianNoise, AnnealedDropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2seq(Module):
    """Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        mid_dropout (dictionary, optonal): mid dropout ratio.

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        input_variable (Tensor, optional): tensor of shape (batch_size,
            max_bacth_len), of input token IDs. Examples that are shorter are
            padded with <pad>.
        input_lengths (list of int, optional): A list that contains the lengths
            of sequences in the mini-batch.
        target_variable (dictionary, optional): Dictionary containing possible
            targets. Example: "decoder_output", "encoder_input", "attention_target"
        teacher_forcing_ratio (float, optional): The probability that teacher
          forcing will be used.

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder, mid_dropout=0.5):
        super(Seq2seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.mid_dropout = nn.Dropout(p=mid_dropout)

    def forward(self, input_variable,
                input_lengths=None,
                target_variables=None,
                teacher_forcing_ratio=0):

        self._update_n_training_calls()

        # precomputes a float tensor of the source lengths as it will be used a lot
        # removes the need of having to change the variable from CPU to GPU
        # multiple times at each iter
        input_lengths_tensor = torch.FloatTensor(input_lengths).to(device)
        input_lengths = (input_lengths, input_lengths_tensor)

        if target_variables is not None:
            target_output = target_variables.get('decoder_output', None)
            # remove  preprended SOS step
            provided_attention = (target_variables['attention_target'][:, 1:]
                                  if 'attention_target' in target_variables else None)
        else:
            target_output = None
            provided_attention = None

        keys, values, encoder_hidden, last_enc_control_out = self.encoder(input_variable,
                                                                          input_lengths)

        if isinstance(encoder_hidden, tuple):
            # for lstm
            encoder_hidden = tuple(self.mid_dropout(el) for el in encoder_hidden)
        else:
            encoder_hidden = self.mid_dropout(encoder_hidden)

        (decoder_outputs,
         decoder_hidden,
         ret_dict) = self.decoder(encoder_hidden, keys, values, last_enc_control_out,
                                  inputs=target_output,
                                  teacher_forcing_ratio=teacher_forcing_ratio,
                                  source_lengths=input_lengths,
                                  provided_attention=provided_attention)

        ret_dict["test"].update(self.get_to_test())
        ret_dict["visualize"].update(self.get_to_visualize())
        ret_dict["losses"].update(self.get_regularization_losses())

        return decoder_outputs, decoder_hidden, ret_dict

    def extra_repr(self):
        pass
