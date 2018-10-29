"""
Seq2seq class.

NOTA BENE:
    - I changed only a few things here.
    - input_lengths should not be this tuple trick
"""
import torch

from seq2seq.util.helpers import get_extra_repr
from seq2seq.util.base import Module
from seq2seq.util.torchextend import AnnealedGaussianNoise, AnnealedDropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2seq(Module):
    """Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        mid_dropout_kwargs (dictionary, optonal): additional arguments to mid dropout.
        mid_noise_kwargs (dictionary, optonal): additional arguments to mid noise.
        is_dev_mode (bool, optional): whether to store many useful variables in
            `additional`. Useful when predicting with a trained model in dev mode
             to understand what the model is doing. Use with `dev_predict`.

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

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

    def __init__(self, encoder, decoder,
                 mid_dropout_kwargs={},
                 mid_noise_kwargs={}):
        super(Seq2seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.mid_dropout = AnnealedDropout(**mid_dropout_kwargs)
        self.is_update_mid_dropout = self.training
        self.mid_noise = AnnealedGaussianNoise(**mid_noise_kwargs)
        self.is_update_mid_noise = self.training

    def forward(self, input_variable,
                input_lengths=None,
                target_variables=None,
                teacher_forcing_ratio=0,
                confusers=dict()):

        self._update_n_training_calls()

        # precomputes a float tensor of the source lengths as it will be used a lot
        # removes the need of having to change the variable from CPU to GPU
        # multiple times at each iter
        input_lengths_tensor = torch.FloatTensor(input_lengths).to(device)
        input_lengths = (input_lengths, input_lengths_tensor)

        # Unpack target variables
        try:
            target_output = target_variables.get('decoder_output', None)
        except AttributeError:
            target_output = None

        keys, values, encoder_hidden, last_enc_control_out = self.encoder(input_variable,
                                                                          input_lengths,
                                                                          confusers=confusers)

        self.is_update_mid_dropout = self.training
        self.is_update_mid_noise = self.training

        last_enc_control_out = self._mid_noise(last_enc_control_out)
        last_enc_control_out = self._mid_dropout(last_enc_control_out)

        if isinstance(encoder_hidden, tuple):
            # for lstm
            encoder_hidden = tuple(self._mid_noise(el) for el in encoder_hidden)
            encoder_hidden = tuple(self._mid_dropout(el) for el in encoder_hidden)
        else:
            encoder_hidden = self._mid_noise(encoder_hidden)
            encoder_hidden = self._mid_dropout(encoder_hidden)

        (decoder_outputs,
         decoder_hidden,
         ret_dict) = self.decoder(encoder_hidden, keys, values, last_enc_control_out,
                                  inputs=target_output,
                                  teacher_forcing_ratio=teacher_forcing_ratio,
                                  source_lengths=input_lengths,
                                  confusers=confusers)

        ret_dict["test"].update(self.get_to_test())
        ret_dict["visualize"].update(self.get_to_visualize())
        ret_dict["losses"].update(self.get_regularization_losses())

        return decoder_outputs, decoder_hidden, ret_dict

    def _mid_dropout(self, x):
        x = self.mid_dropout(x, is_update=self.is_update_mid_dropout)
        self.is_update_mid_dropout = False  # makes sure that updates only once every forward
        return x

    def _mid_noise(self, x):
        x = self.mid_noise(x, is_update=self.is_update_mid_noise)
        self.is_update_mid_noise = False  # makes sure that updates only once every forward
        return x

    def extra_repr(self):
        pass
