"""
General attender.

TO DO:

Contact: Yann Dubois
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from seq2seq.util.helpers import (renormalize_input_length, batch_reduction_f,
                                  get_extra_repr, format_source_lengths)
from seq2seq.util.torchextend import (MLP, ConcreteRounder, get_rounder, get_gate,
                                      StochasticRounder, ProbabilityConverter)
from seq2seq.util.initialization import weights_init
from seq2seq.util.base import Module
from seq2seq.attention.content import ContentAttender
from seq2seq.attention.location import LocationAttender


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attender(Module):
    """Computes the final attention using content and locaton.

    Args:
        controller_size (int): size of the controller tensor.
        max_len (int, optional): a maximum allowed length for the sequence to be
            processed.
        loc_query_size (int, optional) dimension to use for the location query size.
        content_kwargs (dict, optional): additional arguments to `ContentAttender`.
        location_kwargs (dict, optional): additional arguments to `LocationAttender`.
        mixer_kwargs (dict, optional): additional arguments to `AttentionMixer`.
    """

    def __init__(self, controller_size,
                 max_len=50,
                 loc_query_size=64,
                 content_kwargs={},
                 location_kwargs={},
                 mixer_kwargs={}):

        super(Attender, self).__init__()

        self.max_len = max_len
        self.resizer = nn.Linear(controller_size, loc_query_size)
        self.content_attender = ContentAttender(controller_size, **content_kwargs)
        self.location_attender = LocationAttender(loc_query_size, max_len, **location_kwargs)
        self.attn_mixer = AttentionMixer(controller_size,
                                         **mixer_kwargs)

        self.rel_counter = torch.arange(0, self.max_len,
                                        dtype=torch.float,
                                        device=device).unsqueeze(1) / (self.max_len - 1)

        self.reset_parameters()

    def extra_repr(self):
        pass

    def forward(self, keys, query,
                source_lengths=None, step=None, controller=None, **kwargs):
        """Compute and return the final attention.

        Args:
            key (torch.tensor): key of size (batch_size, n_queries, kq_size).
            query (torch.tensor): query of size (batch_size, n_queries, kq_size).
            step (int): current decoding step.
            source_lengths (tuple(list of int, torch.FloatTesnor), optional): A
                list that contains the lengths of sequences in the mini-batch. The
                Tensor has the same information but is preinitialized on teh
                correct device.
            controller (torch.tensor, optional): controller tensor of size
                (batch_size, n_queries, kq_size) used for the generation of some
                variables. Can be `None` if unused.

        Return:
            attn (torch.tensor): attention of size (batch_size, n_queries, n_keys).
        """
        assert source_lengths is not None
        assert step is not None
        assert controller is not None

        content_attn, conf_content = self.content_attender(keys, query)
        query = self.resizer(query)
        query = F.relu(query)

        old_attn = self.storer["old_attn"] if step != 0 else None
        loc_attn, conf_loc = self.location_attender(query, source_lengths, step,
                                                    old_attn)

        self.add_to_visualize([conf_content, conf_loc], ["content_confidence", "loc_confidence"])
        self.add_to_test([content_attn, loc_attn], ["content_attention", "loc_attention"])

        attn = self.attn_mixer(content_attn, loc_attn,
                               conf_content=conf_content,
                               conf_loc=conf_loc,
                               controller=controller)
        self.storer["old_attn"] = attn

        return attn

    def load_locator(self, file):
        """
        Loads a pretrained locator (output from self.save_locator) for transfer
        learning.
        """
        self.location_attender.load_locator(file)

    def save_locator(self, file):
        """Save the pretrained locator to a file."""
        self.location_attender.save_locator(file)

    def named_params_locator(self):
        """Return a generator of named parameters and parameters of the location
        attender"""
        return self.location_attender.named_parameters()


class AttentionMixer(Module):
    """Mixes the content and location attention.

    Args:
        controller_size (int, optional): size of the hidden activations of the
        decoder.
        mode ({"generate","confidence","loc_conf"}, optional) how to get
            `perc_loc`. `"generate"` will generate it from the controller, giving
            good but less interpetable results. `"confidence"` uses
            `perc_loc = conf_loc / (conf_loc + conf_content)`, forcing meaningfull
            confidences for both attentions. The latter should not be used without
            sequential attention because `perc_loc will always be 0.5 if both are
            confident. `"loc_conf"` will directly use `perc_loc=conf_loc`, forcing
            meaningfull positioning confidence but not the content ones. This says
            to the network to always prefer the location attention as it's more
            extrapolable.
        Generator (Module, optional): module to generate various values. It
            should be callable using `Generator(input_size, output_size)(x)`.
            By default `nn.Linear`.
        n_steps_wait (float, optional): number of training steps to wait
            for before starting to generate the positional percentage. Until then
            will use `dflt_perc_loc`.
        default_perc_loc (float, optional): constant positional percentage to
            use while `n_steps_wait`.
        rounder_perc_kwargs (dictionary, optional): Additional arguments to
            the percentage rounder.
        kwargs:
            additional arguments to the Generator.
    """

    def __init__(self, controller_size,
                 mode="loc_conf",
                 Generator=nn.Linear,
                 n_steps_wait=0,
                 dflt_perc_loc=0.5,
                 rounder_perc_kwargs={},
                 **kwargs):

        super().__init__()

        self.mode = mode.lower()
        self.n_steps_wait = n_steps_wait
        self.dflt_perc_loc = torch.tensor(dflt_perc_loc,
                                          dtype=torch.float,
                                          device=device)

        self.rounder_perc = get_rounder(**rounder_perc_kwargs)  # DEV MODE

        if self.mode == "generate":
            self.generator = Generator(controller_size, 1, **kwargs)
            self.to_proba = ProbabilityConverter()

        self.reset_parameters()

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["mode"])

    def forward(self, content_attn, loc_attn,
                conf_content=None,
                conf_loc=None,
                controller=None):
        """Compute and return the final attention and percentage of positional attention.

        Args:
            content_attn (torch.tensor): tensor of size (batch_size, n_queries,
                 n_keys) containing the content attentions.
            loc_attn (torch.tensor): tensor of size (batch_size, n_queries,
                 n_keys) containing the location attentions.
            conf_content (torch.tensor, optional): tensor of size (batch_size,
                n_queries) containing the confidence for the content attentions.
            conf_loc (torch.tensor, optional): tensor of size (batch_size,
                n_queries) containing the confidence for the location attentions.
            controller (torch.tensor, optional): controller tensor of size
                (batch_size, n_queries, kq_size) used for the generation of some
                variables. Can be `None` if unused.
        """
        if not self.training or self.n_training_calls >= self.n_steps_wait:
            if self.mode == "loc_conf":
                perc_loc = conf_loc.unsqueeze(-1)
            elif self.mode == "confidence":
                perc_loc = (conf_loc / (conf_loc + conf_content)).unsqueeze(-1)
            elif self.mode == "generate":
                perc_loc = self.generator(controller)
                perc_loc = self.to_proba(perc_loc)
            else:
                raise ValueError("Unkown mode={}".format(self.mode))
        else:
            batch_size = content_attn.size(0)
            perc_loc = self.dflt_perc_loc.expand(batch_size, 1)

        if self.rounder_perc is not None:
            perc_loc = self.rounder_perc(perc_loc)

        # Convex combination
        attn = (loc_attn * perc_loc) + ((1 - perc_loc) * content_attn)

        self.add_to_test(perc_loc, "position_percentage")
        self.add_to_visualize(perc_loc, "position_percentage")

        return attn
