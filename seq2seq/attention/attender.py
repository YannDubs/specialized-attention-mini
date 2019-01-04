"""
General attender.

TO DO:

Contact: Yann Dubois
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from seq2seq.util.helpers import (batch_reduction_f, get_extra_repr)
from seq2seq.util.torchextend import (MLP, ConcreteRounder, get_rounder,
                                      StochasticRounder, get_gate)
from seq2seq.util.initialization import weights_init
from seq2seq.util.base import Module
from seq2seq.attention.content import ContentAttender
from seq2seq.attention.location import LocationAttender


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attender(Module):
    """Computes the final attention using content and locaton.

    Args:
        kq_size (int): size of the key and query.
        controller_size (int): size of the controller tensor.
        content_kwargs (dict, optional): additional arguments to `ContentAttender`.
        location_kwargs (dict, optional): additional arguments to `LocationAttender`.
        mixer_kwargs (dict, optional): additional arguments to `AttentionMixer`.
    """

    def __init__(self, kq_size, controller_size, max_len,
                 loc_query_size=64,
                 content_kwargs={},
                 location_kwargs={},
                 mixer_kwargs={}):

        super(AttentionMixer, self).__init__()

        self.resizer = nn.Linear(kq_size, loc_query_size)
        self.content_attender = ContentAttender(kq_size, **content_kwargs)
        self.location_attender = LocationAttender(loc_query_size, max_len, **location_kwargs)
        self.attn_mixer = AttentionMixer(controller_size=controller_size,
                                         **mixer_kwargs)

        self.reset_parameters()

    def extra_repr(self):
        pass

    def forward(self, key, query, controller=None):
        """Compute and return the final attention.

        Args:
            key (torch.tensor): key of size (batch_size, n_queries, kq_size).
            query (torch.tensor): query of size (batch_size, n_queries, kq_size).
            controller (torch.tensor, optional): controller tensor of size
                (batch_size, n_queries, kq_size) used for the generation of some
                variables. Can be `None` if unused.

        Return:
            attn (torch.tensor): attention of size (batch_size, n_queries, n_keys).
        """
        content_attn, conf_content = self.content_attender(key, query)
        query = self.resizer(query)
        query = F.relu(query)
        loc_attn, conf_loc = self.location_attender(query)

        self.add_to_visualize([conf_content, conf_loc], ["content_confidence", "loc_confidence"])
        self.add_to_test([content_attn, loc_attn], ["content_attention", "loc_attention"])

        attn = self.attn_mixer(content_attn, loc_attn,
                               conf_content=conf_content,
                               conf_loc=conf_loc,
                               controller=controller)

        return attn


class AttentionMixer(Module):
    """Mixes the content and location attention.

    Args:
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
        controller_size (int, optional): size of the hidden activations of the
            decoder.
        Generator (Module, optional): module to generate various values. It
            should be callable using `Generator(input_size, output_size)(x)`.
            By default `nn.Linear`.
        n_steps_wait (float, optional): number of training steps to wait
            for before starting to generate the positional percentage. Until then
            will use `default_pos_perc`.
        default_perc_loc (float, optional): constant positional percentage to
            use while `n_steps_wait`.
        gating ({None, "residual", "highway", "custom"}, optional): Gating
            mechanism for generated values. `None` no gating. `"residual"` adds
            the new value to the previous. `"highway"` gating using convex
            combination. `"custom"` gates the previous value and add the new one.
        rounder_perc_kwargs (dictionary, optional): Additional arguments to
            the percentage rounder.
    """

    def __init__(self,
                 mode="loc_conf",
                 controller_size=None,
                 Generator=nn.Linear,
                 n_steps_wait=0,
                 dflt_perc_loc=0.5,
                 gating="custom",
                 rounder_perc_kwargs={}):

        super().__init__()

        self.mode = mode.lower()
        self.n_steps_wait = n_steps_wait
        self.dflt_perc_loc = torch.tensor(dflt_perc_loc,
                                          dtype=torch.float,
                                          device=device)
        self.gate = get_gate(gating, controller_size, 1, is_single_gate=True)
        self.rounder_perc = get_rounder(**rounder_perc_kwargs)

        self.old_attn = Parameter(torch.tensor(dflt_perc_loc), device=device)
        if self.mode == "generate":
            self.generator = Generator(controller_size, 1)

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
                perc_loc = conf_loc
            elif self.mode == "confidence":
                perc_loc = conf_loc / (conf_loc + conf_content)
            elif self.mode == "generate":
                perc_loc = self.generator(controller)
            else:
                raise ValueError("Unkown mode={}".format(self.mode))
        else:
            batch_size = content_attn.size(0)
            perc_loc = self.dflt_perc_loc.expand(batch_size, 1)

        if self.rounder_perc is not None:
            perc_loc = self.rounder_perc(perc_loc)

        perc_loc = perc_loc.unsqueeze(-1)

        # Convex combination
        attn = (loc_attn * perc_loc) + ((1 - perc_loc) * content_attn)

        attn = self.gate(attn, self.old_attn, controller)
        self.old_attn = attn

        self.add_to_test(perc_loc, "position_percentage")
        self.add_to_visualize(perc_loc, "position_percentage")

        return attn
