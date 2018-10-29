"""
Content attention modules.

NOTA BENE:
    - Besides `AdditiveAttn` and the `additional` dictionary this shouldn't need
    much refactoring

Contact : Yann Dubois
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq.util.initialization import weights_init, linear_init
from seq2seq.util.torchextend import MLP, ProbabilityConverter
from seq2seq.util.helpers import Clamper, get_extra_repr
from seq2seq.util.base import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContentAttention(Module):
    """
    Applies a content attention between the keys and queries.

    Args:
        dim(int): The number of expected features in the output
        method({'multiplicative', "scaledot", "dot", "scalemult"}, optional):
            The method to compute the alignment. `"dot"` corresponds to a simple
            product. `"additive"` is the original  attention [Bahdanau et al., 2015].
            `"Multiplicative"` is faster and more space efficient [Luong et al., 2015]
            but performs a little bit worst for high dimensions. `"scaledot"
            [Vaswani et al., 2017] mitigates the highdimensional issue by rescaling
            the dot product. `"scalemult"` uses the same rescaling trick but with
            a multiplicative attention.
    """

    def __init__(self, dim, method="scalemult"):

        super(ContentAttention, self).__init__()

        self.mask = None
        self.method = self.get_method(method, dim)

        # low initial temperature because logits can take a very high range of values
        # so don't want to have vanishing gradients from the start
        self.maxlogit_to_conf = ProbabilityConverter(is_temperature=True,
                                                     is_bias=True,
                                                     initial_temperature=0.1,
                                                     temperature_transformer=Clamper(minimum=0.05,
                                                                                     maximum=10,
                                                                                     is_leaky=True,
                                                                                     hard_min=0.01
                                                                                     ))

        self.reset_parameters()

    def extra_repr(self):
        pass

    def set_mask(self, mask):
        """
        Sets indices to be masked.

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, queries, keys, step):
        """Compute the content attention.

        Args:
            queries (torch.tensor): tensor of size (batch_size, n_queries, kq_size)
                containing the queries.
            keys (torch.tensor): tensor of size (batch_size, n_keys, kq_size)
                containing the keys.
        """

        batch_size, n_queries, kq_size = queries.size()
        n_keys = keys.size(1)

        logits = self.method(queries, keys)

        approx_max_logit = logits.logsumexp(dim=-1)

        confidence = self.maxlogit_to_conf(approx_max_logit)

        attn = F.softmax(logits.view(-1, n_keys), dim=1).view(batch_size, -1, n_keys)

        self.add_to_test([logits, approx_max_logit],
                         ["logits", "approx_max_logit"])

        return attn, confidence

    def get_method(self, method, dim):
        """
        Set method to compute attention
        """
        if method == 'multiplicative':
            method = MultiplicativeAttn(dim, is_scale=False)
        elif method == 'scalemult':
            method = MultiplicativeAttn(dim, is_scale=True)
        elif method == 'scaledot':
            method = DotAttn(is_scale=True)
        elif method == 'dot':
            method = DotAttn(is_scale=False)
        else:
            raise ValueError("Unknown attention method {}".format(method))

        return method


class DotAttn(Module):
    """
    Dot product attention.

    is_scale (bool, optional): whether to use a scaled attention just like
        in "attention is all you need". Scaling can help when dimension is large :
        making sure that there are no  extremely small gradients.
    """

    def __init__(self, is_scale=True):
        super().__init__()
        self.is_scale = is_scale

    def forward(self, queries, keys):
        logits = torch.bmm(queries, keys.transpose(1, 2))

        if self.is_scale:
            logits = logits / math.sqrt(queries.size(-1))

        return logits

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["is_scale"])


class MultiplicativeAttn(Module):
    """
    Multiplicative product attention [Luong et al., 2015].

    is_scale (bool, optional): whether to use the same idea as the scaled attention
        in "attention is all you need". Scaling can help when dimension is large :
        making sure that there are no  extremely small gradients.
    """

    def __init__(self, dim, is_scale=True):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.scaled_dot = DotAttn(is_scale=is_scale)

        self.reset_parameters()

    def forward(self, queries, keys):
        transformed_queries = self.linear(queries)
        logits = self.scaled_dot(transformed_queries, keys)
        return logits

    def extra_repr(self):
        pass
