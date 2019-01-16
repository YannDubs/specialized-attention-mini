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
from torch.nn.modules.distance import PairwiseDistance, CosineSimilarity

from seq2seq.util.initialization import weights_init, linear_init
from seq2seq.util.torchextend import MLP, ProbabilityConverter
from seq2seq.util.helpers import Clamper, get_extra_repr
from seq2seq.util.base import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContentAttender(Module):
    """
    Applies a content attention between the keys and queries.

    Args:
        kq_size (int): key and query size.
        scorer({'multiplicative', "additive", "euclidean", "scaledot", "cosine"}, optional):
            The method to compute the alignment. `"scaledot" [Vaswani et al., 2017]
            mitigates the high dimensional issue by rescaling the dot product.
            `"additive"` is the original  attention [Bahdanau et al., 2015].
            `"multiplicative"` is faster and more space efficient [Luong et al., 2015]
            but performs a little bit worst for high dimensions. `"cosine"` cosine
            distance. `"euclidean"` Euclidean distance.
    """

    def __init__(self, kq_size, scorer="multiplicative"):
        super().__init__()

        self.scorer = get_scorer(scorer, kq_size)

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

    def forward(self, keys, queries):
        """Compute the content attention.

        Args:
            keys (torch.tensor): tensor of size (batch_size, n_keys, kq_size)
                containing the keys.
            queries (torch.tensor): tensor of size (batch_size, n_queries, kq_size)
                containing the queries.

        Return:
            attn (torch.tensor): tensor of size (batch_size, n_queries, n_keys)
                containing the content attention.
            confidence (torch.tensor): tensor of size (batch_size, n_queries)
                containing the content confidence.
        """
        logits = self.scorer(keys, queries)

        max_logit = logits.logsumexp(dim=-1)

        confidence = self.maxlogit_to_conf(max_logit)

        attn = logits.softmax(dim=-1)

        self.add_to_test([logits, max_logit],
                         ["logits", "max_logit"])

        # TEST
        batch_size, n_queries, kq_size = queries.size()
        n_keys = keys.size(1)
        assert attn.size() == (batch_size, n_queries, n_keys)
        assert confidence.size() == (batch_size, n_queries)
        assert logits.size() == (batch_size, n_queries, n_keys)

        return attn, confidence


def get_scorer(scorer, kq_size):
    """
    Set scorer that matches key and query to compute attention along `dim=1`.
    """
    if scorer == 'multiplicative':
        scorer = MultiplicativeScorer(kq_size)
    elif scorer == 'additive':
        scorer = AdditiveScorer(kq_size)
    elif scorer == 'scaledot':
        scorer = DotScorer(is_scale=True)
    elif scorer == "cosine":
        scorer = MetricScorer(metric="cosine")
    elif scorer == "euclidean":
        scorer = MetricScorer(metric="l2")
    elif scorer == "manhattan":
        scorer = MetricScorer(metric="l1")
    elif scorer == "kq":
        scorer = KQScorer(kq_size)
    else:
        raise ValueError("Unknown attention method {}".format(scorer))

    return scorer


class MultiplicativeScorer(Module):
    """
    Multiplicative scorer for attention [Luong et al., 2015].

    Note:
        - Only difference is that adds bias.

    Shape:
        keys: `(batch_size, n_keys, kq_size)`
        queries: `(batch_size, n_queries, kq_size)`
        logits: `(batch_size, n_queries, n_keys)`
    """

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.dot = DotScorer(is_scale=False)

        self.reset_parameters()

    def forward(self, keys, queries):
        transformed_queries = self.linear(queries)
        logits = self.dot(keys, transformed_queries)
        return logits

    def extra_repr(self):
        pass


class KQScorer(Module):
    """
    Key Query scorer.

    Shape:
        keys: `(batch_size, n_keys, kq_size)`
        queries: `(batch_size, n_queries, kq_size)`
        logits: `(batch_size, n_queries, n_keys)`
    """

    def __init__(self, dim, kq_size=32):
        super().__init__()
        self.key_mlp = MLP(dim, kq_size, hidden_size=kq_size)
        self.query_mlp = MLP(dim, kq_size, hidden_size=kq_size)
        self.dot = DotScorer(is_scale=False)

        self.reset_parameters()

    def forward(self, keys, queries):
        batch_size, n_queries, kq_size = queries.size()

        keys = self.key_mlp(keys)
        queries = self.query_mlp(queries)

        logits = self.dot(keys, queries)
        return logits

    def extra_repr(self):
        pass


class AdditiveScorer(Module):
    """
    Additive scorer for the original attention [Bahdanau et al., 2015].

    Note:
        - Only difference is that adds bias.

    Shape:
        keys: `(batch_size, n_keys, kq_size)`
        queries: `(batch_size, n_queries, kq_size)`
        logits: `(batch_size, n_queries, n_keys)`
    """

    def __init__(self, dim):
        super().__init__()
        self.mlp = MLP(dim * 2, 1, hidden_size=32, activation=nn.Tanh)

        self.reset_parameters()

    def forward(self, keys, queries):
        batch_size, n_queries, kq_size = queries.size()
        n_keys = keys.size(1)

        keys = keys.unsqueeze(1).expand(batch_size, n_queries, n_keys, kq_size)
        queries = queries.unsqueeze(1).expand(batch_size, n_queries, n_keys, kq_size)

        logits = self.mlp(torch.cat((keys, queries), dim=-1)).squeeze(-1)
        return logits

    def extra_repr(self):
        pass


class DotScorer(Module):
    """
    Dot product attention.

    is_scale (bool, optional): whether to use a scaled attention just like
        in "attention is all you need". Scaling can help when dimension is large :
        making sure that there are no  extremely small gradients.

    Shape:
        keys: `(batch_size, n_keys, kq_size)`
        queries: `(batch_size, n_queries, kq_size)`
        logits: `(batch_size, n_queries, n_keys)`
    """

    def __init__(self, is_scale=True):
        super().__init__()
        self.is_scale = is_scale

    def forward(self, keys, queries):
        logits = torch.bmm(queries, keys.transpose(1, 2))

        if self.is_scale:
            kq_size = queries.size(-1)
            logits = logits / math.sqrt(kq_size)

        return logits

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["is_scale"])


class MetricScorer(Module):
    """
    Metric scorer for attention.

    Args:
        metric ({"l1","l2","cosine"}): metric to use to compute the similarity.
            If `"l1"` or `"l2"` converts the distance to similarity by
            `sim=1/(1+dist)`.

    Shape:
        keys: `(batch_size, n_keys, kq_size)`
        queries: `(batch_size, n_queries, kq_size)`
        logits: `(batch_size, n_queries, n_keys)`
    """

    def __init__(self, metric):
        super().__init__()

        self.metric = metric
        self.is_distance = self.metric in ["l1", "l2"]

        if self.metric == "l1":
            self.distance = PairwiseDistance(p=1, dim=1)
        elif self.metric == "l2":
            self.distance = PairwiseDistance(p=2, dim=1)
        elif self.metric == "cosine":
            self.similarity = CosineSimilarity(dim=1)

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["metric"])

    def forward(self, keys, queries):
        batch_size, n_queries, kq_size = queries.size()
        n_keys = keys.size(1)

        #keys = keys.view(batch_size, kq_size, n_queries, 1)
        keys = keys.expand(batch_size, kq_size, n_queries, n_keys)
        queries = queries.view(batch_size, kq_size, n_queries, 1)

        if self.is_distance:
            dist = self.distance(keys, queries)
            logits = 1 / (1 + dist)
        else:
            logits = self.similarity(keys, queries)

        return logits
