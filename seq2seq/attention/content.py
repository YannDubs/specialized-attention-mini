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
        scorer({'multiplicative', "additive", "euclidean", "scaledot", "cosine",
            "kq"}, optional):
            The method to compute the alignment. `"scaledot" [Vaswani et al., 2017]
            mitigates the high dimensional issue by rescaling the dot product.
            `"additive"` is the original  attention [Bahdanau et al., 2015].
            `"multiplicative"` is faster and more space efficient [Luong et al., 2015]
            but performs a little bit worst for high dimensions. `"cosine"` cosine
            distance. `"euclidean"` Euclidean distance. "kq" first uses 2 different
            mlps to convert the encoder and decoder hidden state to low dimensional
            key and queries.
    """

    def __init__(self, kq_size, scorer="multiplicative", max_len=50):
        super().__init__()

        self.scorer = get_scorer(scorer, kq_size, max_len=max_len)

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

    def forward(self, keys, queries, step):
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
        logits = self.scorer(keys, queries, step=step)

        max_logit = logits.logsumexp(dim=-1)

        confidence = self.maxlogit_to_conf(max_logit)

        attn = logits.softmax(dim=-1)

        self.add_to_test([logits, max_logit],
                         ["logits", "max_logit"])

        return attn, confidence


def get_scorer(scorer, kq_size, max_len):
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
    elif scorer == "transformer":
        scorer = TransformerScore(kq_size, max_len=max_len)
    elif scorer == "transformerxl":
        scorer = TransformerXLScore(kq_size, max_len=max_len)
    else:
        raise ValueError("Unknown attention method {}".format(scorer))

    return scorer


class MultiplicativeScorer(Module):
    """
    Multiplicative scorer for attention [Luong et al., 2015].

    Args:
        dim (int): key and query size.

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

    def forward(self, keys, queries, step=None):
        transformed_queries = self.linear(queries)
        logits = self.dot(keys, transformed_queries)
        return logits

    def extra_repr(self):
        pass


class KQScorer(Module):
    """
    Key Query scorer.

    Args:
        dim (int): key and query size.
        kq_size (int, optional): effective key and query size to use (after
            transformation).
        Generator (Module, optional): module to generate various values. It
            should be callable using `Generator(input_size, output_size)(x)`.
            By default `nn.Linear`.
        kwargs:
            Additional arguments to the Generator.

    Shape:
        keys: `(batch_size, n_keys, kq_size)`
        queries: `(batch_size, n_queries, kq_size)`
        logits: `(batch_size, n_queries, n_keys)`
    """

    def __init__(self, dim, kq_size=32, Generator=MLP, **kwargs):
        super().__init__()
        self.key_generator = Generator(dim, kq_size, **kwargs)
        self.query_generator = Generator(dim, kq_size, **kwargs)
        self.dot = DotScorer(is_scale=True)

        self.reset_parameters()

    def forward(self, keys, queries, step=None):

        keys = self.key_generator(keys)
        queries = self.query_generator(queries)

        logits = self.dot(keys, queries)
        return logits

    def extra_repr(self):
        pass


class AdditiveScorer(Module):
    """
    Additive scorer for the original attention [Bahdanau et al., 2015].

    Args:
        dim (int): key and query size.

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

    def forward(self, keys, queries, step=None):
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

    def forward(self, keys, queries, step=None):
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

    def forward(self, keys, queries, step=None):
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


def get_sin_pos_enc(dim, max_len):
    """Return the sinusoidal position encodings.

    Args:
        dim (int): number of different sinusoidals corresponding to the dimension
            of the vector to which to add the positional embeddings.
        max_len (int, optional): maximum possible length of any source sentence.
    """

    pos_enc = torch.zeros(max_len, dim, dtype=torch.float)
    counter = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    two_i_d = torch.arange(0, dim, 2, dtype=torch.float) / dim
    # in log domain for math stability
    denom = torch.exp(two_i_d * math.log(10000.0))

    # apply sin on 0th,2nd,4th...emb_dim
    pos_enc[:, 0::2] = torch.sin(counter / denom)
    # apply cos on 1st,3rd,5th...emb_dim
    pos_enc[:, 1::2] = torch.cos(counter / denom)
    return pos_enc.unsqueeze(0).to(device)


class TransformerScore(Module):
    def __init__(self, dim, max_len=50):
        """
        Content attention with sinusoidal positioning like in the transformer
        [Vaswani et al., 2017].

        Args:
            dim (int): key and query size.
            max_len (int, optional): maximum possible length of any source sentence.

        Note:
            - Only difference is that adds bias.
        """
        super().__init__()

        self.dim = dim
        self.max_len = max_len

        self.pos_enc = get_sin_pos_enc(self.dim, self.max_len)
        self.scorer = KQScorer(dim,
                               kq_size=64,
                               Generator=nn.Linear)

        self.reset_parameters()

    def extra_repr(self):
        pass

    def forward(self, keys, queries, step):
        """Compute the transformer content + position attention.

        Args:
            keys (torch.tensor): tensor of size (batch_size, n_keys, kq_size)
                containing the keys.
            queries (torch.tensor): tensor of size (batch_size, n_queries, kq_size)
                containing the queries.
            step (int): position of first query step (i.e positions :
                `step:step+n_queries`)

        Return
            logits (torch.tensor): tensor of size (batch_size, n_queries, n_keys)
                containing the logits that score the key-query matching.
        """
        n_keys = keys.size(1)
        batch_size, n_queries, kq_size = queries.size()

        keys = keys + self.pos_enc[:, :n_keys, :]
        queries = queries + self.pos_enc[:, step:step + n_queries, :]
        logits = self.scorer(keys, queries)
        return logits


class TransformerXLScore(Module):
    def __init__(self, dim, max_len=50, kq_size=64):
        """
        Content attention with improved sinusoidal positioning like in the
        TRANSFORMER-XL [Dai et al., 2019].

        Args:
            dim (int): key and query size.
            max_len (int, optional): maximum possible length of any source sentence.
        """
        super().__init__()

        self.dim = dim
        self.max_len = max_len
        self.kq_size = kq_size

        self.pos_enc = get_sin_pos_enc(self.dim, self.max_len)
        self.key_cont_generator = nn.Linear(dim, dim)
        self.key_loc_generator = nn.Linear(dim, dim)
        self.query_generator = nn.Linear(dim, dim)
        self.dot = DotScorer(is_scale=True)

        self.reset_parameters()

    def extra_repr(self):
        pass

    def forward(self, keys, queries, step):
        """Compute the transformer content + position attention.

        Args:
            keys (torch.tensor): tensor of size (batch_size, n_keys, kq_size)
                containing the keys.
            queries (torch.tensor): tensor of size ()
                containing the queries.
            step (int): position of first query step (i.e positions :
                `step:step+n_queries`)

        Return
            logits (torch.tensor): tensor of size (batch_size, n_queries, n_keys)
                containing the logits that score the key-query matching.
        """
        n_keys = keys.size(1)
        batch_size, n_queries, kq_size = queries.size()

        assert n_queries == 1  # DEV MODE

        #keys = self.pos_enc[:, :n_keys, :].expand(batch_size, n_queries, n_keys, kq_size)
        #queries = self.pos_enc[:, step:step + n_queries, :].expand(batch_size, n_queries, n_keys,kq_size)

        rel_pos_enc = (self.pos_enc[:, :n_keys, :] -
                       self.pos_enc[:, step:step + n_queries, :])

        keys_cont = self.key_cont_generator(keys)
        keys_loc = self.key_loc_generator(rel_pos_enc).expand(batch_size, n_keys, kq_size)
        queries = self.query_generator(queries)

        cont_address = self.dot(keys_cont, queries)
        cont_loc_address = self.dot(keys_loc, queries)

        # in the paper they use 4 terms but 2 correspond to the biases
        logits = cont_address + cont_loc_address

        return logits
