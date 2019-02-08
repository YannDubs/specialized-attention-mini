"""
Other possible attenders for comparaison purpose.
"""
import ipdb  # Dev mode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

from seq2seq.util.helpers import (renormalize_input_length, get_rnn, get_extra_repr,
                                  clamp, format_source_lengths,
                                  HyperparameterInterpolator)
from seq2seq.util.torchextend import get_rounder, L0Gates, MLP
from seq2seq.util.initialization import replicate_hidden0
from seq2seq.util.base import Module
from seq2seq.attention.location import SigmaGenerator, get_loc_pdf, MuGenerator
from seq2seq.attention.content import get_scorer, scorer_filter_args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocationOnlyAttender(Module):
    """Location Attender.

    Args:
        controller_size (int): size of the controler.
        max_len (int): a maximum allowed length for the sequence to be processed
        n_steps_prepare_pos (int): number of steps during which to consider
            the positioning as in a preparation mode. During preparation mode,
            the model have less parameters to tweak, it will focus on what I thought
            were the most crucial bits. For example it will have a fix
            sigma and won't have many of the regularization term, this is to
            help it start at a decent place in a lower dimensional space, before
            going to the hard task of tweaking all at the same time.
        pdf ({"gaussian", "laplace"}, optional): name of the pdf to use to generate
            the location attention.
        Generator (Module, optional): module to generate various values. It
            should be callable using `Generator(input_size, output_size)(x)`.
            By default `nn.Linear`.
        hidden_size (int, optional): number of neurones to use in the hidden layer
            of the weighter.
        gating ({None, "residual", "highway", "gates_res"}, optional): Gating
            mechanism for generated values. `None` no gating. `"residual"` adds
            the new value to the previous. `"highway"` gating using convex
            combination. `"gates_res"` gates the previous value and add the new one.
        sigma_kwargs (dictionary, optional): additional arguments to the
            `SigmaGenerator`.
        mu_kwargs (dictionary, optional): additional arguments to the
            `MuGenerator`.
    """

    def __init__(self, controller_size, max_len,
                 n_steps_prepare_pos=100,
                 pdf="gaussian",
                 Generator=nn.Linear,
                 hidden_size=64,
                 gating="gated_res",
                 pretrained_locator=None,
                 sigma_kwargs={},
                 mu_kwargs={},
                 is_recurrent=True,  # DEV MODE
                 ):
        super().__init__()

        self.controller_size = controller_size
        self.max_len = max_len
        self.n_steps_prepare_pos = n_steps_prepare_pos
        self.pdf = pdf
        self.gating = gating
        self.pretrained_locator = pretrained_locator
        self.is_recurrent = is_recurrent

        self.rel_counter = torch.arange(0, self.max_len,
                                        dtype=torch.float,
                                        device=device).unsqueeze(1) / (self.max_len - 1)

        self.resizer = nn.Linear(self.controller_size, hidden_size)

        if self.is_recurrent:
            self.weighter, self.hidden0 = get_rnn("gru", hidden_size, hidden_size,
                                                  batch_first=True,
                                                  is_get_hidden0=True)
        else:
            self.weighter = MLP(hidden_size, hidden_size)

        self.mu_generator = MuGenerator(hidden_size,
                                        max_len=max_len,
                                        Generator=Generator,
                                        gating=self.gating,
                                        n_steps_prepare_pos=n_steps_prepare_pos,
                                        **mu_kwargs)

        self.sigma_generator = SigmaGenerator(hidden_size,
                                              n_steps_const_sigma=n_steps_prepare_pos,
                                              Generator=Generator,
                                              gating=self.gating,
                                              **sigma_kwargs)

        self.pdf = get_loc_pdf(pdf)

        self.reset_parameters()

        if self.pretrained_locator is not None:
            self.load_locator(self.pretrained_locator)

    def reset_parameters(self):
        """Reset and initialize the module parameters."""
        if self.pretrained_locator is None:
            # only reset param if not pretrained
            super().reset_parameters()

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["pdf"],
                              conditional_shows=["gating"])

    def load_locator(self, file):
        """
        Loads a pretrained locator (output from self.save_locator) for transfer
        learning.
        """
        locator = torch.load(file)
        self.weighter.load_state_dict(locator["weighter"])
        self.mu_generator.load_state_dict(locator["mu_generator"])
        self.sigma_generator.load_state_dict(locator["sigma_generator"])
        self.hidden0 = locator["hidden0"]
        self.pdf = locator["pdf"]

    def save_locator(self, file):
        """Save the pretrained locator to a file."""
        locator = dict(weighter=self.weighter.state_dict(),
                       hidden0=self.hidden0,
                       mu_generator=self.mu_generator.state_dict(),
                       sigma_generator=self.sigma_generator.state_dict(),
                       pdf=self.pdf)
        torch.save(locator, file)

    def named_params_locator(self):
        """Return a generator of named parameters and parameters of the location
        attender"""
        return self.named_parameters()

    def forward(self, keys, query, source_lengths=None, step=None, **kwargs):
        """Compute and return the location attention.

        Args:
            keys (torch.tensor): key of size (batch_size, n_keys, kq_size).
            query (torch.tensor): query of size (batch_size, n_queries, kq_size).
            step (int): current decoding step.
            source_lengths (tuple(list of int, torch.FloatTesnor), optional): A
                list that contains the lengths of sequences in the mini-batch. The
                Tensor has the same information but is preinitialized on teh
                correct device.

        Return:
            loc_attn (torch.tensor): location attention. Shape: (batch_size,
                n_queries, n_keys).
        """
        assert source_lengths is not None
        assert step is not None

        query = self.resizer(query)
        query = F.relu(query)

        batch_size, n_queries, _ = query.size()
        if n_queries != 1:
            txt = "`n_queries = {}` but only single query supported for now."
            raise NotImplementedError(txt.format(n_queries))
        source_lengths_list, source_lengths_tensor = format_source_lengths(source_lengths)

        mu, sigma, mu_weights = self._compute_parameters(query, step, source_lengths_tensor)

        to_store = [x.squeeze(1) for x in [mu, sigma]]
        labels_to_store = ["mu", "sigma"]
        self.add_to_visualize(to_store, labels_to_store)
        self.add_to_test(to_store, labels_to_store)

        sigma = renormalize_input_length(sigma, source_lengths_tensor, 1)

        loc_attn = self._compute_attn(mu, sigma, source_lengths)

        self.add_to_test([loc_attn], ["loc_attention"])
        self.storer["old_attn"] = loc_attn

        return loc_attn, mu_weights

    def _compute_parameters(self, weighter_inputs, step, source_lengths_tensor):
        """Compute the parameters of the positioning function.

        Return:
            mu (torch.FloatTensor): mean location of size. Shape:
                (batch_size, n_queries, 1).
            sigma (torch.FloatTensor): standard deviation of location. Shape:
                (batch_size, n_queries, 1)
        """
        if self.is_recurrent:
            if step == 0:
                batch_size = weighter_inputs.size(0)
                self.storer["weighter_hidden"] = replicate_hidden0(self.hidden0, batch_size)

            (weighter_out,
             self.storer["weighter_hidden"]) = self.weighter(weighter_inputs,
                                                             self.storer["weighter_hidden"])
        else:
            weighter_out = self.weighter(weighter_inputs)

        # for this the mean attn actually corresponds to mu which would be easier
        # to give, but as the focus in on the both attentions lets use the mean attn
        old_attn = self.storer["old_attn"] if step != 0 else None
        mu, mu_weights = self.mu_generator(weighter_out, step, source_lengths_tensor, old_attn)

        sigma = self.sigma_generator(weighter_out, mu, step)

        return mu, sigma, mu_weights

    def _compute_attn(self, mu, sigma, source_lengths):
        """
        Compute the attention given the sufficient parameeters of the probability
        density function.

        Shape:
            pos_confidence: `(batch_size, n_queries, n_keys)`
        """
        batch_size, _, _ = mu.size()
        source_lengths_list, source_lengths_tensor = format_source_lengths(source_lengths)

        rel_counter = self.rel_counter.expand(batch_size, self.max_len, 1)
        rel_counter = renormalize_input_length(rel_counter,
                                               source_lengths_tensor - 1,
                                               self.max_len - 1)

        # slow because list comprehension : should optimize
        loc_attn = pad_sequence([self.pdf(rel_counter[i_batch, :length, :],
                                          mu[i_batch, ...].squeeze(),
                                          sigma[i_batch, ...].squeeze())
                                 for i_batch, length in enumerate(source_lengths_list)],
                                batch_first=True)

        loc_attn = loc_attn.transpose(2, 1)

        return loc_attn


class ContentOnlyAttender(Module):
    """Content Attender.

    Args:
        controller_size (int): size of the controler.
        scorer({'multiplicative', "additive", "euclidean", "scaledot", "cosine"}, optional):
            The method to compute the alignment. `"scaledot" [Vaswani et al., 2017]
            mitigates the high dimensional issue by rescaling the dot product.
            `"additive"` is the original  attention [Bahdanau et al., 2015].
            `"multiplicative"` is faster and more space efficient [Luong et al., 2015]
            but performs a little bit worst for high dimensions. `"cosine"` cosine
            distance. `"euclidean"` Euclidean distance.
    """

    def __init__(self, controller_size, max_len=None, scorer="multiplicative"):
        super().__init__()

        self.scorer = get_scorer(scorer, controller_size, max_len=max_len)

        self.reset_parameters()

    def extra_repr(self):
        pass

    def forward(self, keys, queries, **kwargs):
        """Compute the content attention.

        Args:
            keys (torch.tensor): tensor of size (batch_size, n_keys, kq_size)
                containing the keys.
            queries (torch.tensor): tensor of size (batch_size, n_queries, kq_size)
                containing the queries.

        Return:
            attn (torch.tensor): tensor of size (batch_size, n_queries, n_keys)
                containing the content attention.
        """
        kwargs = scorer_filter_args(self.scorer, **kwargs)

        logits = self.scorer(keys, queries, **kwargs)

        attn = logits.softmax(dim=-1)

        self.add_to_test([attn], ["content_attention"])

        return attn


class HardAttender(Module):
    """Hard attention for data sets that are annotated with attentive guidance.

    Shape:
        keys: `(batch_size, n_keys, kq_size)`
        queries: `(batch_size, n_queries, kq_size)`
        provided_attention: `(batch_size, target_len).
        logits: `(batch_size, n_queries, n_keys)`
    """

    def extra_repr(self):
        pass

    def forward(self, keys, queries, step=None, provided_attention=None, **kwargs):
        """Compute the content attention.

        Args:
            keys (torch.tensor): tensor of size (batch_size, n_keys, kq_size)
                containing the keys.
            queries (torch.tensor): tensor of size (batch_size, n_queries, kq_size)
                containing the queries.
            step (int): current decoding step.
            provided_attention (LongTensor, optional): attention gauidance if using
                hard attention of shape (batch_size, target_len).

        Return:
            attn (torch.tensor): tensor of size (batch_size, n_queries, n_keys)
                containing the hard attention.
        """
        assert provided_attention is not None
        assert step is not None

        batch_size, n_queries, kq_size = queries.size()
        n_keys = keys.size(1)

        # If we have shorter examples in a batch, attend the PAD outputs to the
        # first encoder state
        provided_attention.masked_fill_(provided_attention.eq(-1), 0)
        current_attn = provided_attention[:, step:step + n_queries].unsqueeze(-1)

        attn = torch.full([batch_size, n_queries, n_keys],
                          fill_value=0.,
                          device=device)
        attn = attn.scatter_(dim=2, index=current_attn, value=1)

        return attn
