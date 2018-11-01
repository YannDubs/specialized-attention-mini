
from __future__ import print_function
import math
import warnings
import torch.nn as nn
import torch
import numpy as np

from seq2seq.util.helpers import HyperparameterInterpolator, Rate2Steps, add_to_visualize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_loss(loss_name, token_loss_weight, tgt, **kwargs):
    loss_weight = 1.0
    if isinstance(loss_name, tuple):
        loss_name, loss_weight = loss_name

    output_pad = tgt.vocab.stoi[tgt.pad_token]
    loss_name = loss_name.lower()
    if loss_name == "nll":
        loss = NLLLoss(ignore_index=output_pad, weight=token_loss_weight, **kwargs)
    elif loss_name == "perplexity":
        loss = Perplexity(ignore_index=output_pad, weight=token_loss_weight, **kwargs)
    elif loss_name == "attention loss":
        loss = AttentionLoss(ignore_index=output_pad, weight=token_loss_weight, **kwargs)
    else:
        raise ValueError("Unkown loss : {}".format(loss_name))

    return loss, loss_weight


def get_losses(loss_names, tgt, is_predict_eos, eos_weight=None, **kwargs):
    """Gets a list of losses.

    loss_names (list, optional): list where each element is either the name of
        the loss or a tuple of (loss_name, loss_weight). Current available losses :
         {"nll", "perplexity", "attention loss"}
    tgt (TargetField): target field.
    is_predict_eos (bool, optional): whether the mdoel has to predict the <eos>
        token.
    is_predict_eos (bool, optional): whether the mdoel has to predict the <eos>
        token.
    eos_weight (int, optional): weight of the loss that should be given to the
        <eos> token.
    """
    if is_predict_eos:
        token_loss_weight = torch.ones(len(tgt.vocab.stoi))
        if eos_weight is not None:
            token_loss_weight[tgt.eos_id] = eos_weight
        token_loss_weight.to(device)
    else:
        token_loss_weight = None

    losses = []
    loss_weights = []
    for loss_name in loss_names:
        loss, loss_weight = _get_loss(loss_name, token_loss_weight, tgt, **kwargs)
        loss.to(device)
        losses.append(loss)
        loss_weights.append(loss_weight)

    return losses, loss_weights


class LossWeightUpdater:
    """Helper class to update the weight of the tokens of a loss.

    Args:
        indices (list of str): ordered list of the vocabulary indices of the token
            to modify.
        final_weights (list of float): ordered list of name of the final weights.
        inital_weights (list of float): ordered list of name of the initial weights.
        n_steps_interpolates (list of int): ordered list of the numebr of steps to
            wait for before reaching the final weights.
        modes (list of str or str): list of modes or single mode for all.
    """

    def __init__(self, indices, initial_weights, final_weights, n_steps_interpolates,
                 modes="geometric"):
        self.updaters = dict()

        if isinstance(modes, str):
            modes = [modes] * len(indices)

        zips = zip(indices, final_weights, initial_weights, n_steps_interpolates, modes)
        for index, final_weight, initial_weight, n_steps_interpolate, mode in zips:
            self.updaters[index] = HyperparameterInterpolator(initial_weight,
                                                              final_weight,
                                                              n_steps_interpolate,
                                                              mode)

    def reset_parameters(self):
        """Reset the updater."""
        for _, updater in self.updaters.items():
            self.updater.reset_parameters()

    def __call__(self, is_training):
        """Returns the `new_weights` in a format understandable by `Loss.update_weights`"""
        return {index: updater(is_training) for index, updater in self.updaters.items()}


class Loss(object):
    """ Base class for encapsulation of the loss functions.
    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss functions.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.
    Attributes:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss functions.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.  Implementation depends on individual
            sub-classes.
        acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the loss of multiple batches.  Implementation depends on individual
            sub-classes.
    """

    def __init__(self, name, log_name, inputs, target, criterion,
                 total_training_calls=None, max_p_interpolators=None):
        self.name = name
        self.log_name = log_name
        self.inputs = inputs
        self.target = target
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        self.regularization_loses = dict()
        # normalization term
        self.norm_term = 0

        self.rate2steps = (Rate2Steps(total_training_calls)
                           if total_training_calls is not None else None)
        self.max_p_interpolators = (max_p_interpolators
                                    if max_p_interpolators is not None else dict())
        self.counters = {k: 0 for k, _ in self.max_p_interpolators.items()}

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0
        self.regularization_loses = dict()

    def get_loss(self):
        """ Get the loss.
        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.
        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, decoder_outputs, other, target_variable):
        """ Evaluate and accumulate loss given outputs and expected results.
        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.
        Args:
            decoder_outputs (torch.Tensor): outputs of a batch.
            other (dictionary): extra outputs of the model
            target_variable (torch.Tensor): expected output of a batch.
        """

        # lists with:
        # decoder outputs # (batch, vocab_size?)
        # attention scores # (batch, 1, input_length)

        if self.inputs == 'decoder_output':
            outputs = decoder_outputs
        else:
            outputs = other[self.inputs]

        targets = target_variable[self.target]

        for step, step_output in enumerate(outputs):
            step_target = targets[:, step + 1]
            self.eval_step(step_output, step_target)

    def eval_step(self, outputs, target):
        """ Function called by eval batch to evaluate a timestep of the batch.
        When called it updates self.acc_loss with the loss of the current step.
        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()

    def to(self, device):
        self.criterion.to(device)

    def backward(self, retain_graph=False):
        """ Backpropagate the computed loss.
        """
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        total_loss = self._apply_regularization_losses()
        total_loss.backward(retain_graph=retain_graph)

    def scale_loss(self, factor):
        """ Scale loss with a factor
        """
        self.acc_loss = self.acc_loss * factor
        if self.acc_loss.item() < 0:
            raise ValueError("The loss appears to be negative loss={}. factor={}".format(self.acc_loss.item()), factor)

    def store_regularization_loss(self, name_other, other_loss,
                                  weight=None,
                                  is_update=True,
                                  initial_max_proportion=0,
                                  final_max_proportion=1e-2,
                                  anneal_max_proportion=0,
                                  rate_start_step=0.1,
                                  add_every_i=1,
                                  to_visualize=None,
                                  training_step=None,
                                  **kwargs):
        """ Use a regularization loss. """
        if name_other not in self.max_p_interpolators:
            warnings.warn("using default interpolator for {}".format(name_other))
            n_steps_interpolate = self.rate2steps(anneal_max_proportion)
            start_step = self.rate2steps(rate_start_step)
            self.max_p_interpolators[name_other
                                     ] = HyperparameterInterpolator(initial_max_proportion,
                                                                    final_max_proportion,
                                                                    n_steps_interpolate,
                                                                    start_step=start_step,
                                                                    **kwargs)
            self.counters[name_other] = 0
        elif self.max_p_interpolators[name_other] is None:
            max_proportion = None
            weight = 1
        else:
            max_proportion = self.max_p_interpolators[name_other](is_update)

        self.counters[name_other] += 1

        if self.counters[name_other] % add_every_i != 0:
            return

        if weight is None:
            max_loss = max_proportion * self.acc_loss.detach()
            other_loss_detached = other_loss.detach()
            # only scale ifother_loss_detached >  max_loss
            weight = (max_loss / other_loss_detached).clamp(max=1.)
            weight[~torch.isfinite(weight)] = 1.

        weighted_loss = weight * other_loss
        self.regularization_loses[name_other] = weighted_loss

        # # # # # DEV MODE # # # # #
        if to_visualize is not None:
            if self.acc_loss.item() < 0:
                raise ValueError("The loss appears to be negative loss={}.".format(self.acc_loss.item()))
            elif self.acc_loss.item() == 0:
                warnings.warn("Skipping losses_weighted_{} as acc_loss == 0".format(name_other))
            else:
                add_to_visualize(weighted_loss.mean().item() / self.acc_loss.item(),
                                 "losses_weighted_{}".format(name_other),
                                 to_visualize, is_training=True, training_step=training_step)

            add_to_visualize(other_loss.mean().item(),
                             "losses_{}".format(name_other),
                             to_visualize, is_training=True, training_step=training_step)
        # # # # # # # # # # # # # # #

    def _apply_regularization_losses(self):
        """Use the stored regularization losses."""

        total_loss = self.acc_loss + sum(self.regularization_loses.values())

        return total_loss.mean()

    def update_weights(self, new_weights):
        """Update the weight of the loss of certain tokens.

        Args:
            new_weights (torch.tensor or dictionary): if a torch tensor, will
                replace the whole wights. If a dictionary the key must be the index
                to modify and the values the new weights.
        """
        if self.criterion.weight is None:
            raise ValueError("to call `update_weights` you must have initialized the weights when instantiating the loss using the parameter `weight`.")

        if isinstance(new_weights, dict):
            for index, v in new_weights.items():
                self.criterion.weight[index] = v
        else:
            if self.criterion.weight.shape != new_weights.shape:
                raise ValueError("Previous weight shapes was : {} and cannot be raplced by a shape of {}.".format(self.criterion.weight.shape, new_weights.shape))
            self.criterion.weight = new_weights.to(device)


class NLLLoss(Loss):
    """ Batch averaged negative log-likelihood loss.
    Args:
        ignore_index (int, optional): index of masked token
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """

    _NAME = "Avg NLLLoss"
    _SHORTNAME = "nll_loss"
    _INPUTS = "decoder_output"
    _TARGETS = "decoder_output"

    def __init__(self,
                 ignore_index=-1,
                 size_average=True,
                 total_training_calls=None,
                 max_p_interpolators=None,
                 **kwargs):
        self.ignore_index = ignore_index
        self.size_average = size_average

        super(NLLLoss, self).__init__(self._NAME,
                                      self._SHORTNAME,
                                      self._INPUTS,
                                      self._TARGETS,
                                      nn.NLLLoss(ignore_index=ignore_index,
                                                 size_average=size_average,
                                                 **kwargs),
                                      total_training_calls=total_training_calls,
                                      max_p_interpolators=max_p_interpolators)

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.item()
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_step(self, step_outputs, target):
        batch_size = target.size(0)
        outputs = step_outputs.contiguous().view(batch_size, -1)
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1
        if self.acc_loss.item() < 0:
            raise ValueError("The loss appears to be negative loss={}. Added:{}".format(self.acc_loss.item()), self.criterion(outputs, target))


class Perplexity(NLLLoss):
    """ Language model perplexity loss.
    Perplexity is the token averaged likelihood.  When the averaging options are
    the same, it is the exponential of negative log-likelihood.
    Args:
        ignore_index (int, optional): index to be masked, refer to
        http://pytorch.org/docs/master/nn.html#nllloss
    """

    _NAME = "Perplexity"
    _SHORTNAME = "ppl"
    _MAX_EXP = 100
    _INPUTS = "decoder_output"

    def __init__(self, ignore_index=-100, **kwargs):
        super(Perplexity, self).__init__(ignore_index=ignore_index,
                                         size_average=False,
                                         **kwargs)

    def eval_step(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        if self.ignore_index is -100:
            self.norm_term += np.prod(target.size())
        else:
            self.norm_term += target.data.ne(self.ignore_index).sum()
        if self.acc_loss.item() < 0:
            raise ValueError("The loss appears to be negative loss={}. Added:{}".format(self.acc_loss.item()), self.criterion(outputs, target))

    def get_loss(self):
        nll = super(Perplexity, self).get_loss()
        nll /= self.norm_term.item()
        if nll > Perplexity._MAX_EXP:
            warnings.warn("Loss exceeded maximum value, capping to e^100")
            return math.exp(Perplexity._MAX_EXP)
        return math.exp(nll)


class AttentionLoss(NLLLoss):
    """ Cross entropy loss over attentions
    Args:
        ignore_index (int, optional): index of token to be masked
    """
    _NAME = "Attention Loss"
    _SHORTNAME = "attn_loss"
    _INPUTS = "attention_score"
    _TARGETS = "attention_target"

    def __init__(self, ignore_index=-1, **kwargs):
        super(AttentionLoss, self).__init__(ignore_index=ignore_index,
                                            size_average=True,
                                            **kwargs)

    def eval_step(self, step_outputs, step_target):
        batch_size = step_target.size(0)
        outputs = torch.log(step_outputs.contiguous().view(batch_size, -1).clamp(min=1e-20))
        self.acc_loss += self.criterion(outputs, step_target)
        self.norm_term += 1
        if self.acc_loss.item() < 0:
            raise ValueError("The loss appears to be negative loss={}. Added:{}".format(self.acc_loss.item()), self.criterion(outputs, target))
