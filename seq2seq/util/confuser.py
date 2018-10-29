"""
Confuser related objects used to remove certain features.

Contact: Yann Dubois
"""

import numpy as np
import math

import torch

import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer as BaseOptimizer

from seq2seq.util.torchextend import MLP
from seq2seq.util.initialization import linear_init
from seq2seq.util.helpers import (clamp, batch_reduction_f,
                                  HyperparameterInterpolator, add_to_visualize,
                                  SummaryStatistics)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Confuser(object):
    """Confuser object used to remove certain features from a generator. It forces
    the generator to maximize a certain criterion and a discrimiator to minimize it.

    Args:
        discriminator_criterion (callable) loss function of the discriminator.
            If the output is not a scalar, it will be averaged to get the final loss.
        input_size (int) dimension of the input.
        target_size (int) dimension of the target.
        generator (parameters, optional): part of the model to confuse.
        generator_criterion (callable, optional) oss function of the generator.
             if `None`, `discriminator_criterion` will be used. If the output is
             not a scalar, it will be averaged to get the final loss.
        hidden_size (int, optional): number of hidden neurones to use for the
            discriminator. In `None` uses a linear layer.
        default_targets (torch.tensor): default target if not given to the forward
            function.
        final_max_scale (float, optional): Final (at the end of annealing) maximum
            percentage of the total loss that the confuser loss can reach. Note
            that this only limits the gradients for the generator not the discrimator.
        n_steps_discriminate_only (int, optional): Number of steps at the begining
            where you only train the discriminator.
        final_factor (float, optional): Final (at the end of annealing) factor
            by which to decrease max loss when comparing it to the current loss.
            If factor is 2 it means that you consider that the maximum loss will
            be achieved if your prediction is 1/factor (i.e half) way between the
            correct i and the best worst case output. Factor = 10 means it can be
            a lot closer to i. This is usefull as there will always be some noise,
            and you don't want to penalize the model for some noise. Note that the
            factor generator is not used in the code but can be called to get
            the current factor with `confuser.get_factor(<is_update>)`.
        n_steps_interpolate (int, optional): number of interpolating steps before
            reaching the `final_factor` and `final_max_scale`.
        factor_kwargs (dictionary, optional): additional dictionary to the factor
            interpolator.
        max_scale_kwargs (dictionary, optional): additional dictionary to the max_scale
            interpolator.
        kwargs:
            Additional parameters for the discriminator.
    """

    def __init__(self, discriminator_criterion, input_size, target_size, generator,
                 generator_criterion=None,
                 hidden_size=32,
                 default_targets=None,
                 final_max_scale=5e-2,
                 n_steps_discriminate_only=10,
                 final_factor=1.5,
                 n_steps_interpolate=0,
                 factor_kwargs={},
                 max_scale_kwargs={},
                 **kwargs):

        self.summarize_stats = SummaryStatistics(statistics_name="all")
        input_size = input_size + self.summarize_stats.n_statistics

        self.discriminator_criterion = discriminator_criterion
        self.generator_criterion = (generator_criterion if generator_criterion is not None
                                    else discriminator_criterion)

        self.generator = generator

        if default_targets is not None:
            self.default_targets = default_targets

        if hidden_size is not None:
            self.discriminator = MLP(input_size, hidden_size, target_size, **kwargs)
        else:
            self.discriminator = nn.Linear(input_size, target_size, **kwargs)

        self.discriminator_optim = AdamPre(self.discriminator.parameters())
        self.generator_optim = AdamPre(self.generator.parameters())

        self.n_steps_discriminate_only = n_steps_discriminate_only

        initial_factor = 10
        self.get_factor = HyperparameterInterpolator(initial_factor,
                                                     final_factor,
                                                     n_steps_interpolate,
                                                     mode="geometric",
                                                     start_step=self.n_steps_discriminate_only,
                                                     default=initial_factor,
                                                     **factor_kwargs)

        initial_max_scale = 0.5
        self.get_max_scale = HyperparameterInterpolator(initial_max_scale,
                                                        final_max_scale,
                                                        n_steps_interpolate,
                                                        mode="geometric",
                                                        start_step=self.n_steps_discriminate_only,
                                                        default=initial_max_scale,
                                                        **max_scale_kwargs)

        self.to_backprop_generator = None
        self.n_training_calls = 0

        self.reset_parameters()

    def to(self, device):
        self.discriminator.to(device)

    def reset_parameters(self):
        if isinstance(self.discriminator, MLP):
            self.discriminator.reset_parameters()
        else:
            linear_init(self.discriminator)

        self._prepare_for_new_batch()

        self.n_training_calls = 0

    def _prepare_for_new_batch(self):
        """Prepares the confuser for a new batch."""
        self.discriminator_losses = torch.tensor(0.,
                                                 requires_grad=True,
                                                 dtype=torch.float,
                                                 device=device)
        self.generator_losses = self.discriminator_losses.clone()
        self.to_backprop_generator = None

    def _scale_generator_loss(self, generator_loss, main_loss=None):
        """Scales the generator loss."""
        max_scale = self.get_max_scale(True)
        if main_loss is not None:
            if generator_loss > max_scale * main_loss:
                scaling_factor = (max_scale * main_loss / generator_loss).detach()
            else:
                scaling_factor = 1
        else:
            scaling_factor = max_scale

        return generator_loss * scaling_factor

    def _compute_1_loss(self, criterion, inputs, targets, seq_len, mask,
                        is_multi_call, to_summarize_stats=None):
        """Computes one single loss."""
        if to_summarize_stats is None:
            to_summarize_stats = inputs
        inputs = torch.cat((inputs, self.summarize_stats(inputs)), dim=-1)
        outputs = self.discriminator(inputs)

        if targets is None:
            targets = self.default_targets.expand_as(outputs)

        losses = criterion(outputs, targets).squeeze(-1)

        if mask is not None:
            losses.masked_fill_(mask, 0.)

        if is_multi_call:
            # mean of all besides batch size
            losses = batch_reduction_f(losses, torch.mean)
            losses = losses + losses / seq_len

        return losses

    def compute_loss(self, inputs,
                     targets=None,
                     seq_len=None,
                     max_losses=None,
                     mask=None,
                     is_multi_call=False,
                     to_summarize_stats=None):
        """Computes the loss for the confuser.

        inputs (torch.tensor): inputs to the confuser. I.e where you want to remove
            the `targets` from.
        targets (torch.tensor, optional): targets of the confuser. I.e what you want to remove
            from the `inputs`. If `None` will use `default_targets`.
        seq_len (int or torch.tensor, optional): number of calls per batch / sequence
            length. This is used so that the loss is an average, not a sum of losses.
            I.e to make it independant of the number of calls. The shape of it must
            be broadcastable with the shape of the output of the criterion.
        max_losses (float or torch.tensor, optional): max losses. This is used so
            that you don't add unnessessary noise when the generator is "confused"
            enough. The current losses will be clamped in a leaky manner, then
            hard manner f reaches `max_losses*2`. The shape of  max_losses must
            be broadcastable with the shape of the output of the criterion. It is
            important to have an "admissible" heuristic, i.e an upper bound that
            is never greater than the real bound. If not when the confuser outputs
            some random noise, the generator would be regularized in a random manner.
        mask (torch.tensor, optional): mask to apply to the output of the criterion.
            Should be used to mask when varying sequence lengths. The shape of mask
            must be broadcastable with the shape of the output of the criterion.
        is_multi_call (bool, optional): whether will make multiple calls of
            `compute_loss` between `backward`. In this case max_losses cannot
            clamp the loss at each call but only the final average loss over all
            calls.
        to_summarize_stats (Tensor, optional): tensor whose summary statistics
            will be added as features. If `None` then use the whole input Tensor.
        """
        if self.n_training_calls > self.n_steps_discriminate_only:
            self.generator_losses = self._compute_1_loss(self.generator_criterion,
                                                         inputs, targets, seq_len,
                                                         mask, is_multi_call,
                                                         to_summarize_stats)

            if max_losses is not None:
                self.to_backprop_generator = self.generator_losses < max_losses

        self.discriminator_losses = self._compute_1_loss(self.discriminator_criterion,
                                                         inputs.detach(), targets,
                                                         seq_len, mask, is_multi_call,
                                                         to_summarize_stats)

    def __call__(self, main_loss=None, additional=None, name="", **kwargs):
        """
        Computes the gradient of the generator parameters to minimize the
        confuing loss and of the discriminator parameters to maximize the same
        loss.

        Note:
            Should call model.zero_grad() at the end to be sure that clean slate.
        """
        if self.to_backprop_generator is not None and bool(self.to_backprop_generator.any()):
            if self.to_backprop_generator is not None:
                generator_losses = self.generator_losses[self.to_backprop_generator]
            else:
                generator_losses = self.generator_losses

            # generator should try maximizing loss so inverse sign
            generator_loss = -1 * generator_losses.mean()
            generator_loss = self._scale_generator_loss(generator_loss, main_loss)

            if additional is not None:
                add_to_visualize(generator_losses.mean().item(),
                                 "losses_generator_{}".format(name),
                                 to_visualize=additional.get("visualize", None),
                                 is_training=True,
                                 training_step=self.n_training_calls)

                add_to_visualize(generator_loss.item(),
                                 "losses_weighted_generator_{}".format(name),
                                 to_visualize=additional.get("visualize", None),
                                 is_training=True,
                                 training_step=self.n_training_calls)

            # has to retain graph to not recompute all
            generator_loss.backward(retain_graph=True)
            self.generator_optim.step()
            self.generator.zero_grad()
            # retaining graph => has to zero the grad of the discriminator also
            self.discriminator.zero_grad()

        discriminator_loss = self.discriminator_losses.mean()

        if additional is not None:
            add_to_visualize(discriminator_loss.item(),
                             "losses_discriminator_{}".format(name),
                             to_visualize=additional.get("visualize", None),
                             is_training=True,
                             training_step=self.n_training_calls)

        discriminator_loss.backward(**kwargs)
        self.discriminator_optim.step()
        self.discriminator.zero_grad()

        self._prepare_for_new_batch()
        self.n_training_calls += 1


def get_p_generator_criterion(generator_criterion):
    """Get `p` (from lp norm) given the generator criterion."""
    if isinstance(generator_criterion, nn.L1Loss):
        return 1
    elif isinstance(generator_criterion, nn.MSELoss):
        return 2
    elif generator_criterion.__name__ == "_l05loss":
        return 0.5
    else:
        raise ValueError("Please define p for {}".format(generator_criterion))


# KEY / QUERY CONFUSER SPECIFIC

def _precompute_max_loss(p, max_n=100):
    """
    Precomputes a table of length `max_n`, containing the expected maximum loss
    of a range of number when approximating them with their means depending on p used.
    `max_loss = ∑_{i=1}^n (i-N/2)**p`
    """
    return torch.tensor(np.array([np.mean([np.abs(i - (n + 1) / 2)**p
                                           for i in range(1, n + 1)])
                                  for n in range(0, max_n)], dtype=np.float32)
                        ).float().to(device)


MAX_LOSSES_P05 = _precompute_max_loss(0.5)


def get_max_loss_loc_confuser(input_lengths_tensor, p=2, factor=1):
    """
    Returns the expected maximum loss of the key confuser depending on p used.
    `max_loss = ∑_{i=1}^n (i-N/2)**p`

    Args:
        input_lengths_list (tensor): Float tensor containing the legnth of each
            sentence of the batch. Should already be on the correc device.
        p (float, optional): p of the Lp pseudo-norm used as loss.
        factor (float, optional): by how much to decrease the maxmum loss. If factor
            is 2 it means that you consider that the maximum loss will be achieved
            if your prediction is 1/factor (i.e half) way between the correct i
            and the best worst case output N/2. Factor = 10 means it can be a lot
            closer to i. This is usefull as there will always be some noise, and you
            don't want to penalize the model for some noise.
    """

    # E[(i-N/2)**2] = VAR(i) = (n**2 - 1)/12
    if p == 2:
        max_losses = (input_lengths_tensor**2 - 1) / 12
    elif p == 1:
        # computed by hand and use modulo because different if odd
        max_losses = (input_lengths_tensor**2 - input_lengths_tensor % 2) / (4 * input_lengths_tensor)
    elif p == 0.5:
        max_losses = MAX_LOSSES_P05[input_lengths_tensor.long()]
    else:
        raise ValueError("Unkown p={}".format(p))

    max_losses = max_losses / (factor**p)

    return max_losses


def confuse_keys_queries(lengths_tensor, kq, confuser, counter, is_training):
    """
    Confuse the keys by removing the counter i (i.e ehich source word ) from
    the hidden activations of the encoder.

    Note:
        - If the generator has no information about i, the best prediction
        for the discriminator it can get without any information is always
        predict N/2
        - Use a annealing factor to make the bound tighter and tighter to
        the real best prediction : N/2. Do not use low factor at begining
        as the discriminator will generate random numbers which could be correct
        and we don't want to penalize the generator for that.
        - Give N (the number of i to predict) as a feature to the key_confuser
        input to help the discriminator predicting N/2.

    Args:
        lengths_tensor (FloatTensor): tensor containing the batch length of the
            input / output
        kq (FloatTensor): key or query.
        confuser (Confuser): key or query confuser.
        counter (FloatTensor): counter from 1 to `max_len` to remove from the
            keys and queries.
        is_training (bool): whether in training mode.
    """
    batch_size, max_len, _ = kq.shape

    counting_target = counter.expand(batch_size, -1)[:, :max_len]

    # masks everything which finished decoding
    mask_finish = counting_target > lengths_tensor.unsqueeze(1)

    p = get_p_generator_criterion(confuser.generator_criterion)

    max_losses = get_max_loss_loc_confuser(lengths_tensor,
                                           p=p,
                                           factor=confuser.get_factor(is_training))

    # gives N as a feature to make it simpler to output N/2
    N = lengths_tensor.view(-1, 1, 1).expand(-1, max_len, 1)
    confuser_input = torch.cat([kq, N], dim=-1)
    confuser.compute_loss(confuser_input,
                          targets=counting_target.unsqueeze(-1),
                          seq_len=lengths_tensor.unsqueeze(-1),
                          max_losses=max_losses.unsqueeze(-1),
                          mask=mask_finish,
                          to_summarize_stats=kq)


# OPTIMIZER

class AdamPre(BaseOptimizer):
    """Implements Adam algorithm with prediction step.
    This class implements lookahead version of Adam Optimizer.
    The structure of class is similar to Adam class in Pytorch.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Note:
        - This is an optimizer that was designed specifically for Mini Max Games:
        Stabilizing Adversarial Nets With Prediction Methods.
        - Code copied from https://github.com/shahsohil/stableGAN
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, name='NotGiven'):
        self.name = name
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamPre, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                    state['oldWeights'] = p.data.clone()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** min(state['step'], 1022)
                bias_correction2 = 1 - beta2 ** min(state['step'], 1022)
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss

    def stepLookAhead(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                temp_grad = p.data.sub(state['oldWeights'])
                state['oldWeights'].copy_(p.data)
                p.data.add_(temp_grad)
        return loss

    def restoreStepLookAhead(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                p.data.copy_(state['oldWeights'])
        return loss
