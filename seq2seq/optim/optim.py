"""Optimizer class.

NOTA BENE:
- I haven't changed much here.
"""
import itertools

import torch


def get_optim(optim_name):
    optims = {'adam': torch.optim.Adam, 'adagrad': torch.optim.Adagrad,
              'adadelta': torch.optim.Adadelta, 'adamax': torch.optim.Adamax,
              'rmsprop': torch.optim.RMSprop, 'sgd': torch.optim.SGD,
              None: torch.optim.Adam}
    return optims[optim_name]


class Optimizer(object):
    """ The Optimizer class encapsulates torch.optim package and provides functionalities
    for learning rate scheduling and gradient norm clipping.

    Args:
        optim (torch.optim.Optimizer): optimizer object.
        max_grad_norm (float, optional): value used for gradient norm clipping,
            set None to disable (default None)
    """

    _ARG_MAX_GRAD_NORM = 'max_grad_norm'

    def __init__(self, optim, param,
                 scheduler=None,
                 max_grad_norm=None,
                 max_grad_value=None,
                 scheduler_kwargs={},
                 **kwargs):
        self.optimizer = optim(param, **kwargs)
        self.scheduler = (scheduler(self.optimizer, **scheduler_kwargs)
                          if scheduler is not None else None)
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value

    def step(self):
        """ Performs a single optimization step, including gradient norm clipping if necessary. """
        if self.max_grad_value is not None:
            params = itertools.chain.from_iterable([group['params']
                                                    for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_value_(params, self.max_grad_norm)
        if self.max_grad_norm is not None:
            params = itertools.chain.from_iterable([group['params']
                                                    for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()

    def update(self, loss, epoch):
        """ Update the learning rate if the criteria of the scheduler are met.

        Args:
            loss (float): The current loss.  It could be training loss or developing loss
                depending on the caller.  By default the supervised trainer uses developing
                loss.
            epoch (int): The current epoch number.
        """
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()
