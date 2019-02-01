"""
Base classes

Contact: Yann Dubois
"""
import abc

import torch.nn as nn

from seq2seq.util.helpers import add_to_test, add_to_visualize
from seq2seq.util.initialization import weights_init


class Module(abc.ABC, nn.Module):
    """
    Base class for modules.

    Args:
        is_dev_mode (bool, optional): whether to store many useful variables in
            `additional`. Useful when predicting with a trained model in dev mode
             to understand what the model is doing. Use with `dev_predict`.
        is_viz_train (bool, optional): whether to save how the averages of some
            intepretable variables change during training in `_to_visualize`.
    """

    def __init__(self, is_dev_mode=False, is_viz_train=False):
        super(Module, self).__init__()

        self._to_visualize = dict()
        self._to_test = dict()
        self._regularization_losses = dict()
        self.storer = dict()
        self.n_training_calls = 0
        self.set_dev_mode(value=is_dev_mode)
        self.set_viz_train(value=is_viz_train)

    @abc.abstractmethod
    def extra_repr(self):
        """Set the extra information about this module for printing.

        Note:
            - Should use `seq2seq.util.herlpers.get_extra_repr`
        """
        pass

    @property
    def is_regularize(self):
        """Whether accepts to use possible regularization losses."""
        return self.training

    def _update_n_training_calls(self, is_update=True):
        """Update the current model."""
        if is_update and self.training:
            self.n_training_calls += 1

            for child in self.children():
                if isinstance(child, Module):
                    child._update_n_training_calls()

    def set_dev_mode(self, value=True):
        """
        Sets dev mode. If `True`, store many useful variables in `_to_test`. Useful
        when predicting with a trained model in dev mode to understand what the
        model is doing.
        """
        self.is_dev_mode = value
        for child in self.children():
            try:
                child.set_dev_mode(value=value)
            except (AttributeError, NotImplementedError):
                pass

    def set_viz_train(self, value=True):
        """
        Sets visualization train mode. If `True`, store how the averages of some
            intepretable variables change during training in `_to_visualize`.
        """
        self.is_viz_train = value
        for child in self.children():
            try:
                child.set_viz_train(value=value)
            except (AttributeError, NotImplementedError):
                pass

    def reset_parameters(self):
        """
        Reset the parameters of this and all submodules.

        Note:
            - Should always call this function if overrides it and didn't
            care of submodules.
        """
        self.n_training_calls = 0
        for child in self.children():
            child.apply(weights_init)

        self._to_visualize = dict()
        self._to_test = dict()
        self._regularization_losses = dict()
        self.storer = dict()

        self.is_resetted = True

    def flatten_parameters(self):
        """Flattens the parameters of this and all submodules."""
        for child in self.children():
            try:
                child.flatten_parameters()
            except (AttributeError, NotImplementedError):
                pass

    def add_to_test(self, values, keys):
        """
        Save a variable to `self._to_test` only if dev mode is on. The
        variables saved should be the interpretable ones for which you want to
        know the value of during test time.

        Note:
            - Batch size should always be 1 when predicting with dev mode.

        Args:
            values (list or item): list or item to save in `self._to_test`.
            keys (list or string): list or string of names of the variables corresponding
                to the `values`.
        """
        add_to_test(values, keys, self._to_test, self.is_dev_mode)

    def add_to_visualize(self, values, keys, **kwargs):
        """
        Every `save_every_n_batches` batch, adds a certain variable to the
        `visualization` dictionary. Such variables should be the ones that are
        interpretable, and for which the size is independant of the source length.
        I.e avaregae over the source length if it is dependant.

        Args:
            values (list or item): list or item to save in `self._to_visualize`.
            keys (list or string): list or string of names of the variables corresponding
                to the `values`.
        """
        add_to_visualize(values, keys, self._to_visualize, self.training,
                         self.n_training_calls, **kwargs)

    def get_to_test(self, is_reset=True):
        """
        Get all the variables that can be interesting at test time from the current
        module and its childrens. The dictionary will be non empty only if
        `dev mode` is on.

        Args:
            is_reset (bool, optional): whether to remove the values to test
                from the class after returning them.
        """
        to_test = self._to_test

        for child in self.children():
            try:
                to_test.update(child.get_to_test(is_reset=is_reset))
            except (AttributeError, NotImplementedError):
                pass

        if is_reset:
            self._to_test = dict()

        return to_test

    def get_to_visualize(self, is_reset=True):
        """
        Get all the variables that should be visualized from the current module
        and its childrens. I.e that are interpretable, and whose size independant
        of the source length. If the size is dependant on the source length, you
        can average over the source length.

        Args:
            is_reset (bool, optional): whether to remove the values to visualize
                from the class after returning them.
        """
        to_visualize = self._to_visualize

        for child in self.children():
            try:
                to_visualize.update(child.get_to_visualize(is_reset=is_reset))
            except (AttributeError, NotImplementedError):
                pass

        if is_reset:
            self._to_visualize = dict()

        return to_visualize

    def add_regularization_loss(self, loss_name, loss):
        """
        Save a regularization loss.

        Args:
            loss_name (string): name of the loss to add.
            loss (tensor): value of the loss to add.
        """
        assert self.is_regularize
        if (loss < 0).any():
            raise ValueError("Negative loss {} : {}".format(loss_name, loss))
        self._regularization_losses[loss_name] = loss

    def get_regularization_losses(self, is_reset=True):
        """
        Get all the regularization losses from the current module and its childrens.

        Args:
            is_reset (bool, optional): whether to remove the vlosses from the
                class after returning them.
        """
        if not self.is_regularize:
            return dict()

        regularization_losses = self._regularization_losses

        for child in self.children():
            try:
                regularization_losses.update(child.get_regularization_losses(is_reset=is_reset))
            except (AttributeError, NotImplementedError):
                pass

        if is_reset:
            self._regularization_losses = dict()

        return regularization_losses
