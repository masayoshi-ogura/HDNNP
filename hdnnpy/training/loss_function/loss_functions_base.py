# coding: utf-8

"""Base class for loss functions."""

from abc import (ABC, abstractmethod)


class LossFunctionBase(ABC):
    """Base class for loss functions."""
    name = None
    """str: Name of this loss function class."""
    order = {
        'descriptor': None,
        'property': None,
        }
    """dict: Required orders of each dataset to calculate loss function.
    """

    def __init__(self, model):
        """
        Args:
            model (HighDimensionalNNP):
                HDNNP object to optimize parameters.
        """
        self._model = model
        self._observation_keys = []

    @property
    def observation_keys(self):
        """list [str]: Names of metrics which trainer observes."""
        return self._observation_keys

    @abstractmethod
    def eval(self, **dataset):
        """Calculate loss function from given datasets and model.

        This is abstract method.
        Subclass of this base class have to override.

        Args:
            **dataset (~numpy.ndarray):
                Datasets passed as kwargs. Name of each key is in the
                format 'inputs/N' or 'labels/N'. 'N' is the order of
                the dataset.

        Returns:
            ~chainer.Variable:
            A scalar value calculated with loss function.
        """
        pass
