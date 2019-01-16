# coding: utf-8

"""Loss function to optimize 0th and 1st-order property."""

import warnings

import chainer
import chainer.functions as F

from hdnnpy.training.loss_function.loss_functions_base import (
    LossFunctionBase)


class First(LossFunctionBase):
    """Loss function to optimize 0th and 1st-order property."""
    name = 'first'
    """str: Name of this loss function class."""
    order = {
        'descriptor': 1,
        'property': 1,
        }
    """dict: Required orders of each dataset to calculate loss function.
    """

    def __init__(self, model, properties, mixing_beta=1.0, **_):
        """
        Args:
            model (HighDimensionalNNP):
                HDNNP object to optimize parameters.
            properties (list [str]): Names of properties to optimize.
            mixing_beta (float, optional):
                Mixing parameter of errors of 0th and 1st order.
                It accepts 0.0 to 1.0. If 0.0 it optimizes HDNNP by only
                0th order property and it is equal to loss function
                ``Zeroth``. If 1.0 it optimizes HDNNP by only 1st order
                property.
        """
        assert 0.0 <= mixing_beta <= 1.0
        super().__init__(model)
        self._observation_keys = [
            f'RMSE/{properties[0]}', f'RMSE/{properties[1]}', 'RMSE/total']
        self._mixing_beta = mixing_beta

        if mixing_beta == 0.0:
            warnings.warn(
                'If mixing_beta=0.0, you should use loss function type '
                '`zeroth` instead of `first`.')

    def eval(self, **dataset):
        """Calculate loss function from given datasets and model.

        Args:
            **dataset (~numpy.ndarray):
                Datasets passed as kwargs. Name of each key is in the
                format 'inputs/N' or 'labels/N'. 'N' is the order of
                the dataset.

        Returns:
            ~chainer.Variable:
            A scalar value calculated with loss function.
        """
        inputs = [dataset[f'inputs/{i}'] for i
                  in range(self.order['descriptor'] + 1)]
        labels = [dataset[f'labels/{i}'] for i
                  in range(self.order['property'] + 1)]
        predictions = self._model.predict(inputs, self.order['descriptor'])
        loss0 = F.mean_squared_error(predictions[0], labels[0])
        loss1 = F.mean_squared_error(predictions[1], labels[1])
        total_loss = ((1.0 - self._mixing_beta) * loss0
                      + self._mixing_beta * loss1)

        RMSE0 = F.sqrt(loss0)
        RMSE1 = F.sqrt(loss1)
        total_RMSE = ((1.0 - self._mixing_beta) * RMSE0
                      + self._mixing_beta * RMSE1)

        observation = {
            self._observation_keys[0]: RMSE0,
            self._observation_keys[1]: RMSE1,
            self._observation_keys[2]: total_RMSE,
            }
        chainer.report(observation, observer=self._model)
        return total_loss
