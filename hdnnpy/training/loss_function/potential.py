# coding: utf-8

"""Loss function to optimize 0th property as scalar potential."""

import warnings

import chainer
import chainer.functions as F

from hdnnpy.training.loss_function.loss_functions_base import (
    LossFunctionBase)


class Potential(LossFunctionBase):
    """Loss function to optimize 0th property as scalar potential."""
    name = 'potential'
    """str: Name of this loss function class."""
    order = {
        'descriptor': 2,
        'property': 1,
        }
    """dict: Required orders of each dataset to calculate loss function.
    """

    def __init__(self, model, properties, mixing_beta=1.0, penalty=1e-3, **_):
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
            penalty (float, optional):
                Penalty term coefficient parameter. This loss function
                adds following penalty to 1st order property vector.
                :math:`\rot \bm{F} = 0`
        """
        assert 0.0 <= mixing_beta <= 1.0
        super().__init__(model)
        self._observation_keys = [
            f'RMSE/{properties[0]}', f'RMSE/{properties[1]}',
            f'RMSE/rot-{properties[1]}', 'RMSE/total']
        self._mixing_beta = mixing_beta
        self._penalty = penalty

        if mixing_beta == 0.0:
            warnings.warn(
                'If mixing_beta=0.0, you should use loss function type '
                '`zeroth` instead of `first`.')
        if penalty == 0.0:
            warnings.warn(
                'If penalty=0.0, you should use loss function type '
                '`first` instead of `first`.')

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
        inputs = [data for key, data in dataset.items()
                  if key.startswith('inputs')]
        labels = [data for key, data in dataset.items()
                  if key.startswith('labels')]
        predictions = self._model.predict(inputs, self.order['descriptor'])

        loss0 = F.mean_squared_error(predictions[0], labels[0])
        loss1 = F.mean_squared_error(predictions[1], labels[1])
        loss_rot = F.mean_squared_error(
            predictions[2], F.swapaxes(predictions[2], 2, 3))
        total_loss = ((1.0 - self._mixing_beta) * loss0
                      + self._mixing_beta * loss1
                      + self._penalty * loss_rot)

        RMSE0 = F.sqrt(loss0)
        RMSE1 = F.sqrt(loss1)
        RMSE_rot = F.sqrt(loss_rot)
        total_RMSE = ((1.0 - self._mixing_beta) * RMSE0
                      + self._mixing_beta * RMSE1)

        observation = {
            self._observation_keys[0]: RMSE0,
            self._observation_keys[1]: RMSE1,
            self._observation_keys[2]: RMSE_rot,
            self._observation_keys[3]: total_RMSE,
            }
        chainer.report(observation, observer=self._model)
        return total_loss
