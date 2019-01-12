# coding: utf-8

"""Loss function to optimize 1st-order property."""

import chainer
import chainer.functions as F

from hdnnpy.training.loss_function.loss_functions_base import (
    LossFunctionBase)


class First(LossFunctionBase):
    """Loss function to optimize 1st-order property."""
    name = 'first'
    """str: Name of this loss function class."""
    order = {
        'descriptor': 1,
        'property': 1,
        }
    """dict: Required orders of each dataset to calculate loss function.
    """

    def __init__(self, model, properties, **_):
        """
        Args:
            model (HighDimensionalNNP):
                HDNNP object to optimize parameters.
            properties (list [str]): Names of properties to optimize.
        """
        super().__init__(model)
        self._observation_keys = [f'RMSE/{properties[1]}', 'RMSE/total']

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
        _, pred1, *_ = predictions
        _, true1, *_ = labels
        loss1 = F.mean_squared_error(pred1, true1)
        RMSE1 = F.sqrt(loss1)

        observation = {
            self._observation_keys[0]: RMSE1,
            self._observation_keys[1]: RMSE1,
            }
        chainer.report(observation, observer=self._model)
        return loss1
