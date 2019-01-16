# coding: utf-8

"""Loss function to optimize 0th-order property."""

import chainer
import chainer.functions as F

from hdnnpy.training.loss_function.loss_functions_base import (
    LossFunctionBase)


class Zeroth(LossFunctionBase):
    """Loss function to optimize 0th-order property."""
    name = 'zeroth'
    """str: Name of this loss function class."""
    order = {
        'descriptor': 0,
        'property': 0,
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
        self._observation_keys = [f'RMSE/{properties[0]}', 'total']

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
        RMSE0 = F.sqrt(loss0)

        observation = {
            self._observation_keys[0]: RMSE0,
            self._observation_keys[1]: RMSE0,
            }
        chainer.report(observation, observer=self._model)
        return loss0
