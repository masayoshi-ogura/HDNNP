# -*- coding: utf-8 -*-

import numpy as np

from config import hp
from activation_function import ACTIVATIONS, DERIVATIVES


class LayerBase(object):
    @property
    def parameter(self):
        return ()

    @parameter.setter
    def parameter(self, _):
        pass

    @property
    def gradient(self):
        return ()


class FullyConnectedLayer(LayerBase):
    def __init__(self, input_nodes, output_nodes):
        self._weight = np.random.normal(0.0, 1.0, (input_nodes, output_nodes))
        self._bias = np.random.normal(0.0, 1.0, (output_nodes))

    @property
    def parameter(self):
        return self._weight, self._bias

    @parameter.setter
    def parameter(self, parameter):
        self._weight, self._bias = parameter

    @property
    def gradient(self):
        return self._weight_grad, self._bias_grad

    def feedforward(self, input, dinput, *_):
        self._input = input
        self._dinput = dinput
        output = np.dot(input, self._weight) + self._bias
        doutput = np.tensordot(dinput, self._weight, ((2,), (0,)))
        return output, doutput

    def backprop(self, output_error, doutput_error, batch_size, nderivative):
        self._weight_grad = 1./batch_size * (np.dot(self._input.T, output_error)
                                             + hp.mixing_beta / nderivative * np.tensordot(self._dinput, doutput_error, ((0, 1), (0, 1))))
        self._bias_grad = np.sum(output_error, axis=0)

        input_error = np.dot(output_error, self._weight.T)
        dinput_error = np.tensordot(doutput_error, self._weight, ((2,), (1,)))
        return input_error, dinput_error


class ActivationLayer(LayerBase):
    def __init__(self, activation):
        self._activation = ACTIVATIONS[activation]
        self._deriv_activation = DERIVATIVES[activation]

    def feedforward(self, input, dinput, *_):
        self._deriv_input = self._deriv_activation(input)
        output = self._activation(input)
        doutput = self._deriv_input[:, None, :] * dinput
        return output, doutput

    def backprop(self, output_error, doutput_error, *_):
        input_error = self._deriv_input * output_error
        dinput_error = self._deriv_input[:, None, :] * doutput_error
        return input_error, dinput_error


class BatchNormalizationLayer(LayerBase):
    def __init__(self, nodes):
        self._beta = np.zeros(nodes)
        self._gamma = np.ones(nodes)

    @property
    def parameter(self):
        return self._beta, self._gamma

    @parameter.setter
    def parameter(self, parameter):
        self._beta, self._gamma = parameter

    @property
    def gradient(self):
        return self._beta_grad, self._gamma_grad

    @property
    def mean_EMA(self):
        return self._mean_EMA

    @mean_EMA.setter
    def mean_EMA(self, mean):
        if hasattr(self, '_mean_EMA'):
            self._mean_EMA = hp.smooth_factor * mean + (1 - hp.smooth_factor) * self._mean_EMA
        else:
            self._mean_EMA = mean

    @property
    def variance_EMA(self):
        return self._variance_EMA

    @variance_EMA.setter
    def variance_EMA(self, variance):
        if hasattr(self, '_variance_EMA'):
            self._variance_EMA = hp.smooth_factor * variance + (1 - hp.smooth_factor) * self._variance_EMA
        else:
            self._variance_EMA = variance

    def feedforward(self, input, dinput, batch_size, mode, eps=1e-3):
        if mode == 'training':
            mean = np.mean(input, axis=0)
            variance = np.var(input, axis=0)
            self.mean_EMA = mean
            self.variance_EMA = variance
        elif mode == 'test':
            mean = self._mean_EMA
            variance = self._variance_EMA
        self._stddev = np.sqrt(variance + eps)
        self._coef = self._gamma / self._stddev
        self._xmean = input - mean
        self._norm = self._xmean / self._stddev
        output = self._gamma * self._norm + self._beta
        doutput = (self._coef / batch_size * ((batch_size - 1) - self._norm**2))[:, None, :] * dinput
        return output, doutput

    def backprop(self, output_error, doutput_error, batch_size, nderivative):
        self._beta_grad = 1./batch_size * (np.sum(output_error, axis=0)
                                           + hp.mixing_beta / nderivative * np.sum(doutput_error, axis=(0, 1)))
        self._gamma_grad = 1./batch_size * (np.sum(output_error * self._norm, axis=0)
                                            + hp.mixing_beta / nderivative * np.sum(doutput_error * self._norm[:, None, :], axis=(0, 1)))

        input_error = self._coef * (output_error - np.mean(output_error, axis=0)
                                    - self._xmean
                                    * np.mean(output_error * self._xmean, axis=0)
                                    / self._stddev**2)
        dinput_error = self._coef * (doutput_error - np.mean(doutput_error, axis=0)
                                     - self._xmean[:, None, :]
                                     * (np.mean(doutput_error * self._xmean[:, None, :], axis=0)
                                     / self._stddev**2)[None, :, :])
        return input_error, dinput_error
