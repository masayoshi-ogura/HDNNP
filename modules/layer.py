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

    def backprop(self, output_error, doutput_error, d_size):
        self._weight_grad = np.dot(self._input.T, output_error) \
            + hp.mixing_beta / d_size * np.tensordot(self._dinput, doutput_error, ((0, 1), (0, 1)))
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
    def mu_EMA(self):
        return self._mu_EMA

    @mu_EMA.setter
    def mu_EMA(self, mu):
        if hasattr(self, '_mu_EMA'):
            self._mu_EMA = hp.smooth_factor * mu + (1 - hp.smooth_factor) * self._mu_EMA
        else:
            self._mu_EMA = mu

    @property
    def sigma_EMA(self):
        return self._sigma_EMA

    @sigma_EMA.setter
    def sigma_EMA(self, sigma):
        if hasattr(self, '_sigma_EMA'):
            self._sigma_EMA = hp.smooth_factor * sigma + (1 - hp.smooth_factor) * self._sigma_EMA
        else:
            self._sigma_EMA = sigma

    def feedforward(self, input, dinput, batch_size, mode, eps=1e-5):
        if mode == 'training':
            mu = np.mean(input, axis=0)
            sigma = np.sqrt(np.var(input, axis=0) + eps)
            self.mu_EMA = mu
            self.sigma_EMA = sigma
        elif mode == 'test':
            mu = self._mu_EMA
            sigma = self._sigma_EMA
        self._norm = (input - mu) / sigma
        self._deriv_input = self._gamma * ((batch_size - 1) - self._norm**2) / (batch_size * sigma)
        output = self._gamma * self._norm + self._beta
        doutput = self._deriv_input[:, None, :] * dinput
        return output, doutput

    def backprop(self, output_error, doutput_error, d_size):
        self._beta_grad = np.sum(output_error, axis=0) \
            + hp.mixing_beta / d_size * np.sum(doutput_error, axis=(0, 1))
        self._gamma_grad = np.sum(output_error * self._norm, axis=0) \
            + hp.mixing_beta / d_size * np.sum(doutput_error * self._norm[:, None, :], axis=(0, 1))

        input_error = self._deriv_input * output_error
        dinput_error = self._deriv_input[:, None, :] * doutput_error
        return input_error, dinput_error
