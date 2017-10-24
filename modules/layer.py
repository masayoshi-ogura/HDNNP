# -*- coding: utf-8 -*-

import numpy as np

# from config import hp
from activation_function import ACTIVATIONS, DERIVATIVES, SECOND_DERIVATIVES


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
    def __init__(self, input_nodes, output_nodes, final):
        self._ninput = input_nodes
        self._weight = np.random.normal(0.0, 1.0, (input_nodes, output_nodes))
        self._bias = np.random.normal(0.0, 1.0, (output_nodes))
        self._final = final

    @property
    def parameter(self):
        return self._weight, self._bias

    @parameter.setter
    def parameter(self, parameter):
        self._weight, self._bias = parameter

    @property
    def gradient(self):
        return self._weight_grad, self._bias_grad

    def feedforward(self, input, dinput):
        self._input = input
        self._dinput = dinput
        output = np.dot(input, self._weight) + self._bias
        doutput = np.tensordot(dinput, self._weight, ((2,), (0,)))
        if not self._final:
            output /= self._ninput
            doutput /= self._ninput
        return output, doutput

    def backprop(self, output_error, doutput_error1, doutput_error2):
        if not self._final:
            output_error /= self._ninput
            doutput_error1 /= self._ninput
            doutput_error2 /= self._ninput

        self._weight_grad = np.dot(self._input.T, output_error) \
            + np.tensordot(self._dinput, doutput_error1, ((0, 1), (0, 1))) \
            + np.tensordot(self._input, doutput_error2, ((0,), (0,))).sum(axis=1)
        self._bias_grad = np.sum(output_error, axis=0) \
            + np.sum(doutput_error2, axis=(0, 1))

        input_error = np.dot(output_error, self._weight.T)
        dinput_error1 = np.tensordot(doutput_error1, self._weight, ((2,), (1,)))
        dinput_error2 = np.tensordot(doutput_error2, self._weight, ((2,), (1,)))
        return input_error, dinput_error1, dinput_error2


class ActivationLayer(LayerBase):
    def __init__(self, activation):
        self._activation = ACTIVATIONS[activation]
        self._deriv_activation = DERIVATIVES[activation]
        self._second_deriv_activation = SECOND_DERIVATIVES[activation]

    def feedforward(self, input, dinput):
        self._deriv = self._deriv_activation(input)
        self._deriv1 = self._deriv_activation(input)[:, None, :]
        self._deriv2 = self._second_deriv_activation(input)[:, None, :] * dinput

        output = self._activation(input)
        doutput = self._deriv1 * dinput
        return output, doutput

    def backprop(self, output_error, doutput_error1, doutput_error2):
        input_error = self._deriv * output_error
        dinput_error1 = self._deriv1 * doutput_error1
        dinput_error2 = self._deriv1 * doutput_error2 + self._deriv2 * doutput_error1
        return input_error, dinput_error1, dinput_error2
#
#
# class BatchNormalizationLayer(LayerBase):
#     def __init__(self, nodes, trainable=True):
#         self._beta = np.zeros(nodes)
#         self._gamma = np.ones(nodes)
#         self._trainable = trainable
#
#     @property
#     def parameter(self):
#         if self._trainable:
#             return self._beta, self._gamma
#         else:
#             return ()
#
#     @parameter.setter
#     def parameter(self, parameter):
#         if self._trainable:
#             self._beta, self._gamma = parameter
#
#     @property
#     def gradient(self):
#         if self._trainable:
#             return self._beta_grad, self._gamma_grad
#         else:
#             return ()
#
#     @property
#     def mean_EMA(self):
#         return self._mean_EMA
#
#     @mean_EMA.setter
#     def mean_EMA(self, mean):
#         if hasattr(self, '_mean_EMA'):
#             self._mean_EMA = hp.smooth_factor * mean + (1 - hp.smooth_factor) * self._mean_EMA
#         else:
#             self._mean_EMA = mean
#
#     @property
#     def variance_EMA(self):
#         return self._variance_EMA
#
#     @variance_EMA.setter
#     def variance_EMA(self, variance):
#         if hasattr(self, '_variance_EMA'):
#             self._variance_EMA = hp.smooth_factor * variance + (1 - hp.smooth_factor) * self._variance_EMA
#         else:
#             self._variance_EMA = variance
#
#     def feedforward(self, input, dinput, batch_size, mode, eps=1e-3):
#         if mode == 'training':
#             mean = np.mean(input, axis=0)
#             variance = np.var(input, axis=0)
#             self.mean_EMA = mean
#             self.variance_EMA = variance
#         elif mode == 'test':
#             mean = self._mean_EMA
#             variance = self._variance_EMA
#         stddev = np.sqrt(variance + eps)
#         self._norm = (input - mean) / stddev
#         self._deriv = (batch_size - 1 - self._norm**2) / (batch_size * stddev)
#         self._deriv2 = self._deriv * (1 - (3. / batch_size) / stddev * (input * self._norm))
#
#         output = self._gamma * self._norm + self._beta
#         doutput = (self._gamma * self._deriv)[:, None, :] * dinput
#         return output, doutput
#
#     def backprop(self, output_error, doutput_error, batch_size, nderivative):
#         if self._trainable:
#             self._beta_grad = np.mean(output_error, axis=0)
#             self._gamma_grad = 1./batch_size * (np.sum(output_error * self._norm, axis=0)
#                                                 + hp.mixing_beta / nderivative * np.sum(doutput_error * self._norm[:, None, :], axis=(0, 1)))
#
#         input_error = self._gamma * self._deriv * output_error
#         dinput_error = (self._gamma * self._deriv2)[:, None, :] * doutput_error
#         return input_error, dinput_error
