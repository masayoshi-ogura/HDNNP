import numpy as np
from scipy.special import expit


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return expit(x)


def relu(x):
    np.clip(x, 0, np.finfo(x.dtype).max)
    return x


ACTIVATIONS = {'tanh': tanh, 'sigmoid': sigmoid, 'relu': relu}


def deriv_tanh(x):
    return 1 - np.tanh(x)**2


def deriv_sigmoid(x):
    return expit(x) * (1 - expit(x))


def deriv_relu(x):
    return (x > 0).astype(int)


DERIVATIVES = {'tanh': deriv_tanh, 'sigmoid': deriv_sigmoid, 'relu': deriv_relu}


def second_deriv_tanh(x):
    return -2 * (1 - np.tanh(x)**2) * np.tanh(x)


def second_deriv_sigmoid(x):
    return expit(x) * (1 - expit(x)) * (1 - 2 * expit(x))


def second_deriv_relu(x):
    return 0


SECOND_DERIVATIVES = {'tanh': second_deriv_tanh, 'sigmoid': second_deriv_sigmoid, 'relu': second_deriv_relu}
