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
