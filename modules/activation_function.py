import numpy as np
from scipy.special import expit


def sigmoid(x):
    return expit(x)


def tanh(x):
    return np.tanh(x)


def relu(x):
    np.clip(x, 0, np.finfo(x.dtype).max)
    return x


ACTIVATIONS = {'tanh': tanh, 'sigmoid': sigmoid, 'relu': relu}


def deriv_tanh(Z, delta):
    # expit(x) * (1 - expit(x))
    return


def deriv_sigmoid(Z, delta):
    return


def deriv_relu(Z, delta):
    # 0 for x < 0
    # 1 for x > 0
    return


DERIVATIVES = {'tanh': deriv_tanh, 'sigmoid': deriv_sigmoid, 'relu': deriv_relu}
