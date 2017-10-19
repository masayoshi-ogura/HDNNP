import numpy as np
from scipy.special import expit


def identity(x):
    return x


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return expit(x)


def relu(x):
    return np.clip(x, 0., None)


def leaky_relu(x, a=0.01):
    ret = x.copy()
    ret[x < 0.] *= a
    return ret


def elu(x):
    ret = x.copy()
    filter = x < 0.
    ret[filter] = np.exp(x[filter]) - 1.
    return ret


def truncated(x, n=2):
    return np.clip(x, 0., None)**n


def softplus(x):
    return np.log(np.exp(x) + 1.)


ACTIVATIONS = {
               'identity': identity,
               'tanh': tanh,
               'sigmoid': sigmoid,
               'relu': relu,
               'leaky_relu': leaky_relu,
               'elu': elu,
               'truncated': truncated,
               'softplus': softplus,
               }


def d_identity(x):
    return np.ones_like(x)


def d_tanh(x):
    return 1. - np.tanh(x)**2


def d_sigmoid(x):
    return expit(x) * (1. - expit(x))


def d_relu(x):
    return (x > 0.).astype(int)


def d_leaky_relu(x, a=0.01):
    ret = np.ones_like(x)
    ret[x < 0.] = a
    return ret


def d_elu(x):
    ret = np.ones_like(x)
    filter = x < 0.
    ret[filter] = np.exp(x[filter])
    return ret


def d_truncated(x, n=2):
    return n * np.clip(x, 0., None)**(n-1)


def d_softplus(x):
    return 1. / (1. + np.exp(-x))


DERIVATIVES = {
               'identity': d_identity,
               'tanh': d_tanh,
               'sigmoid': d_sigmoid,
               'relu': d_relu,
               'leaky_relu': d_leaky_relu,
               'elu': d_elu,
               'truncated': d_truncated,
               'softplus': d_softplus,
               }


def d2_identity(x):
    return np.zeros_like(x)


def d2_tanh(x):
    return - 2. * np.tanh(x) * (1. - np.tanh(x)**2)


def d2_sigmoid(x):
    return expit(x) * (1. - expit(x)) * (1. - 2. * expit(x))


def d2_relu(x):
    return np.zeros_like(x)


def d2_leaky_relu(x):
    return np.zeros_like(x)


def d2_elu(x):
    ret = np.zeros_like(x)
    filter = x < 0.
    ret[filter] = np.exp(x[filter])
    return ret


def d2_truncated(x, n=2):
    return n * (n-1) * np.clip(x, 0., None)**(n-2)


def d2_softplus(x):
    return np.exp(x) / (1. + np.exp(x))**2


SECOND_DERIVATIVES = {
                      'identity': d2_identity,
                      'tanh': d2_tanh,
                      'sigmoid': d2_sigmoid,
                      'relu': d2_relu,
                      'leaky_relu': d2_leaky_relu,
                      'elu': d2_elu,
                      'truncated': d2_truncated,
                      'softplus': d2_softplus,
                      }
