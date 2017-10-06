import numpy as np
import math

from config import hp


class SGDOptimizer(object):
    def __init__(self, params, nesterov=True, momentum=0.9):
        self._params = params
        self._t = 0
        self._nesterov = nesterov
        self._momentum = momentum
        if nesterov:
            self.velocities = [np.zeros_like(param) for param in params]

    @property
    def params(self):
        return self._params

    def update_params(self, grads):
        # learning_rate = hp.learning_rate / (1 + hp.learning_rate_decay * self._t)
        learning_rate = hp.learning_rate * math.exp(- hp.learning_rate_decay * self._t)
        self._t += 1

        if self._nesterov:
            updates = [self._momentum * velocity - learning_rate * grad
                       for velocity, grad in zip(self.velocities, grads)]
            self.velocities = updates
        else:
            updates = [- learning_rate * grad for grad in grads]
        self._params = [param + update for param, update in zip(self._params, updates)]


class AdamOptimizer(object):
    def __init__(self, params, beta1=0.9, beta2=0.999):
        self._params = params
        self._ms = [np.zeros_like(param) for param in params]
        self._vs = [np.zeros_like(param) for param in params]
        self._t = 0
        self._beta1 = beta1
        self._beta2 = beta2

    @property
    def params(self):
        return self._params

    def update_params(self, grads, eps=1e-8):
        # learning_rate = hp.learning_rate / (1 + hp.learning_rate_decay * self._t)
        learning_rate = hp.learning_rate * math.exp(- hp.learning_rate_decay * self._t)
        self._t += 1

        self._ms = [self._beta1 * m + (1 - self._beta1) * grad
                    for m, grad in zip(self._ms, grads)]
        self._vs = [self._beta2 * v + (1 - self._beta2) * grad**2
                    for v, grad in zip(self._vs, grads)]
        self._params = [param - learning_rate
                        * (m / (1 - self._beta1**self._t))
                        / (np.sqrt(v / (1 - self._beta2**self._t)) + eps)
                        for param, m, v in zip(self._params, self._ms, self._vs)]


class BFGSOptimizer(object):
    pass


OPTIMIZERS = {'sgd': SGDOptimizer, 'adam': AdamOptimizer, 'bfgs': BFGSOptimizer}
