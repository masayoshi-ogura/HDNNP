import numpy as np

from config import hp


class SGDOptimizer(object):
    def __init__(self, weights, bias, nesterov=True):
        self.weights = weights
        self.bias = bias
        self.nesterov = nesterov
        if nesterov:
            self.weight_velocities = [np.zeros_like(w) for w in weights]
            self.bias_velocities = [np.zeros_like(b) for b in bias]

    def update_params(self, weight_grads, bias_grads):
        if self.nesterov:
            weight_updates = [hp.momentum * velocity - hp.learning_rate * grad
                              for velocity, grad in zip(self.weight_velocities, weight_grads)]
            bias_updates = [hp.momentum * velocity - hp.learning_rate * grad
                            for velocity, grad in zip(self.bias_velocities, bias_grads)]
            self.weight_velocities = weight_updates
            self.bias_velocities = bias_updates
        else:
            weight_updates = [- hp.learning_rate * grad for grad in weight_grads]
            bias_updates = [- hp.learning_rate * grad for grad in bias_grads]

        for weight, bias, weight_update, bias_update in zip(self.weights, self.bias, weight_updates, bias_updates):
            weight += weight_update
            bias += bias_update

        return self.weights, self.bias


class AdamOptimizer(object):
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.epsilon = hp.epsilon
        self.t = 0
        self.weight_ms = [np.zeros_like(w) for w in self.weights]
        self.weight_vs = [np.zeros_like(w) for w in self.weights]
        self.bias_ms = [np.zeros_like(b) for b in self.bias]
        self.bias_vs = [np.zeros_like(b) for b in self.bias]

    def update_params(self, weight_grads, bias_grads):
        self.t += 1

        self.weight_ms = [hp.adam_beta1 * m + (1 - hp.adam_beta1) * grad
                          for m, grad in zip(self.weight_ms, weight_grads)]
        self.weight_vs = [hp.adam_beta2 * v + (1 - hp.adam_beta2) * (grad ** 2)
                          for v, grad in zip(self.weight_vs, weight_grads)]

        self.bias_ms = [hp.adam_beta1 * m + (1 - hp.adam_beta1) * grad
                        for m, grad in zip(self.bias_ms, bias_grads)]
        self.bias_vs = [hp.adam_beta2 * v + (1 - hp.adam_beta2) * (grad ** 2)
                        for v, grad in zip(self.bias_vs, bias_grads)]

        weight_updates = [- hp.learning_rate *
                          (m / (1 - hp.adam_beta1**self.t)) /
                          (np.sqrt(v / (1 - hp.adam_beta2**self.t)) + self.epsilon)
                          for m, v in zip(self.weight_ms, self.weight_vs)]
        bias_updates = [- hp.learning_rate *
                        (m / (1 - hp.adam_beta1**self.t)) /
                        (np.sqrt(v / (1 - hp.adam_beta2**self.t)) + self.epsilon)
                        for m, v in zip(self.bias_ms, self.bias_vs)]

        for weight, bias, weight_update, bias_update in zip(self.weights, self.bias, weight_updates, bias_updates):
            weight += weight_update
            bias += bias_update

        return self.weights, self.bias


class BFGSOptimizer(object):
    pass


OPTIMIZERS = {'sgd': SGDOptimizer, 'adam': AdamOptimizer, 'bfgs': BFGSOptimizer}
