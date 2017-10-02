import numpy as np

from config import hp


class SGDOptimizer(object):
    def __init__(self, weights, bias, beta, gamma, nesterov=True):
        self.weights = weights
        self.bias = bias
        self.beta = beta
        self.gamma = gamma
        self.t = 0
        self.nesterov = nesterov
        if nesterov:
            self.weight_velocities = [np.zeros_like(w) for w in weights]
            self.bias_velocities = [np.zeros_like(b) for b in bias]
            self.beta_velocities = np.zeros_like(beta)
            self.gamma_velocities = np.zeros_like(gamma)

    def update_params(self, weight_grads, bias_grads, beta_grads, gamma_grads):
        learning_rate = hp.learning_rate / (1 + hp.learning_rate_decay * self.t)
        self.t += 1

        if self.nesterov:
            weight_updates = [hp.momentum * velocity - learning_rate * grad
                              for velocity, grad in zip(self.weight_velocities, weight_grads)]
            bias_updates = [hp.momentum * velocity - learning_rate * grad
                            for velocity, grad in zip(self.bias_velocities, bias_grads)]
            beta_updates = hp.momentum * self.beta_velocities - learning_rate * beta_grads
            gamma_updates = hp.momentum * self.gamma_velocities - learning_rate * gamma_grads
            self.weight_velocities = weight_updates
            self.bias_velocities = bias_updates
            self.beta_velocities = beta_updates
            self.gamma_velocities = gamma_updates
        else:
            weight_updates = [- learning_rate * grad for grad in weight_grads]
            bias_updates = [- learning_rate * grad for grad in bias_grads]
            beta_updates = - learning_rate * beta_grads
            gamma_updates = - learning_rate * gamma_grads

        for weight, bias, weight_update, bias_update in zip(self.weights, self.bias, weight_updates, bias_updates):
            weight += weight_update
            bias += bias_update
        self.beta += beta_updates
        self.gamma += gamma_updates

        return self.weights, self.bias, self.beta, self.gamma


class AdamOptimizer(object):
    def __init__(self, weights, bias, beta, gamma):
        self.weights = weights
        self.bias = bias
        self.beta = beta
        self.gamma = gamma
        self.t = 0
        self.weight_ms = [np.zeros_like(w) for w in self.weights]
        self.weight_vs = [np.zeros_like(w) for w in self.weights]
        self.bias_ms = [np.zeros_like(b) for b in self.bias]
        self.bias_vs = [np.zeros_like(b) for b in self.bias]
        self.beta_ms = np.zeros_like(beta)
        self.beta_vs = np.zeros_like(beta)
        self.gamma_ms = np.zeros_like(gamma)
        self.gamma_vs = np.zeros_like(gamma)

    def update_params(self, weight_grads, bias_grads, beta_grads, gamma_grads, eps=1e-8):
        learning_rate = hp.learning_rate / (1 + hp.learning_rate_decay * self.t)
        self.t += 1

        self.weight_ms = [hp.adam_beta1 * m + (1 - hp.adam_beta1) * grad
                          for m, grad in zip(self.weight_ms, weight_grads)]
        self.weight_vs = [hp.adam_beta2 * v + (1 - hp.adam_beta2) * (grad ** 2)
                          for v, grad in zip(self.weight_vs, weight_grads)]

        self.bias_ms = [hp.adam_beta1 * m + (1 - hp.adam_beta1) * grad
                        for m, grad in zip(self.bias_ms, bias_grads)]
        self.bias_vs = [hp.adam_beta2 * v + (1 - hp.adam_beta2) * (grad ** 2)
                        for v, grad in zip(self.bias_vs, bias_grads)]

        self.beta_ms = hp.adam_beta1 * self.beta_ms + (1 - hp.adam_beta1) * beta_grads
        self.beta_vs = hp.adam_beta2 * self.beta_vs + (1 - hp.adam_beta2) * (beta_grads ** 2)

        self.gamma_ms = hp.adam_beta1 * self.gamma_ms + (1 - hp.adam_beta1) * gamma_grads
        self.gamma_vs = hp.adam_beta2 * self.gamma_vs + (1 - hp.adam_beta2) * (gamma_grads ** 2)

        weight_updates = [- learning_rate *
                          (m / (1 - hp.adam_beta1**self.t)) /
                          (np.sqrt(v / (1 - hp.adam_beta2**self.t)) + eps)
                          for m, v in zip(self.weight_ms, self.weight_vs)]
        bias_updates = [- learning_rate *
                        (m / (1 - hp.adam_beta1**self.t)) /
                        (np.sqrt(v / (1 - hp.adam_beta2**self.t)) + eps)
                        for m, v in zip(self.bias_ms, self.bias_vs)]
        beta_updates = - learning_rate * (self.beta_ms / (1 - hp.adam_beta1**self.t)) \
            / (np.sqrt(self.beta_vs / (1 - hp.adam_beta2**self.t)) + eps)
        gamma_updates = - learning_rate * (self.gamma_ms / (1 - hp.adam_beta1**self.t)) \
            / (np.sqrt(self.gamma_vs / (1 - hp.adam_beta2**self.t)) + eps)

        for weight, bias, weight_update, bias_update in zip(self.weights, self.bias, weight_updates, bias_updates):
            weight += weight_update
            bias += bias_update
        self.beta += beta_updates
        self.gamma += gamma_updates

        return self.weights, self.bias, self.beta, self.gamma


class BFGSOptimizer(object):
    pass


OPTIMIZERS = {'sgd': SGDOptimizer, 'adam': AdamOptimizer, 'bfgs': BFGSOptimizer}
