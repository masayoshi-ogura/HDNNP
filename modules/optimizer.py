import numpy as np
from scipy.optimize import minimize
import math

from config import hp


class SGDOptimizer(object):
    def __init__(self, params, nesterov=True, momentum=0.9):
        self._params = params
        self._t = 0
        self._nesterov = nesterov
        self._momentum = momentum
        if nesterov:
            self._velocities = [np.zeros_like(param) for param in params]

    @property
    def params(self):
        return self._params

    def update_params(self, grads):
        # learning_rate = hp.learning_rate / (1 + hp.learning_rate_decay * self._t)
        learning_rate = hp.learning_rate * math.exp(- hp.learning_rate_decay * self._t)
        self._t += 1

        if self._nesterov:
            updates = [self._momentum * velocity - learning_rate * grad
                       for velocity, grad in zip(self._velocities, grads)]
            self._velocities = updates
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
    def __init__(self, params):
        self._shapes = [param.shape for param in params]
        self._disps = np.add.accumulate([param.size for param in params])
        self._params = self._pack(params)

    @property
    def params(self):
        return self._unpack(self._params)

    def update_params(self, args):
        def loss_func(params, nnp, input, label, dinput, dlabel, nsample, nderivative):
            nnp.params = self._unpack(params)
            output, doutput = nnp.feedforward(input, dinput, nsample, None, 'training')
            return 1./2 * (np.mean((label - output)**2) + hp.mixing_beta * np.mean((dlabel - doutput)**2))

        def loss_grad(params, nnp, input, label, dinput, dlabel, nsample, nderivative):
            nnp.params = self._unpack(params)
            output, doutput = nnp.feedforward(input, dinput, nsample, None, 'training')
            output_error = output - label
            doutput_error = doutput - dlabel
            nnp.backprop(output_error, doutput_error, nsample, nderivative)
            return self._pack(nnp.grads)

        response = minimize(loss_func,
                            self._params,
                            method='BFGS',
                            args=args,
                            jac=loss_grad,
                            options={'disp': False, 'maxiter': 100})
        self._params = response.x

    def _pack(self, params):
        return np.concatenate([param.flatten() for param in params])

    def _unpack(self, flatten):
        return [f.reshape(shape)
                for f, shape in zip(np.split(flatten, self._disps), self._shapes)]


OPTIMIZERS = {'sgd': SGDOptimizer, 'adam': AdamOptimizer, 'bfgs': BFGSOptimizer}
