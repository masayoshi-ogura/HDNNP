from config import hp

import numpy as np
from scipy.optimize import minimize
# from scipy.optimize import check_grad
# import math

from util import mpiprint
from util import mpiwrite
from util import rmse


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
        learning_rate = hp.learning_rate / (1 + hp.learning_rate_decay * self._t)
        # learning_rate = hp.learning_rate * math.exp(- hp.learning_rate_decay * self._t)
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
        learning_rate = hp.learning_rate / (1 + hp.learning_rate_decay * self._t)
        # learning_rate = hp.learning_rate * math.exp(- hp.learning_rate_decay * self._t)
        self._t += 1

        self._ms = [self._beta1 * m + (1 - self._beta1) * grad
                    for m, grad in zip(self._ms, grads)]
        self._vs = [self._beta2 * v + (1 - self._beta2) * grad**2
                    for v, grad in zip(self._vs, grads)]
        self._params = [param - learning_rate
                        * (m / (1 - self._beta1**self._t))
                        / (np.sqrt(v / (1 - self._beta2**self._t)) + eps)
                        for param, m, v in zip(self._params, self._ms, self._vs)]


class qNewtonOptimizer(object):
    def __init__(self, params):
        self._shapes = [param.shape for param in params]
        self._disps = np.add.accumulate([param.size for param in params])
        self._params = self._pack(params)

    def update_params(self, progress, nnp, dataset, training_indices):
        def loss_func(params, nnp, input, label, dinput, dlabel, nsample, nderivative):
            nnp.params = self._unpack(params)
            self.output, self.doutput = nnp.feedforward(input, dinput, nsample, nderivative)
            lf = 1./2 * ((1 - hp.mixing_beta) * ((label - self.output)**2).mean()
                         + hp.mixing_beta * ((dlabel - self.doutput)**2).mean()) \
                + hp.l1_norm * np.sum(np.absolute(params)) \
                + hp.l2_norm / 2. * np.sum(params**2)
            return lf

        def loss_grad(_, nnp, input, label, dinput, dlabel, nsample, nderivative):
            output_error = self.output - label
            doutput_error = self.doutput - dlabel
            nnp.backprop(output_error, doutput_error, nsample, nderivative)
            return self._pack(nnp.grads)

        def generate_callback(progress, nnp, dataset, training_indices):
            dict = {'iteration': 1}

            def callback(params):
                result = nnp.evaluate(dataset, training_indices)
                mpiwrite(progress, '{} {} {} {}\n'.format(dict['iteration'], *result[2:]))
                # mpiprint('Loss Func: {}'
                #          .format(1./2 * ((1 - hp.mixing_beta) * ((label - self.output)**2).mean()
                #                          + hp.mixing_beta * ((dlabel - self.doutput)**2).mean())))
                # mpiprint('check_grad: {}'
                #          .format(check_grad(loss_func, loss_grad, params, *args)))
                dict['iteration'] += 1
            return callback

        input = dataset.input[training_indices]
        label = dataset.label[training_indices]
        dinput = dataset.dinput[training_indices]
        dlabel = dataset.dlabel[training_indices]
        nsample = len(training_indices)
        nderivative = dataset.nderivative
        args = (nnp, input, label, dinput, dlabel, nsample, nderivative)
        callback_func = generate_callback(progress, nnp, dataset, training_indices)

        loss_func(self._params, *args)
        if hp.optimizer == 'bfgs':
            response = minimize(loss_func,
                                self._params,
                                method='BFGS',
                                args=args,
                                jac=loss_grad,
                                callback=callback_func,
                                options={'maxiter': hp.nepoch})
        elif hp.optimizer == 'cg':
            response = minimize(loss_func,
                                self._params,
                                method='CG',
                                args=args,
                                jac=loss_grad,
                                callback=callback_func,
                                options={'maxiter': hp.nepoch})
        elif hp.optimizer == 'cg-bfgs':
            response = minimize(loss_func,
                                self._params,
                                method='CG',
                                args=args,
                                jac=loss_grad,
                                callback=callback_func,
                                options={'maxiter': hp.nepoch})
            response = minimize(loss_func,
                                response.x,
                                method='BFGS',
                                args=args,
                                jac=loss_grad,
                                callback=callback_func,
                                options={'maxiter': hp.nepoch})
        mpiprint('Success: {}\n{}'.format(response.success, response.message))

    def _pack(self, params):
        return np.concatenate([param.flatten() for param in params])

    def _unpack(self, flatten):
        return [f.reshape(shape)
                for f, shape in zip(np.split(flatten, self._disps), self._shapes)]


OPTIMIZERS = {'sgd': SGDOptimizer, 'adam': AdamOptimizer, 'bfgs': qNewtonOptimizer, 'cg': qNewtonOptimizer, 'cg-bfgs': qNewtonOptimizer}
