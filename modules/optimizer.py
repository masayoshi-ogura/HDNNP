import numpy as np
from scipy.optimize import minimize, check_grad
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


class qNewtonOptimizer(object):
    def __init__(self, params):
        self._shapes = [param.shape for param in params]
        self._disps = np.add.accumulate([param.size for param in params])
        self._params = self._pack(params)

    def update_params(self, *args):
        def loss_func(params, nnp, input, label, dinput, dlabel, nsample, nderivative):
            nnp.params = self._unpack(params)
            self.output, self.doutput = nnp.feedforward(input, dinput, nsample, nderivative)
            return 1./2 * (((label - self.output)**2).mean() + hp.mixing_beta * ((dlabel - self.doutput)**2).mean())

        def loss_grad(_, nnp, input, label, dinput, dlabel, nsample, nderivative):
            output_error = self.output - label
            doutput_error = self.doutput - dlabel
            nnp.backprop(output_error, doutput_error, nsample, nderivative)
            return self._pack(nnp.grads)

        def generate_callback(label, dlabel):
            result = {'iteration': 0, 'output': [], 'doutput': []}

            def callback(params):
                print 'Iteration: {}'.format(result['iteration'])
                print 'RMSE: {}'.format(np.sqrt(((label - self.output)**2).mean()) + hp.mixing_beta * np.sqrt(((dlabel - self.doutput)**2).mean()))
                # print 'Loss Func: {}'.format(1./2 * (((label - self.output)**2).mean() + hp.mixing_beta * ((dlabel - self.doutput)**2).mean()))
                # print 'check_grad: {}'.format(check_grad(loss_func, loss_grad, params, *args))
                result['iteration'] += 1
                # result['output'].append(self.output)
                # result['doutput'].append(self.doutput)
            return result, callback

        loss_func(self._params, *args)
        result, callback_func = generate_callback(args[2], args[4])
        if hp.optimizer == 'bfgs':
            minimize(loss_func,
                     self._params,
                     method='BFGS',
                     args=args,
                     jac=loss_grad,
                     callback=callback_func,
                     options={'disp': True, 'maxiter': hp.nepoch})
        elif hp.optimizer == 'cg':
            minimize(loss_func,
                     self._params,
                     method='CG',
                     args=args,
                     jac=loss_grad,
                     callback=callback_func,
                     options={'disp': True, 'maxiter': hp.nepoch})
        elif hp.optimizer == 'cg-bfgs':
            response = minimize(loss_func,
                                self._params,
                                method='CG',
                                args=args,
                                jac=loss_grad,
                                callback=callback_func,
                                options={'disp': True, 'maxiter': hp.nepoch})
            minimize(loss_func,
                     response.x,
                     method='BFGS',
                     args=args,
                     jac=loss_grad,
                     callback=callback_func,
                     options={'disp': True, 'maxiter': hp.nepoch})

    def _pack(self, params):
        return np.concatenate([param.flatten() for param in params])

    def _unpack(self, flatten):
        return [f.reshape(shape)
                for f, shape in zip(np.split(flatten, self._disps), self._shapes)]


OPTIMIZERS = {'sgd': SGDOptimizer, 'adam': AdamOptimizer, 'bfgs': qNewtonOptimizer, 'cg': qNewtonOptimizer, 'cg-bfgs': qNewtonOptimizer}
