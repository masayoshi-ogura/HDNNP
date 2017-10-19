# -*- coding: utf-8 -*-

from config import hp
from config import file_
from config import mpi

from os import path
from os import makedirs
from itertools import combinations
import numpy as np
from mpi4py import MPI
import dill

from layer import FullyConnectedLayer, ActivationLayer, BatchNormalizationLayer
from optimizer import OPTIMIZERS


def rmse(pred, true):
    return np.sqrt(((pred - true)**2).mean())


def comb(n, r):
    for c in combinations(xrange(1, n), r-1):
        ret = []
        low = 0
        for p in c:
            ret.append(p - low)
            low = p
        ret.append(n - low)
        yield ret


def allocate(size, symbol, natom):
    min = 0
    for worker in comb(size, len(symbol)):
        obj = 0
        for w, n in zip(worker, natom):
            if w > n:
                break
            obj += n*(-(-n/w))**2
        else:
            if min == 0 or min > obj:
                min = obj
                min_worker = {symbol[i]: worker[i] for i in xrange(len(symbol))}  # worker(node)
    return min_worker


class SingleNNP(object):
    def __init__(self, ninput, high_dimension=False):
        layers = [{'node': ninput}] + hp.hidden_layers + [{'node': 1}]
        self._layers = []
        for i in xrange(len(hp.hidden_layers)):
            self._layers.append(FullyConnectedLayer(layers[i]['node'], layers[i+1]['node']))
            self._layers.append(ActivationLayer(layers[i+1]['activation']))
            # self._layers.append(BatchNormalizationLayer(layers[i+1]['node'], trainable=True))
        self._layers.append(FullyConnectedLayer(layers[-2]['node'], layers[-1]['node']))
        if not high_dimension:
            self._optimizer = OPTIMIZERS[hp.optimizer](self.params)

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, layers):
        self._layers = layers

    @property
    def params(self):
        return [param for layer in self._layers for param in layer.parameter]

    @params.setter
    def params(self, params):
        i = 0
        for layer in self._layers:
            num = len(layer.parameter)
            layer.parameter = tuple(params[i:i+num])
            i += num

    @property
    def grads(self):
        return [grad for layer in self._layers for grad in layer.gradient]

    def feedforward(self, input, dinput, batch_size, _, mode):
        for layer in self._layers:
            output, doutput = layer.feedforward(input, dinput, batch_size, mode)
            input, dinput = output, doutput
        return output, doutput

    def backprop(self, output_error, doutput_error, batch_size, nderivative):
        for layer in reversed(self._layers):
            input_error, dinput_error = layer.backprop(output_error, doutput_error, batch_size, nderivative)
            output_error, doutput_error = input_error, dinput_error

    def fit(self, training_data, validation_data, training_animator, validation_animator):
        nsample = training_data.nsample
        nderivative = training_data.nderivative
        input = training_data.input
        label = training_data.label
        dinput = training_data.dinput
        dlabel = training_data.dlabel
        if hp.optimizer in ['sgd', 'adam']:
            for m in xrange(hp.nepoch):
                batch_size = int(hp.batch_size * (1 + hp.batch_size_growth * m))
                if batch_size < 0 or batch_size > nsample:
                    batch_size = nsample

                niter = -(- nsample / batch_size)
                for i in xrange(niter):
                    sampling = np.random.randint(0, nsample, batch_size)
                    output, doutput = self.feedforward(input[sampling], dinput[sampling], batch_size, None, 'training')
                    output_error = output - label[sampling]
                    doutput_error = doutput - dlabel[sampling]
                    self.backprop(output_error, doutput_error, batch_size, nderivative)
                    self._optimizer.update_params(self.grads)
                    self.params = self._optimizer.params
                yield m, self.evaluate(m, nsample, training_data, training_animator), self.evaluate(m, nsample, validation_data, validation_animator)
        elif hp.optimizer == 'bfgs':
            for m in xrange(hp.nepoch):
                self._optimizer.update_params((self, input, label, dinput, dlabel, nsample, nderivative))
                self.params = self._optimizer.params
                yield m, self.evaluate(m, nsample, training_data, training_animator), self.evaluate(m, nsample, validation_data, validation_animator)

    def evaluate(self, ite, nsample, dataset, animator):
        nsample = dataset.nsample
        nderivative = dataset.nderivative
        input = dataset.input
        label = dataset.label
        dinput = dataset.dinput
        dlabel = dataset.dlabel
        output, doutput = self.feedforward(input, dinput, nsample, nderivative, 'test')

        if animator:
            if ite == 0:
                animator.true = (label, dlabel)
            animator.preds = (output, doutput)

        RMSE = rmse(output, label)
        dRMSE = rmse(doutput, dlabel)
        total_RMSE = RMSE + hp.mixing_beta * dRMSE
        return RMSE, dRMSE, total_RMSE

    def save(self, subdir):
        save_dir = path.join(file_.save_dir, subdir)
        makedirs(save_dir)
        with open(path.join(save_dir, 'optimizer.dill'), 'w') as f:
            dill.dump(self._optimizer, f)
        with open(path.join(save_dir, 'layers.dill'), 'w') as f:
            dill.dump(self._layers, f)

    def load(self, subdir):
        load_dir = path.join(file_.save_dir, subdir)
        if path.exists(load_dir):
            with open(path.join(load_dir, 'optimizer.dill'), 'r') as f:
                self._optimizer = dill.load(f)
            with open(path.join(load_dir, 'layers.dill'), 'r') as f:
                self._layers = dill.load(f)
        else:
            print 'pretrained data directory {} is not found. Initialized parameters will be used.'.format(load_dir)


class HDNNP(SingleNNP):
    def __init__(self, natom, ninput, composition):
        self._active = self._allocate(natom, ninput, composition)

    @property
    def active(self):
        return self._active

    @property
    def params(self):
        return [param for layer in self._nnp[0].layers for param in layer.parameter]

    @params.setter
    def params(self, params):
        for nnp in self._nnp:
            nnp.params = params

    @property
    def grads(self):
        grads_send = [np.zeros_like(param) for layer in self._nnp[0].layers for param in layer.parameter]
        grads = [np.zeros_like(param) for layer in self._nnp[0].layers for param in layer.parameter]
        for nnp in self._nnp:
            for send, grad in zip(grads_send, nnp.grads):
                send += grad
        for send, recv in zip(grads_send, grads):
            self._all_comm.Allreduce(send, recv, op=MPI.SUM)
        return grads

    def feedforward(self, input, dinput, batch_size, nderivative, mode):
        output = np.zeros((batch_size, 1))
        doutput = np.zeros((batch_size, nderivative, 1))
        output_send = np.zeros((batch_size, 1))
        doutput_send = np.zeros((batch_size, nderivative, 1))
        for nnp, i in zip(self._nnp, self._index):
            o, do = nnp.feedforward(input[:, i, :], dinput[:, i, :, :], batch_size, None, mode)
            output_send += o
            doutput_send += do
        self._all_comm.Allreduce(output_send, output, op=MPI.SUM)
        self._all_comm.Allreduce(doutput_send, doutput, op=MPI.SUM)
        doutput *= -1.
        return output, doutput

    def fit(self, training_data, validation_data, training_animator=None, validation_animator=None):
        nsample = training_data.nsample
        nderivative = training_data.nderivative
        input = training_data.input
        label = training_data.label
        dinput = training_data.dinput
        dlabel = training_data.dlabel
        for m in xrange(hp.nepoch):
            batch_size = int(hp.batch_size * (1 + hp.batch_size_growth * m))
            if batch_size < 0 or batch_size > nsample:
                batch_size = nsample

            niter = -(- nsample / batch_size)
            for i in xrange(niter):
                sampling = np.random.randint(0, nsample, batch_size)
                self._all_comm.Bcast(sampling, root=0)
                output, doutput = self.feedforward(input[sampling], dinput[sampling], batch_size, nderivative, 'training')
                output_error = output - label[sampling]
                doutput_error = - (doutput - dlabel[sampling])
                for nnp in self._nnp:
                    nnp.backprop(output_error, doutput_error, batch_size, nderivative)
                self._optimizer.update_params(self.grads)
                self.params = self._optimizer.params
            yield m, self.evaluate(m, nsample, training_data, training_animator), self.evaluate(m, nsample, validation_data, validation_animator)

    def save(self, subdir):
        save_dir = path.join(file_.save_dir, subdir)
        if self._all_rank == 0 and not path.exists(save_dir):
            makedirs(save_dir)
        self._all_comm.Barrier()
        if self._atomic_rank == 0:
            with open(path.join(save_dir, '{}_optimizer.dill'.format(self._symbol)), 'w') as f:
                dill.dump(self._optimizer, f)
            with open(path.join(save_dir, '{}_layers.dill'.format(self._symbol)), 'w') as f:
                dill.dump(self._nnp[0].layers, f)

    def load(self, subdir):
        optimizer_file = path.join(file_.save_dir, subdir, '{}_optimizer.dill'.format(self._symbol))
        layer_file = path.join(file_.save_dir, subdir, '{}_layers.dill'.format(self._symbol))
        if path.exists(optimizer_file) and path.exists(layer_file):
            with open(optimizer_file) as f:
                self._optimizer = dill.load(f)
            for nnp in self._nnp:
                with open(layer_file) as f:
                    nnp.layers = dill.load(f)

    def _allocate(self, natom, ninput, composition):
        s = composition['number'].keys()  # symbol list
        n = composition['number'].values()  # natom list

        # allocate worker for each atom
        if len(s) > mpi.size:
            raise ValueError('the number of process must be {} or more.'.format(len(s)))
        elif mpi.size > natom:
            self._all_comm = mpi.comm.Create(mpi.comm.Get_group().Incl(xrange(natom)))
            if not mpi.rank < natom:
                return False
            self._all_rank = self._all_comm.Get_rank()
            w = composition['number']  # worker(node) ex.) {'Si': 3, 'Ge': 5}
        else:
            self._all_comm = mpi.comm
            self._all_rank = mpi.rank
            w = allocate(mpi.size, s, n)

        # split MPI communicator and set SingleNNP instances and initialize them
        low = 0
        for symbol, num in w.items():
            if low <= self._all_rank < low+num:
                self._symbol = symbol
                self._atomic_comm = self._all_comm.Create(self._all_comm.Get_group().Incl(xrange(low, low+num)))
                self._atomic_rank = self._atomic_comm.Get_rank()
            low += num
        quo, rem = composition['number'][self._symbol] / w[self._symbol], composition['number'][self._symbol] % w[self._symbol]
        if self._atomic_rank < rem:
            self._nnp = [SingleNNP(ninput, high_dimension=True) for _ in xrange(quo+1)]
            self._index = list(composition['index'][self._symbol])[self._atomic_rank*(quo+1): (self._atomic_rank+1)*(quo+1)]
        else:
            self._nnp = [SingleNNP(ninput, high_dimension=True) for _ in xrange(quo)]
            self._index = list(composition['index'][self._symbol])[self._atomic_rank*quo+rem: (self._atomic_rank+1)*quo+rem]

        self._sync()
        self._optimizer = OPTIMIZERS[hp.optimizer](self.params)
        return True

    def _sync(self):
        self._nnp[0].layers = self._atomic_comm.bcast(self._nnp[0].layers, root=0)
        for nnp in self._nnp:
            nnp.layers = self._nnp[0].layers
