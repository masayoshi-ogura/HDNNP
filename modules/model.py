# -*- coding: utf-8 -*-

from config import hp
from config import mpi

from os import path
from os import mkdir
import numpy as np
from mpi4py import MPI
import dill

from layer import FullyConnectedLayer, ActivationLayer
from optimizer import OPTIMIZERS
from util import rmse
from util import comb


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
    def __init__(self, ninput, has_optimizer=True):
        layers = [{'node': ninput}] + hp.hidden_layers + [{'node': 1}]
        self._layers = []
        for i in xrange(len(hp.hidden_layers)):
            self._layers.append(FullyConnectedLayer(layers[i]['node'], layers[i+1]['node'], final=False))
            self._layers.append(ActivationLayer(layers[i+1]['activation']))
            # self._layers.append(BatchNormalizationLayer(layers[i+1]['node'], trainable=True))
        self._layers.append(FullyConnectedLayer(layers[-2]['node'], layers[-1]['node'], final=True))
        if has_optimizer:
            self._optimizer = OPTIMIZERS[hp.optimizer](self.params)

    @property
    def shape(self):
        shape = [layer.ninput for layer in self._layers if layer.__class__ == FullyConnectedLayer] + [1]
        return shape

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

    def feedforward(self, input, dinput, *_):
        for layer in self._layers:
            output, doutput = layer.feedforward(input, dinput)
            input, dinput = output, doutput
        return output, doutput

    def backprop(self, output_error, doutput_error, batch_size, nderivative):
        output_error = (1 - hp.mixing_beta) * output_error / batch_size
        doutput_error1 = doutput_error * hp.mixing_beta / (batch_size * nderivative)
        doutput_error2 = np.zeros_like(doutput_error)
        for layer in reversed(self._layers):
            input_error, dinput_error1, dinput_error2 = layer.backprop(output_error, doutput_error1, doutput_error2)
            output_error, doutput_error1, doutput_error2 = input_error, dinput_error1, dinput_error2

    def fit(self, training_data, validation_data):
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
                    output, doutput = self.feedforward(input[sampling], dinput[sampling])
                    output_error = output - label[sampling]
                    doutput_error = doutput - dlabel[sampling]
                    self.backprop(output_error, doutput_error, batch_size, nderivative)
                    self._optimizer.update_params(self.grads)
                    self.params = self._optimizer.params
                yield m, self.evaluate(m, training_data), self.evaluate(m, validation_data)
        elif hp.optimizer in ['bfgs', 'cg', 'cg-bfgs']:
            self._optimizer.update_params(self, input, label, dinput, dlabel, nsample, nderivative)
            yield 0, self.evaluate(0, training_data), self.evaluate(0, validation_data)

    def evaluate(self, ite, dataset):
        nsample = dataset.nsample
        nderivative = dataset.nderivative
        input = dataset.input
        label = dataset.label
        dinput = dataset.dinput
        dlabel = dataset.dlabel
        output, doutput = self.feedforward(input, dinput, nsample, nderivative)

        RMSE = rmse(output, label)
        dRMSE = rmse(doutput, dlabel)
        total_RMSE = (1 - hp.mixing_beta) * RMSE + hp.mixing_beta * dRMSE
        return output, doutput, RMSE, dRMSE, total_RMSE

    def save(self, save_dir):
        with open(path.join(save_dir, 'optimizer.dill'), 'w') as f:
            dill.dump(self._optimizer, f)
        with open(path.join(save_dir, 'layers.dill'), 'w') as f:
            dill.dump(self._layers, f)

    def load(self, save_dir):
        optimizer_file = path.join(save_dir, 'optimizer.dill')
        layer_file = path.join(save_dir, 'layers.dill')
        if path.exists(optimizer_file) and path.exists(layer_file):
            with open(optimizer_file) as f:
                self._optimizer = dill.load(f)
            with open(layer_file) as f:
                self._layers = dill.load(f)
            return True
        return False


class HDNNP(SingleNNP):
    def __init__(self, natom, ninput, composition):
        self._active = self._allocate(natom, ninput, composition)

    @property
    def active(self):
        return self._active

    if hp.optimizer in ['sgd', 'adam']:
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
    elif hp.optimizer in ['bfgs', 'cg', 'cg-bfgs']:
        @property
        def params(self):
            params = [param for layer in self._nnp[0].layers for param in layer.parameter]
            if self._atomic_rank == 0:
                cat_params = self._root_comm.allreduce(params, op=MPI.SUM)
            else:
                cat_params = None
            cat_params = self._atomic_comm.bcast(cat_params, root=0)
            return cat_params

        @params.setter
        def params(self, params):
            if self._atomic_rank == 0:
                s = len(params) * self._root_rank / self._root_size
                e = len(params) * (self._root_rank + 1) / self._root_size
            else:
                s, e = None, None
            (s, e) = self._atomic_comm.bcast((s, e), root=0)
            params = params[s:e]
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
                self._atomic_comm.Allreduce(send, recv, op=MPI.SUM)

            if self._atomic_rank == 0:
                cat_grads = self._root_comm.allreduce(grads, op=MPI.SUM)
            else:
                cat_grads = None
            cat_grads = self._atomic_comm.bcast(cat_grads, root=0)
            return cat_grads

    def feedforward(self, input, dinput, batch_size, nderivative):
        output = np.zeros((batch_size, 1))
        doutput = np.zeros((batch_size, nderivative, 1))
        output_send = np.zeros((batch_size, 1))
        doutput_send = np.zeros((batch_size, nderivative, 1))
        for nnp, i in zip(self._nnp, self._index):
            o, do = nnp.feedforward(input[:, i, :], dinput[:, i, :, :])
            output_send += o
            doutput_send += do
        self._all_comm.Allreduce(output_send, output, op=MPI.SUM)
        self._all_comm.Allreduce(doutput_send, doutput, op=MPI.SUM)
        doutput *= -1.
        return output, doutput

    def backprop(self, output_error, doutput_error, batch_size, nderivative):
        doutput_error *= -1.
        for nnp in self._nnp:
            nnp.backprop(output_error, doutput_error, batch_size, nderivative)

    def fit(self, training_data, validation_data):
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
                    self._all_comm.Bcast(sampling, root=0)
                    output, doutput = self.feedforward(input[sampling], dinput[sampling], batch_size, nderivative)
                    output_error = output - label[sampling]
                    doutput_error = doutput - dlabel[sampling]
                    self.backprop(output_error, doutput_error, batch_size, nderivative)
                    self._optimizer.update_params(self.grads)
                    self.params = self._optimizer.params
                yield m, self.evaluate(m, training_data), self.evaluate(m, validation_data)
        elif hp.optimizer in ['bfgs', 'cg', 'cg-bfgs']:
            self._optimizer.update_params(self, input, label, dinput, dlabel, nsample, nderivative)
            yield 0, self.evaluate(0, training_data), self.evaluate(0, validation_data)

    def save(self, save_dir):
        save_dir = path.join(save_dir, self._symbol)
        if self._atomic_rank == 0:
            if not path.exists(save_dir):
                mkdir(save_dir)
            with open(path.join(save_dir, 'optimizer.dill'), 'w') as f:
                dill.dump(self._optimizer, f)
            with open(path.join(save_dir, 'layers.dill'), 'w') as f:
                dill.dump(self._nnp[0].layers, f)

    def load(self, save_dir):
        optimizer_file = path.join(save_dir, self._symbol, 'optimizer.dill')
        layer_file = path.join(save_dir, self._symbol, 'layers.dill')
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
        root_nodes = []
        for symbol, num in w.items():
            root_nodes.append(low)
            if low <= self._all_rank < low+num:
                self._symbol = symbol
                self._atomic_comm = self._all_comm.Create(self._all_comm.Get_group().Incl(xrange(low, low+num)))
                self._atomic_rank = self._atomic_comm.Get_rank()
            low += num
        self._root_comm = self._all_comm.Create(self._all_comm.Get_group().Incl(root_nodes))
        if self._atomic_rank == 0:
            self._root_rank = self._root_comm.Get_rank()
            self._root_size = self._root_comm.Get_size()

        quo, rem = composition['number'][self._symbol] / w[self._symbol], composition['number'][self._symbol] % w[self._symbol]
        if self._atomic_rank < rem:
            self._nnp = [SingleNNP(ninput, has_optimizer=False) for _ in xrange(quo+1)]
            self._index = list(composition['index'][self._symbol])[self._atomic_rank*(quo+1): (self._atomic_rank+1)*(quo+1)]
        else:
            self._nnp = [SingleNNP(ninput, has_optimizer=False) for _ in xrange(quo)]
            self._index = list(composition['index'][self._symbol])[self._atomic_rank*quo+rem: (self._atomic_rank+1)*quo+rem]

        self.sync()
        self._optimizer = OPTIMIZERS[hp.optimizer](self.params)
        return True

    def sync(self):
        params = self._atomic_comm.bcast(self.params, root=0)
        self.params = params
