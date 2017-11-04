# -*- coding: utf-8 -*-

from config import hp
from config import bool_
from config import mpi

from os import path
from os import mkdir
from time import time
import numpy as np
from mpi4py import MPI
import dill

from layer import FullyConnectedLayer, ActivationLayer
from optimizer import OPTIMIZERS
from validation import VALIDATIONS
from util import mpiprint
from util import mpiwrite
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
        self._validation = VALIDATIONS[hp.validation]
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

    if hp.optimizer in ['sgd', 'adam']:
        def fit(self, dataset, progress):
            mpiprint('Training start!')
            nsample = dataset.nsample
            mpiprint('nsample: {}'.format(nsample))
            mpiwrite(progress, 'nsample: {}\n'.format(nsample))

            if bool_.VALIDATION:
                final_result = []
                for i, (training_indices, validation_indices) in enumerate(self._validation(nsample)):
                    mpiprint('Validation iteration: {}'.format(i+1))
                    mpiwrite(progress, '\nValidation iteration: {}\nepochs tr_RMSE tr_dRMSE tr_tRMSE val_RMSE val_dRMSE val_tRMSE\n'.format(i+1))
                    for m, elapsed, result in self._gradient_descent(dataset, training_indices, validation_indices):
                        mpiwrite(progress, '{} {} {} {} {} {} {}\n'.format(m+1, *result))
                    mpiprint('\tElapsed time: {}\n\tFinal RMSE: {} {} {} {} {} {}'.format(elapsed, *result))
                    final_result.append(result)
                    self.clear()
                final_result = np.c_[final_result].mean(axis=0)
                mpiprint('Validation result: {} {} {} {} {} {}'.format(*final_result))
                mpiwrite(progress, 'VALIDATION RESULT\n{} {} {} {} {} {}\n'.format(*final_result))

            if bool_.SAVE_MODEL:
                mpiprint('Final training start with full dataset')
                mpiwrite(progress, '\nFinal training\nepochs RMSE dRMSE tRMSE\n')
                for m, elapsed, result in self._gradient_descent(dataset):
                    mpiwrite(progress, '{} {} {} {}\n'.format(m+1, *result))
                mpiprint('\tElapsed time: {}\n\tFinal RMSE: {} {} {}'.format(elapsed, *result))

    elif hp.optimizer in ['bfgs', 'cg', 'cg-bfgs']:
        def fit(self, dataset, progress):
            mpiprint('Training start!')
            nsample = dataset.nsample
            mpiprint('nsample: {}'.format(nsample))
            mpiwrite(progress, 'nsample: {}\n'.format(nsample))

            if bool_.VALIDATION:
                final_result = []
                for i, (training_indices, validation_indices) in enumerate(self._validation(nsample)):
                    mpiprint('Validation iteration: {}'.format(i+1))
                    mpiwrite(progress, '\nValidation iteration: {}\nepochs tr_RMSE tr_dRMSE tr_tRMSE val_RMSE val_dRMSE val_tRMSE\n'.format(i+1))
                    elapsed, result = self._quasi_newton(dataset, progress, training_indices, validation_indices)
                    mpiprint('\tElapsed time: {}\n\tFinal RMSE: {} {} {} {} {} {}'.format(elapsed, *result))
                    final_result.append(result)
                    self.clear()
                final_result = np.c_[final_result].mean(axis=0)
                mpiprint('Validation result: {} {} {} {} {} {}'.format(*final_result))
                mpiwrite(progress, 'VALIDATION RESULT\n{} {} {} {} {} {}\n'.format(*final_result))

            if bool_.SAVE_MODEL:
                mpiprint('Final training start with full dataset')
                mpiwrite(progress, '\nFinal training\nepochs RMSE dRMSE tRMSE\n')
                elapsed, result = self._quasi_newton(dataset, progress)
                mpiprint('\tElapsed time: {}\n\tFinal RMSE: {} {} {}'.format(elapsed, *result))

    def _gradient_descent(self, dataset, training_indices=None, validation_indices=None):
        start = time()
        input = dataset.input
        label = dataset.label
        dinput = dataset.dinput
        dlabel = dataset.dlabel
        nsample = dataset.nsample if training_indices is None else len(training_indices)
        nderivative = dataset.nderivative
        if hp.batch_size < 0 or hp.batch_size > nsample:
            batch_size = nsample
        else:
            batch_size = hp.batch_size
        niter = -(- nsample / batch_size)

        if training_indices is None:
            self.output = np.empty((hp.nepoch+1,) + label.shape)
            self.doutput = np.empty((hp.nepoch+1,) + dlabel.shape)
            self.output[0] = label
            self.doutput[0] = dlabel
            for m in xrange(hp.nepoch):
                for _ in xrange(niter):
                    sampling = np.random.randint(0, nsample, batch_size)
                    output, doutput = self.feedforward(input[sampling], dinput[sampling], batch_size, nderivative)
                    output_error = output - label[sampling]
                    doutput_error = doutput - dlabel[sampling]
                    self.backprop(output_error, doutput_error, batch_size, nderivative)
                    self._optimizer.update_params(self.grads)
                    self.params = self._optimizer.params
                result = self.evaluate(dataset, np.arange(nsample))
                self.output[m+1] = result[0]
                self.doutput[m+1] = result[1]
                yield m, time()-start, result[2:]

        else:
            tr_results = []
            for m in xrange(hp.nepoch):
                for _ in xrange(niter):
                    sampling = training_indices[np.random.randint(0, nsample, batch_size)]
                    output, doutput = self.feedforward(input[sampling], dinput[sampling], batch_size, nderivative)
                    output_error = output - label[sampling]
                    doutput_error = doutput - dlabel[sampling]
                    self.backprop(output_error, doutput_error, batch_size, nderivative)
                    self._optimizer.update_params(self.grads)
                    self.params = self._optimizer.params
                result = np.array(self.evaluate(dataset, training_indices)[2:]
                                  + self.evaluate(dataset, validation_indices)[2:])
                tr_results.append(result[2])
                if bool_.EARLY_STOPPING and not self._check_early_stopping(tr_results, result[5]):
                    mpiprint('!!! EARLY STOPPING by epochs: {} !!!'.format(m+1))
                    break
                yield m, time()-start, result

    def _quasi_newton(self, dataset, progress, training_indices=None, validation_indices=None):
        start = time()
        if training_indices is None:
            training_indices = np.arange(dataset.nsample)
            self._optimizer.update_params(progress, self, dataset, training_indices)
            result = self.evaluate(dataset, training_indices)
            self.output = np.c_[[dataset.label, result[0]]]
            self.doutput = np.c_[[dataset.dlabel, result[1]]]
            result = self.evaluate(dataset, training_indices)[2:]
            return time()-start, result
        else:
            self._optimizer.update_params(progress, self, dataset, training_indices)
            result = np.array(self.evaluate(dataset, training_indices)[2:]
                              + self.evaluate(dataset, validation_indices)[2:])
            return time()-start, result

    def evaluate(self, dataset, indices):
        input = dataset.input[indices]
        label = dataset.label[indices]
        dinput = dataset.dinput[indices]
        dlabel = dataset.dlabel[indices]
        nsample = len(indices)
        nderivative = dataset.nderivative
        output, doutput = self.feedforward(input, dinput, nsample, nderivative)

        RMSE = rmse(output, label)
        dRMSE = rmse(doutput, dlabel)
        total_RMSE = (1 - hp.mixing_beta) * RMSE + hp.mixing_beta * dRMSE
        return output, doutput, RMSE, dRMSE, total_RMSE

    def _check_early_stopping(self, tr_results, val_result, k=10):
        if len(tr_results) < k or self._best_result > val_result:
            self._best_result = val_result
            self._best_params = self.params
            return True
        elif hp.early_stopping['criterion'] == 'GL':
            generalization_loss = 100. * (val_result / self._best_result - 1.)
            if generalization_loss < hp.early_stopping['threshold']:
                return True
            else:
                self.params = self._best_params
                return False
        elif hp.early_stopping['criterion'] == 'PQ':
            tr_results_k = tr_results[-k:]
            generalization_loss = 100. * (val_result / self._best_result - 1.)
            progress = 1000. * (sum(tr_results_k) / (k * min(tr_results_k)) - 1.)
            if generalization_loss / progress < hp.early_stopping['threshold']:
                return True
            else:
                self.params = self._best_params
                return False

    def save(self, save_dir, output_file):
        if bool_.SAVE_MODEL:
            with open(path.join(save_dir, 'optimizer.dill'), 'w') as f:
                dill.dump(self._optimizer, f)
            with open(path.join(save_dir, 'layers.dill'), 'w') as f:
                dill.dump(self._layers, f)
            np.savez(output_file, output=self.output, doutput=self.doutput)

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

    def clear(self):
        self._layers = [layer.clear() for layer in self._layers]
        self._optimizer = OPTIMIZERS[hp.optimizer](self.params)


class HDNNP(SingleNNP):
    def __init__(self, natom, ninput, composition):
        self._validation = VALIDATIONS[hp.validation]
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

    def _gradient_descent(self, dataset, training_indices=None, validation_indices=None):
        start = time()
        input = dataset.input
        label = dataset.label
        dinput = dataset.dinput
        dlabel = dataset.dlabel
        nsample = dataset.nsample if training_indices is None else len(training_indices)
        nderivative = dataset.nderivative
        if hp.batch_size < 0 or hp.batch_size > nsample:
            batch_size = nsample
        else:
            batch_size = hp.batch_size
        niter = -(- nsample / batch_size)

        if training_indices is None:
            self.output = np.empty((hp.nepoch+1,) + label.shape)
            self.doutput = np.empty((hp.nepoch+1,) + dlabel.shape)
            self.output[0] = label
            self.doutput[0] = dlabel
            for m in xrange(hp.nepoch):
                for _ in xrange(niter):
                    sampling = np.random.randint(0, nsample, batch_size)
                    self._all_comm.Bcast(sampling, root=0)
                    output, doutput = self.feedforward(input[sampling], dinput[sampling], batch_size, nderivative)
                    output_error = output - label[sampling]
                    doutput_error = doutput - dlabel[sampling]
                    self.backprop(output_error, doutput_error, batch_size, nderivative)
                    self._optimizer.update_params(self.grads)
                    self.params = self._optimizer.params
                result = self.evaluate(dataset, np.arange(nsample))
                self.output[m+1] = result[0]
                self.doutput[m+1] = result[1]
                yield m, time()-start, result[2:]

        else:
            tr_results = []
            self._all_comm.Bcast(training_indices, root=0)
            self._all_comm.Bcast(validation_indices, root=0)
            for m in xrange(hp.nepoch):
                for _ in xrange(niter):
                    sampling = training_indices[np.random.randint(0, nsample, batch_size)]
                    self._all_comm.Bcast(sampling, root=0)
                    output, doutput = self.feedforward(input[sampling], dinput[sampling], batch_size, nderivative)
                    output_error = output - label[sampling]
                    doutput_error = doutput - dlabel[sampling]
                    self.backprop(output_error, doutput_error, batch_size, nderivative)
                    self._optimizer.update_params(self.grads)
                    self.params = self._optimizer.params
                result = np.array(self.evaluate(dataset, training_indices)[2:]
                                  + self.evaluate(dataset, validation_indices)[2:])
                tr_results.append(result[2])
                if bool_.EARLY_STOPPING and not self._check_early_stopping(tr_results, result[5]):
                    mpiprint('!!! EARLY STOPPING by epochs: {} !!!'.format(m+1))
                    break
                yield m, time()-start, result

    def _quasi_newton(self, dataset, progress, training_indices=None, validation_indices=None):
        start = time()
        if training_indices is None:
            training_indices = np.arange(dataset.nsample)
            self._optimizer.update_params(progress, self, dataset, training_indices)
            result = self.evaluate(dataset, training_indices)
            self.output = np.c_[[dataset.label, result[0]]]
            self.doutput = np.c_[[dataset.dlabel, result[1]]]
            result = self.evaluate(dataset, training_indices)[2:]
            return time()-start, result
        else:
            self._all_comm.Bcast(training_indices, root=0)
            self._all_comm.Bcast(validation_indices, root=0)
            self._optimizer.update_params(progress, self, dataset, training_indices)
            result = np.array(self.evaluate(dataset, training_indices)[2:]
                              + self.evaluate(dataset, validation_indices)[2:])
            return time()-start, result

    def save(self, save_dir, output_file):
        if bool_.SAVE_MODEL:
            save_dir = path.join(save_dir, self._symbol)
            if self._atomic_rank == 0:
                if not path.exists(save_dir):
                    mkdir(save_dir)
                with open(path.join(save_dir, 'optimizer.dill'), 'w') as f:
                    dill.dump(self._optimizer, f)
                with open(path.join(save_dir, 'layers.dill'), 'w') as f:
                    dill.dump(self._nnp[0].layers, f)
            np.savez(output_file, energy=self.output, force=self.doutput)

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

    def clear(self):
        self._nnp = [SingleNNP(nnp.shape[0], has_optimizer=False) for nnp in self._nnp]
        self._optimizer = OPTIMIZERS[hp.optimizer](self.params)
