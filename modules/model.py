# -*- coding: utf-8 -*-

from os import path
from os import makedirs
from itertools import combinations
import numpy as np
from mpi4py import MPI
import dill

from config import hp
from config import file_
from layer import FullyConnectedLayer, ActivationLayer, BatchNormalizationLayer
from optimizer import OPTIMIZERS


def rmse(pred, true):
    return np.sqrt(((pred - true)**2).mean())


class SingleNNP(object):
    def __init__(self, ninput):
        layers = [{'node': ninput}] + hp.hidden_layers + [{'node': 1}]
        self._layers = []
        append = self._layers.append
        for i in range(len(hp.hidden_layers)):
            append(FullyConnectedLayer(layers[i]['node'], layers[i+1]['node']))
            append(BatchNormalizationLayer(layers[i+1]['node']))
            append(ActivationLayer(layers[i+1]['activation']))
        append(FullyConnectedLayer(layers[-2]['node'], layers[-1]['node']))
        self._optimizer = OPTIMIZERS[hp.optimizer](self.params)

    @property
    def params(self):
        return [param for layer in self._layers for param in layer.parameter]

    @params.setter
    def params(self, params):
        i = 0
        for layer in self._layers:
            num = len(layer.parameter)
            layer.parameter = tuple(params[i+j] for j in range(num))
            i += num

    def _feedforward(self, input, dinput, batch_size, mode):
        for layer in self._layers:
            output, doutput = layer.feedforward(input, dinput, batch_size, mode)
            input, dinput = output, doutput
        return output, doutput

    def _backprop(self, output_error, doutput_error, batch_size, d_size):
        # d_size is the number of differentiation
        for layer in reversed(self._layers):
            input_error, dinput_error = layer.backprop(output_error, doutput_error, d_size)
            output_error, doutput_error = input_error, dinput_error
        return [grad/batch_size for layer in self._layers for grad in layer.gradient]

    def fit(self, training_data, validation_data, training_animator, validation_animator):
        input, label, dinput, dlabel = training_data
        nsample, d_size, _ = dinput.shape
        for m in range(hp.nepoch):
            batch_size = int(hp.batch_size * (1 + hp.batch_size_growth * m))
            if batch_size < 0 or batch_size > nsample:
                batch_size = nsample

            niter = -(- nsample / batch_size)
            for i in range(niter):
                sampling = np.random.randint(0, nsample, batch_size)
                output, doutput = self._feedforward(input[sampling], dinput[sampling], batch_size, 'training')
                # Ei = output
                # Fi = - doutput
                output_error = output - label[sampling]
                doutput_error = doutput - dlabel[sampling]
                # E_error = output_error
                # F_error = - doutput_error
                grads = self._backprop(output_error, doutput_error, batch_size, d_size)
                self._optimizer.update_params(grads)
                self.params = self._optimizer.params
            yield m, self.evaluate(m, nsample, training_data, training_animator), self.evaluate(m, nsample, validation_data, validation_animator)

    def evaluate(self, ite, nsample, dataset, animator):
        input, label, dinput, dlabel = dataset
        output, doutput = self._feedforward(input, dinput, nsample, 'test')

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


# class HDNNP(object):
#     def __init__(self, natom, nsample, animator):
#         self.all_natom = natom
#         self.nsample = nsample
#         self.animator = animator
#
#     def inference(self, Gs, dGs, batch_size, mode):
#         E, E_tmp = np.zeros((batch_size, 1)), np.zeros((batch_size, 1))
#         F, F_tmp = np.zeros((batch_size, 3*self.all_natom, 1)), np.zeros((batch_size, 3*self.all_natom, 1))
#         for nnp, i in zip(self.nnp, self.index):
#             Ei, Fi = nnp.feedforward(Gs[:, i, :], dGs[:, i, :, :], batch_size, mode)
#             E_tmp += Ei
#             F_tmp += Fi
#         self.all_comm.Allreduce(E_tmp, E, op=MPI.SUM)
#         self.all_comm.Allreduce(F_tmp, F, op=MPI.SUM)
#         return E, F
#
#     def training(self, m, Es, Fs, Gs, dGs):
#         if hp.optimizer in ['sgd', 'adam']:
#             batch_size = int(hp.batch_size * (1 + hp.batch_size_growth * m))
#             if batch_size < 0 or batch_size > self.nsample:
#                 batch_size = self.nsample
#
#             niter = -(- self.nsample / batch_size)
#             for m in range(niter):
#                 sampling = np.random.randint(0, self.nsample, batch_size)
#                 self.all_comm.Bcast(sampling, root=0)
#
#                 weight_grads = [np.zeros_like(weight) for weight in self.nnp[0].weights]
#                 bias_grads = [np.zeros_like(bias) for bias in self.nnp[0].bias]
#                 weight_para = [np.zeros_like(weight) for weight in self.nnp[0].weights]
#                 bias_para = [np.zeros_like(bias) for bias in self.nnp[0].bias]
#                 E_pred, F_pred = self.inference(Gs[sampling], dGs[sampling], batch_size)
#                 E_error = E_pred - Es[sampling]
#                 F_error = F_pred - Fs[sampling]
#                 for nnp in self.nnp:
#                     weight, bias, beta, gamma = nnp.backprop(E_error, F_error, batch_size)
#                     for weight_p, bias_p, weight_t, bias_t in zip(weight_para, bias_para, weight, bias):
#                         weight_p += weight_t
#                         bias_p += bias_t
#                     beta_para += beta
#                     gamma_para += gamma
#                 for weight_p, bias_p, weight, bias in zip(weight_para, bias_para, weight_grads, bias_grads):
#                     self.atomic_comm.Allreduce(weight_p/(batch_size*self.atomic_natom), weight, op=MPI.SUM)
#                     self.atomic_comm.Allreduce(bias_p/(batch_size*self.atomic_natom), bias, op=MPI.SUM)
#                 self.atomic_comm.Allreduce(beta_para/(batch_size*self.atomic_natom), beta_grads, op=MPI.SUM)
#                 self.atomic_comm.Allreduce(gamma_para/(batch_size*self.atomic_natom), gamma_grads, op=MPI.SUM)
#
#                 weights, bias, beta, gamma = self.optimizer.update_params(weight_grads, bias_grads, beta_grads, gamma_grads)
#                 for nnp in self.nnp:
#                     nnp.weights = weights
#                     nnp.bias = bias
#         elif hp.optimizer == 'bfgs':
#             print 'Info: BFGS is off-line learning method. "batch_size" is ignored.'
#             # TODO: BFGS optimize
#             self.optimizer.update_params()
#         else:
#             raise ValueError('invalid optimizer: select sgd or adam or bfgs')
#
#     def calc_RMSE(self, m, Es, Fs, Gs, dGs):
#         if bool_.SAVE_GIF and m == 0:
#             self.animator.set_true(Es, Fs)
#         E_preds, F_preds = self.inference(Gs, dGs, self.nsample)
#
#         if bool_.SAVE_GIF:
#             self.animator.set_pred(m, E_preds, F_preds)
#         E_RMSE = rmse(E_preds, Es)
#         F_RMSE = rmse(F_preds, Fs)
#         RMSE = E_RMSE + hp.mixing_beta * F_RMSE
#         return E_RMSE, F_RMSE, RMSE
#
#     def save_fig(self, datestr, config, ext):
#         if self.all_rank == 0:
#             print 'saving figures ...'
#             self.animator.save_fig(datestr, config, ext)
#
#     def save_w(self, datestr):
#         weight_save_dir = path.join(file_.weight_dir, datestr)
#         if self.all_rank == 0 and not path.exists(weight_save_dir):
#             makedirs(weight_save_dir)
#         self.all_comm.Barrier()
#         # before saving, sync EMA
#         mu_EMA, sigma_EMA = np.zeros_like(self.nnp[0].mu_EMA), np.zeros_like(self.nnp[0].sigma_EMA)
#         mu_EMA_p, sigma_EMA_p = np.zeros_like(self.nnp[0].mu_EMA), np.zeros_like(self.nnp[0].sigma_EMA)
#         for nnp in self.nnp:
#             mu_EMA_p += nnp.mu_EMA
#             sigma_EMA_p += nnp.sigma_EMA
#         self.atomic_comm.Allreduce(mu_EMA_p, mu_EMA, op=MPI.SUM)
#         self.atomic_comm.Allreduce(sigma_EMA_p, sigma_EMA, op=MPI.SUM)
#         # save weights, bias, beta, gamma, EMA
#         if self.atomic_rank == 0:
#             dicts = {}
#             dicts.update({'weight_{}'.format(i): weight for i, weight in enumerate(self.nnp[0].weights)})
#             dicts.update({'bias_{}'.format(i): bias for i, bias in enumerate(self.nnp[0].bias)})
#             dicts.update({'beta': self.nnp[0].beta, 'gamma': self.nnp[0].gamma,
#                           'mu_EMA': mu_EMA/self.atomic_natom, 'sigma_EMA': sigma_EMA/self.atomic_natom})
#             np.savez(path.join(save_dir, '{}.npz'.format(self.symbol)), **dicts)
#             with open(path.join(save_dir, '{}_optimizer.dill'.format(self.symbol)), 'w') as f:
#                 dill.dump(self.optimizer, f)
#
#     def load(self, datestr):
#         npz_file = path.join(file_.save_dir, datestr, '{}.npz'.format(self.symbol))
#         dill_file = path.join(file_.save_dir, datestr, '{}_optimizer.dill'.format(self.symbol))
#         if path.exists(npz_file) and path.exists(dill_file):
#             data = np.load(npz_file)
#             for nnp in self.nnp:
#                 for i in range(nnp.nweight):
#                     nnp.weights[i] = data['weight_{}'.format(i)]
#                     nnp.bias[i] = data['bias_{}'.format(i)]
#                 nnp.beta = data['beta']
#                 nnp.gamma = data['gamma']
#                 nnp.mu_EMA = data['mu_EMA']
#                 nnp.sigma_EMA = data['sigma_EMA']
#             with open(dill_file, 'r') as f:
#                 self.optimizer = dill.load(f)
#         else:
#             if self.atomic_rank == 0:
#                 print 'weight params file {} is not found. use initialized parameters.'.format(weight_file)
#             pass
#
#     def sync_w(self):
#         for i in range(self.nnp[0].nweight):
#             self.atomic_comm.Bcast(self.nnp[0].weights[i], root=0)
#             self.atomic_comm.Bcast(self.nnp[0].bias[i], root=0)
#
#         for nnp in self.nnp:
#             nnp.weights = self.nnp[0].weights
#             nnp.bias = self.nnp[0].bias
#
#     def initialize(self, comm, rank, size, ninput, composition):
#         def comb(n, r):
#             for c in combinations(range(1, n), r-1):
#                 ret = []
#                 low = 0
#                 for p in c:
#                     ret.append(p - low)
#                     low = p
#                 ret.append(n - low)
#                 yield ret
#
#         def allocate(size, symbol, natom):
#             min = 0
#             for worker in comb(size, len(symbol)):
#                 obj = 0
#                 for w, n in zip(worker, natom):
#                     if w > n:
#                         break
#                     obj += n*(-(-n/w))**2
#                 else:
#                     if min == 0 or min > obj:
#                         min = obj
#                         min_worker = {symbol[i]: worker[i] for i in range(len(symbol))}  # worker(node)
#             return min_worker
#
#         s = composition['number'].keys()  # symbol list
#         n = composition['number'].values()  # natom list
#
#         # allocate worker for each atom
#         if len(s) > size:
#             raise ValueError('the number of process must be {} or more.'.format(len(s)))
#         elif size > self.all_natom:
#             self.all_comm = comm.Create(comm.Get_group().Incl(range(self.all_natom)))
#             if not rank < self.all_natom:
#                 return False
#             self.all_rank = self.all_comm.Get_rank()
#             w = composition['number']  # worker(node) ex.) {'Si': 3, 'Ge': 5}
#         else:
#             self.all_comm = comm
#             self.all_rank = rank
#             w = allocate(size, s, n)
#
#         # split MPI communicator and set SingleNNP instances and initialize them
#         low = 0
#         for symbol, num in w.items():
#             if low <= rank < low+num:
#                 self.symbol = symbol
#                 atomic_group = self.all_comm.Get_group().Incl(range(low, low+num))
#                 self.atomic_comm = self.all_comm.Create(atomic_group)
#                 self.atomic_rank = self.atomic_comm.Get_rank()
#             low += num
#         self.atomic_natom = composition['number'][self.symbol]
#         self.nnp = [SingleNNP(self.all_natom, ninput)]
#         quo, rem = self.atomic_natom / w[self.symbol], self.atomic_natom % w[self.symbol]
#         if self.atomic_rank < rem:
#             self.nnp *= quo+1
#             self.index = list(composition['index'][self.symbol])[self.atomic_rank*(quo+1): (self.atomic_rank+1)*(quo+1)]
#         else:
#             self.nnp *= quo
#             self.index = list(composition['index'][self.symbol])[self.atomic_rank*quo+rem: (self.atomic_rank+1)*quo+rem]
#
#         self.sync_w()
#         self.optimizer = OPTIMIZERS[hp.optimizer](self.nnp[0].weights, self.nnp[0].bias)
#
#         return True
