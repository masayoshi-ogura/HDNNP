# -*- coding: utf-8 -*-

from os import path
from os import makedirs
from itertools import combinations
import numpy as np
from mpi4py import MPI

from config import hp
from config import bool_
from config import file_
from activation_function import ACTIVATIONS, DERIVATIVES, SECOND_DERIVATIVES
from optimizer import OPTIMIZERS


class SingleNNP(object):
    def __init__(self, all_natom, ninput):
        self.all_natom = all_natom
        self.shape = (ninput,) + hp.hidden_layer + (1,)
        self.nlayer = len(hp.hidden_layer)
        self.nweight = len(hp.hidden_layer) + 1
        self.learning_rate = hp.learning_rate
        self.mixing_beta = hp.mixing_beta
        self.activation = ACTIVATIONS[hp.activation]
        self.deriv_activation = DERIVATIVES[hp.activation]
        self.second_deriv_activation = SECOND_DERIVATIVES[hp.activation]

        self.weights, self.bias = [], []
        for i in range(self.nweight):
            self.weights.append(np.random.normal(0.0, 0.5, (self.shape[i], self.shape[i+1])))
            self.bias.append(np.random.normal(0.0, 0.5, (self.shape[i+1])))
        # for batch normalization
        self.beta = np.zeros(ninput)
        self.gamma = np.ones(ninput)
        self.mu_EMA = np.zeros(ninput)
        self.sigma2_EMA = np.zeros(ninput)

    def feedforward(self, Gi, dGi, size, mode):
        # "inputs" means "linear transformation's input", not "each layer's input"
        # "outputs" means "linear transformation's output", not "each layer's output"
        self.inputs, self.outputs, self.deriv_inputs, self.deriv_outputs = [], [], [], []
        for i in range(self.nweight):
            if i == 0:
                Gi, dGi = self.batchnorm_forward(Gi, dGi, mode)
                self.inputs.append(Gi)
                self.deriv_inputs.append(dGi)
            else:
                self.inputs.append(self.activation(self.outputs[i-1]))
                self.deriv_inputs.append(self.deriv_activation(self.outputs[i-1])[:, None, :] * self.deriv_outputs[i-1])
            self.outputs.append(np.dot(self.inputs[i], self.weights[i]) + self.bias[i])
            self.deriv_outputs.append(np.tensordot(self.deriv_inputs[i], self.weights[i], ((2,), (0,))))
        Ei = self.outputs[-1]
        Fi = - self.deriv_outputs[-1]
        return Ei, Fi

    def batchnorm_forward(self, Gi, dGi, mode, eps=1e-5):
        if mode == 'train':
            self.mu = np.mean(Gi, axis=0)  # mean
            self.sigma2 = np.var(Gi, axis=0)  # variance
            self.norm = (Gi - self.mu) / np.sqrt(self.sigma2 + eps)  # NxD
            scaled_Gi = self.gamma * self.norm + self.beta  # NxD
            scaled_dGi = self.gamma * dGi / np.sqrt(self.sigma2 + eps)

            if (self.mu_EMA == 0.0).all():
                self.mu_EMA = self.mu
                self.sigma2_EMA = self.sigma2
            else:
                self.mu_EMA = hp.smooth_factor * self.mu + (1 - hp.smooth_factor) * self.mu_EMA
                self.sigma2_EMA = hp.smooth_factor * self.sigma2 + (1 - hp.smooth_factor) * self.sigma2_EMA
        elif mode == 'test':
            self.mu = self.mu_EMA
            self.sigma2 = self.sigma2_EMA
            self.norm = (Gi - self.mu) / np.sqrt(self.sigma2 + eps)
            scaled_Gi = self.gamma * self.norm + self.beta
            scaled_dGi = self.gamma * dGi / np.sqrt(self.sigma2 + eps)

        return scaled_Gi, scaled_dGi

    def batchnorm_backprop(self, dout, eps=1e-5):
        batch_size = len(dout)
        beta_grad = np.sum(dout, axis=0)
        gamma_grad = np.sum(dout * self.norm, axis=0)
        dxmu = dout * self.gamma / np.sqrt(self.sigma2 + eps) \
            - self.norm * np.sum(dout * self.gamma * (input - self.mu) / batch_size, axis=0) / (self.sigma2 + eps)
        # もしかしたら
        # dxmu = (1. / batch_size) * gamma * (sigma2 + eps)**(-1. / 2.) * \
        #        (batch_size * dout - np.sum(dout, axis=0) - (input - mu) * (sigma2 + eps)**(-1.0) * np.sum(dout * (input - mu), axis=0))
        # かもしれない
        din = dxmu - np.sum(dxmu / batch_size, axis=0)
        return beta_grad, gamma_grad, din

    def backprop(self, E_error, F_error):
        # backprop
        weight_grads = [np.zeros_like(weight) for weight in self.weights]
        bias_grads = [np.zeros_like(bias) for bias in self.bias]
        beta_grads = np.zeros_like(self.beta)
        gamma_grads = np.zeros_like(self.gamma)

        for i in reversed(range(self.nweight)):
            # energy
            e_delta = self.deriv_activation(self.outputs[i]) * np.dot(e_delta, self.weights[i+1].T) \
                if 'e_delta' in locals() else E_error  # squared loss
                # if 'e_delta' in locals() else np.clip(E_error, -1.0, 1.0)  # Huber loss
            weight_grads[i] += np.dot(self.inputs[i].T, e_delta)
            bias_grads[i] += np.sum(e_delta, axis=0)
            if i == 0:
                b, g, _ = self.batchnorm_backprop(np.dot(e_delta, self.weights[0].T))
                beta_grads += b
                gamma_grads += g

            # force
            f_delta = (self.second_deriv_activation(self.outputs[i]) * np.dot(self.inputs[i], self.weights[i]) +
                       self.deriv_activation(self.outputs[i]))[:, None, :] * np.tensordot(f_delta, self.weights[i+1], ((2,), (1,))) \
                if 'f_delta' in locals() else F_error  # squared loss
                # if 'f_delta' in locals() else np.clip(F_error, -1.0, 1.0)  # Huber loss
            weight_grads[i] += (hp.mixing_beta / (3 * self.all_natom)) * \
                np.tensordot(- self.deriv_inputs[i], f_delta, ((0, 1), (0, 1)))
            if i == 0:
                b, g, _ = self.batchnorm_backprop(np.tensordot(f_delta, self.weights[0], ((2,), (1,))))
                beta_grads += (hp.mixing_beta / (3 * self.all_natom)) * b
                gamma_grads += (hp.mixing_beta / (3 * self.all_natom)) * g

        return weight_grads, bias_grads, beta_grads, gamma_grads


class HDNNP(object):
    def __init__(self, natom, nsample):
        self.all_natom = natom
        self.nsample = nsample
        if bool_.SAVE_FIG:
            from animator import Animator
            self.animator = Animator(natom, nsample)

    def inference(self, Gs, dGs, size, mode):
        E, E_tmp = np.zeros((size, 1)), np.zeros((size, 1))
        F, F_tmp = np.zeros((size, 3*self.all_natom, 1)), np.zeros((size, 3*self.all_natom, 1))
        for nnp, i in zip(self.nnp, self.index):
            Ei, Fi = nnp.feedforward(Gs[:, i, :], dGs[:, i, :, :], size, mode)
            E_tmp += Ei
            F_tmp += Fi
        self.all_comm.Allreduce(E_tmp, E, op=MPI.SUM)
        self.all_comm.Allreduce(F_tmp, F, op=MPI.SUM)
        return E, F

    def training(self, m, Es, Fs, Gs, dGs):
        if hp.optimizer in ['sgd', 'adam']:
            batch_size = int(hp.batch_size * (1 + hp.batch_size_growth * m))
            if batch_size < 0 or batch_size > self.nsample:
                batch_size = self.nsample

            niter = -(- self.nsample / batch_size)
            for m in range(niter):
                sampling = np.random.randint(0, self.nsample, batch_size)
                self.all_comm.Bcast(sampling, root=0)

                grads = ([np.zeros_like(weight) for weight in self.nnp[0].weights], [np.zeros_like(bias) for bias in self.nnp[0].bias],
                         np.zeros_like(self.nnp[0].beta), np.zeros_like(self.nnp[0].gamma))
                para = ([np.zeros_like(weight) for weight in self.nnp[0].weights], [np.zeros_like(bias) for bias in self.nnp[0].bias],
                        np.zeros_like(self.nnp[0].beta), np.zeros_like(self.nnp[0].gamma))
                E_pred, F_pred = self.inference(Gs[sampling], dGs[sampling], batch_size, 'train')
                E_error = E_pred - Es[sampling]
                F_error = F_pred - Fs[sampling]
                for nnp in self.nnp:
                    tmp = nnp.backprop(E_error, F_error)
                    for (weight_p, bias_p, beta_p, gamma_p), (weight_t, bias_t, beta_t, gamma_t) in zip(para, tmp):
                        weight_p += weight_t
                        bias_p += bias_t
                        beta_p += beta_t
                        gamma_p += gamma_t
                for (weight, bias, beta, gamma), (weight_p, bias_p, beta_p, gamma_p) in zip(grads, para):
                    self.atomic_comm.Allreduce(weight_p/(batch_size*self.atomic_natom), weight, op=MPI.SUM)
                    self.atomic_comm.Allreduce(bias_p/(batch_size*self.atomic_natom), bias, op=MPI.SUM)
                    self.atomic_comm.Allreduce(beta_p/(batch_size*self.atomic_natom), beta, op=MPI.SUM)
                    self.atomic_comm.Allreduce(gamma_p/(batch_size*self.atomic_natom), gamma, op=MPI.SUM)

                weights, bias, beta, gamma = self.optimizer.update_params(*grads)
                for nnp in self.nnp:
                    nnp.weights = weights
                    nnp.bias = bias
                    nnp.beta = beta
                    nnp.gamma = gamma
        elif hp.optimizer == 'bfgs':
            print 'Info: BFGS is off-line learning method. "batch_size" is ignored.'
            # TODO: BFGS optimize
            self.optimizer.update_params()
        else:
            raise ValueError('invalid optimizer: select sgd or adam or bfgs')

    def calc_RMSE(self, m, Es, Fs, Gs, dGs):
        def rmse(pred, true):
            return np.sqrt(((pred - true)**2).mean())

        if bool_.SAVE_FIG and m == 0:
            self.animator.set_true(Es, Fs)
        E_preds, F_preds = self.inference(Gs, dGs, self.nsample, 'test')

        if bool_.SAVE_FIG:
            self.animator.set_pred(m, E_preds, F_preds)
        E_RMSE = rmse(E_preds, Es)
        F_RMSE = rmse(F_preds, Fs)
        RMSE = E_RMSE + hp.mixing_beta * F_RMSE
        return E_RMSE, F_RMSE, RMSE

    def save_fig(self, datestr, config, ext):
        if self.all_rank == 0:
            print 'saving figures ...'
            self.animator.save_fig(datestr, config, ext)

    def save(self, datestr):
        weight_save_dir = path.join(file_.weight_dir, datestr)
        if self.all_rank == 0 and not path.exists(weight_save_dir):
            makedirs(weight_save_dir)
        # before saving, sync EMA
        mu_EMA, sigma2_EMA = np.zeros_like(self.nnp[0].mu_EMA), np.zeros_like(self.nnp[0].sigma2_EMA)
        mu_EMA_p, sigma2_EMA_p = np.zeros_like(self.nnp[0].mu_EMA), np.zeros_like(self.nnp[0].sigma2_EMA)
        for nnp in self.nnp:
            mu_EMA_p += nnp.mu_EMA
            sigma2_EMA_p += nnp.sigma2_EMA
        self.atomic_comm.Allreduce(mu_EMA_p, mu_EMA, op=MPI.SUM)
        self.atomic_comm.Allreduce(sigma2_EMA_p, sigma2_EMA, op=MPI.SUM)
        # save weights, bias, beta, gamma, EMA
        if self.atomic_rank == 0:
            dicts = {}
            dicts.update({'weight_{}'.format(i): weight for i, weight in enumerate(self.nnp[0].weights)})
            dicts.update({'bias_{}'.format(i): bias for i, bias in enumerate(self.nnp[0].bias)})
            dicts.update({'beta': self.nnp[0].beta, 'gamma': self.nnp[0].gamma,
                          'mu_EMA': mu_EMA/self.atomic_natom, 'sigma2_EMA': sigma2_EMA/self.atomic_natom})
            np.savez(path.join(weight_save_dir, '{}.npz'.format(self.symbol)), **dicts)

    def load(self, datestr):
        npz_file = path.join(file_.weight_dir, datestr, '{}.npz'.format(self.symbol))
        if path.exists(npz_file):
            data = np.load(npz_file)
            for nnp in self.nnp:
                for i in range(nnp.nweight):
                    nnp.weights[i] = data['weight_{}'.format(i)]
                    nnp.bias[i] = data['bias_{}'.format(i)]
                nnp.beta = data['beta']
                nnp.gamma = data['gamma']
                nnp.mu_EMA = data['mu_EMA']
                nnp.sigma2_EMA = data['sigma2_EMA']
        else:
            if self.atomic_rank == 0:
                print 'pretrained data file {} is not found. use initialized parameters.'.format(npz_file)
            pass

    def sync(self):
        for i in range(self.nnp[0].nweight):
            self.atomic_comm.Bcast(self.nnp[0].weights[i], root=0)
            self.atomic_comm.Bcast(self.nnp[0].bias[i], root=0)

        for nnp in self.nnp:
            nnp.weights = self.nnp[0].weights
            nnp.bias = self.nnp[0].bias

    def initialize(self, comm, rank, size, ninput, composition):
        def comb(n, r):
            for c in combinations(range(1, n), r-1):
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
                        min_worker = {symbol[i]: worker[i] for i in range(len(symbol))}  # worker(node)
            return min_worker

        s = composition['number'].keys()  # symbol list
        n = composition['number'].values()  # natom list

        # allocate worker for each atom
        if len(s) > size:
            raise ValueError('the number of process must be {} or more.'.format(len(s)))
        elif size > self.all_natom:
            self.all_comm = comm.Create(comm.Get_group().Incl(range(self.all_natom)))
            if not rank < self.all_natom:
                return False
            self.all_rank = self.all_comm.Get_rank()
            w = composition['number']  # worker(node) ex.) {'Si': 3, 'Ge': 5}
        else:
            self.all_comm = comm
            self.all_rank = rank
            w = allocate(size, s, n)

        # split MPI communicator and set SingleNNP instances and initialize them
        low = 0
        for symbol, num in w.items():
            if low <= rank < low+num:
                self.symbol = symbol
                atomic_group = self.all_comm.Get_group().Incl(range(low, low+num))
                self.atomic_comm = self.all_comm.Create(atomic_group)
                self.atomic_rank = self.atomic_comm.Get_rank()
            low += num
        self.atomic_natom = composition['number'][self.symbol]
        quo, rem = self.atomic_natom / w[self.symbol], self.atomic_natom % w[self.symbol]
        if self.atomic_rank < rem:
            self.nnp = [SingleNNP(self.all_natom, ninput) for _ in range(quo+1)]
            self.index = list(composition['index'][self.symbol])[self.atomic_rank*(quo+1): (self.atomic_rank+1)*(quo+1)]
        else:
            self.nnp = [SingleNNP(self.all_natom, ninput) for _ in range(quo)]
            self.index = list(composition['index'][self.symbol])[self.atomic_rank*quo+rem: (self.atomic_rank+1)*quo+rem]

        self.sync()
        self.optimizer = OPTIMIZERS[hp.optimizer](self.nnp[0].weights, self.nnp[0].bias, self.nnp[0].beta, self.nnp[0].gamma)

        return True
