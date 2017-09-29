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

    # def loss_func(self, E_true, E_pred, F_true, F_pred):
    #     # sync weights between the NNP of the same atom
    #     for i in range(self.nweight):
    #         tmp_weights = np.zeros_like(self.weights[i])
    #         tmp_bias = np.zeros_like(self.bias[i])
    #         self.comm.Allreduce(self.weights[i], tmp_weights, op=MPI.SUM)
    #         self.comm.Allreduce(self.bias[i], tmp_bias, op=MPI.SUM)
    #         self.weights[i] = tmp_weights / self.natom
    #         self.bias[i] = tmp_bias / self.natom
    #     # loss function
    #     loss = (E_pred - E_true)**2 + hp.mixing_beta * (F_pred - F_true)**2 / (3 * self.natom)
    #     return loss

    def feedforward(self, Gi, dGi, size):
        # "inputs" means "linear transformation's input", not "each layer's input"
        # "outputs" means "linear transformation's output", not "each layer's output"
        inputs, outputs, deriv_inputs, deriv_outputs = [], [], [], []
        for i in range(self.nweight):
            if i == 0:
                inputs.append(Gi)
                deriv_inputs.append(dGi)
            else:
                inputs.append(self.activation(outputs[i-1]))
                deriv_inputs.append(self.deriv_activation(outputs[i-1])[:, None, :] * deriv_outputs[i-1])
            outputs.append(np.dot(inputs[i], self.weights[i]) + self.bias[i])
            deriv_outputs.append(np.dot(deriv_inputs[i], self.weights[i]))
        Ei = outputs[-1]
        Fi = - deriv_outputs[-1]
        return Ei, Fi, inputs, outputs, deriv_inputs, deriv_outputs

    def backprop(self, Gi, dGi, E_error, F_error, size):
        # feedforward
        _, _, inputs, outputs, deriv_inputs, deriv_outputs = self.feedforward(Gi, dGi, size)

        # backprop
        weight_grads = [np.zeros_like(weight) for weight in self.weights]
        bias_grads = [np.zeros_like(bias) for bias in self.bias]

        for i in reversed(range(self.nweight)):
            # energy
            e_delta = self.deriv_activation(outputs[i]) * np.dot(e_delta, self.weights[i+1].T) \
                if 'e_delta' in locals() else E_error  # squared loss
                # if 'e_delta' in locals() else np.clip(E_error, -1.0, 1.0)  # Huber loss
            weight_grads[i] += np.dot(inputs[i].T, e_delta)
            bias_grads[i] += np.sum(e_delta, axis=0)

            # force
            f_delta = (self.second_deriv_activation(outputs[i]) * np.dot(inputs[i], self.weights[i]) +
                       self.deriv_activation(outputs[i]))[:, None, :] * np.dot(f_delta, self.weights[i+1].T) \
                if 'f_delta' in locals() else F_error  # squared loss
                # if 'f_delta' in locals() else np.clip(F_error, -1.0, 1.0)  # Huber loss
            weight_grads[i] += (hp.mixing_beta / (3 * self.all_natom)) * \
                np.tensordot(- deriv_inputs[i], f_delta, ((0, 1), (0, 1)))

        return weight_grads, bias_grads


class HDNNP(object):
    def __init__(self, natom, nsample):
        self.all_natom = natom
        self.nsample = nsample
        if bool_.SAVE_FIG:
            from animator import Animator
            self.animator = Animator(natom, nsample)

    def inference(self, Gs, dGs, size):
        E, E_tmp = np.zeros((size, 1)), np.zeros((size, 1))
        F, F_tmp = np.zeros((size, 3*self.all_natom, 1)), np.zeros((size, 3*self.all_natom, 1))
        for nnp, i in zip(self.nnp, self.index):
            Ei, Fi, _, _, _, _ = nnp.feedforward(Gs[:, i, :], dGs[:, i, :, :], size)
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

                weight_grads = [np.zeros_like(weight) for weight in self.nnp[0].weights]
                bias_grads = [np.zeros_like(bias) for bias in self.nnp[0].bias]
                weight_para = [np.zeros_like(weight) for weight in self.nnp[0].weights]
                bias_para = [np.zeros_like(bias) for bias in self.nnp[0].bias]
                E_pred, F_pred = self.inference(Gs[sampling], dGs[sampling], batch_size)
                E_error = E_pred - Es[sampling]
                F_error = F_pred - Fs[sampling]
                for nnp, i in zip(self.nnp, self.index):
                    w_tmp, b_tmp = nnp.backprop(Gs[sampling, i, :], dGs[sampling, i, :, :], E_error, F_error, batch_size)
                    for w_p, b_p, w_t, b_t in zip(weight_para, bias_para, w_tmp, b_tmp):
                        w_p += w_t / (batch_size * self.atomic_natom)
                        b_p += b_t / (batch_size * self.atomic_natom)
                for w_g, b_g, w_p, b_p in zip(weight_grads, bias_grads, weight_para, bias_para):
                    self.atomic_comm.Allreduce(w_p, w_g, op=MPI.SUM)
                    self.atomic_comm.Allreduce(b_p, b_g, op=MPI.SUM)

                weights, bias = self.optimizer.update_params(weight_grads, bias_grads)
                for nnp in self.nnp:
                    nnp.weights = weights
                    nnp.bias = bias
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
        E_preds, F_preds = self.inference(Gs, dGs, self.nsample)

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

    def save_w(self, datestr):
        weight_save_dir = path.join(file_.weight_dir, datestr)
        if self.all_rank == 0 and not path.exists(weight_save_dir):
            makedirs(weight_save_dir)
        self.all_comm.Barrier()
        if self.atomic_rank == 0:
            weights = {str(i): self.nnp[0].weights[i] for i in range(self.nnp[0].nweight)}
            bias = {str(i): self.nnp[0].bias[i] for i in range(self.nnp[0].nweight)}
            np.savez(path.join(weight_save_dir, '{}_weights.npz'.format(self.symbol)), **weights)
            np.savez(path.join(weight_save_dir, '{}_bias.npz'.format(self.symbol)), **bias)

    def load_w(self, datestr):
        weight_file = path.join(file_.weight_dir, datestr, '{}_weights.npz'.format(self.symbol))
        bias_file = path.join(file_.weight_dir, datestr, '{}_bias.npz'.format(self.symbol))
        if path.exists(weight_file):
            weights = np.load(weight_file)
            bias = np.load(bias_file)
            for nnp in self.nnp:
                for i in range(nnp.nweight):
                    nnp.weights[i] = weights[str(i)]
                    nnp.bias[i] = bias[str(i)]
        else:
            if self.atomic_rank == 0:
                print 'weight params file {} is not found. use initialized parameters.'.format(weight_file)
            pass

    def sync_w(self):
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
        if size == 1:
            raise ValueError('the number of process must be 2 or more.')
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
        self.nnp = [SingleNNP(self.all_natom, ninput)]
        quo, rem = self.atomic_natom / w[self.symbol], self.atomic_natom % w[self.symbol]
        if self.atomic_rank < rem:
            self.nnp *= quo+1
            self.index = list(composition['index'][self.symbol])[self.atomic_rank*(quo+1): (self.atomic_rank+1)*(quo+1)]
        else:
            self.nnp *= quo
            self.index = list(composition['index'][self.symbol])[self.atomic_rank*quo+rem: (self.atomic_rank+1)*quo+rem]

        self.sync_w()
        self.optimizer = OPTIMIZERS[hp.optimizer](self.nnp[0].weights, self.nnp[0].bias)

        return True
