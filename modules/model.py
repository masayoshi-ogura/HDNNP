# -*- coding: utf-8 -*-

from os import path
from os import mkdir
from random import sample
import numpy as np
from mpi4py import MPI

from config import hp
from config import bool_
from config import file_
from activation_function import ACTIVATIONS, DERIVATIVES, SECOND_DERIVATIVES
from optimizer import OPTIMIZERS


class SingleNNP(object):
    def __init__(self, comm, natom, nsample, ninput):
        self.comm = comm
        self.natom = natom
        self.nsample = nsample
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

    def loss_func(self, E_true, E_pred, F_true, F_pred):
        # sync weights between the NNP of the same atom
        for i in range(self.nweight):
            tmp_weights = np.zeros_like(self.weights[i])
            tmp_bias = np.zeros_like(self.bias[i])
            self.comm.Allreduce(self.weights[i], tmp_weights, op=MPI.SUM)
            self.comm.Allreduce(self.bias[i], tmp_bias, op=MPI.SUM)
            self.weights[i] = tmp_weights / self.natom
            self.bias[i] = tmp_bias / self.natom
        # loss function
        loss = (E_pred - E_true)**2 + hp.mixing_beta * (F_pred - F_true)**2 / (3 * self.natom)
        return loss

    def feedforward(self, Gi, dGi):
        # "inputs" means "linear transformation's input", not "each layer's input"
        # "outputs" means "linear transformation's output", not "each layer's output"
        inputs, outputs, deriv_inputs, deriv_outputs = [], [], [], []
        for i in range(self.nweight):
            if i == 0:
                inputs.append(Gi)
                deriv_inputs.append(np.identity(self.shape[0]))
            else:
                inputs.append(self.activation(outputs[i-1]))
                deriv_inputs.append(self.deriv_activation(outputs[i-1])[None, :] * deriv_outputs[i-1])
            outputs.append(np.dot(inputs[i], self.weights[i]) + self.bias[i])
            deriv_outputs.append(np.dot(deriv_inputs[i], self.weights[i]))
        Ei = outputs[-1]
        Fi = - np.dot(dGi, deriv_outputs[-1])
        return Ei, Fi, inputs, outputs, deriv_inputs, deriv_outputs

    def backprop(self, Gi, dGi, E_error, F_error):
        # feedforward
        _, _, inputs, outputs, deriv_inputs, deriv_outputs = self.feedforward(Gi, dGi)

        # backprop
        weight_grads = [np.zeros_like(weight) for weight in self.weights]
        bias_grads = [np.zeros_like(bias) for bias in self.bias]

        for i in reversed(range(self.nweight)):
            # energy
            e_delta = self.deriv_activation(outputs[i]) * np.dot(self.weights[i+1], e_delta) \
                if 'e_delta' in locals() else E_error  # squared loss
                # if 'e_delta' in locals() else np.clip(E_error, -1.0, 1.0)  # Huber loss
            weight_grads[i] += inputs[i][:, None] * e_delta[None, :]
            bias_grads[i] += e_delta

            # force
            f_delta = (self.second_deriv_activation(outputs[i]) * np.dot(inputs[i], self.weights[i]) +
                       self.deriv_activation(outputs[i]))[None, :] * np.dot(f_delta, self.weights[i+1].T) \
                if 'f_delta' in locals() else F_error  # squared loss
                # if 'f_delta' in locals() else np.clip(F_error, -1.0, 1.0)  # Huber loss
            weight_grads[i] += (hp.mixing_beta / (3 * self.natom)) * \
                np.tensordot(- np.dot(dGi, deriv_inputs[i]), f_delta, ((0,), (0,)))

        return weight_grads, bias_grads


class HDNNP(object):
    def __init__(self, comm, rank, natom, nsample, ninput, composition):
        self.comm = comm
        self.rank = rank
        self.nnp = SingleNNP(comm, natom, nsample, ninput)
        self.optimizer = OPTIMIZERS[hp.optimizer](self.nnp.weights, self.nnp.bias)
        self.natom = natom
        self.nsample = nsample
        if bool_.SAVE_FIG:
            from animator import Animator
            self.animator = Animator(natom, nsample)

    def inference(self, Gi, dGi):
        Ei, Fi, _, _, _, _ = self.nnp.feedforward(Gi, dGi)
        E, F = np.zeros_like(Ei), np.zeros_like(Fi)
        self.comm.Allreduce(Ei, E, op=MPI.SUM)
        self.comm.Allreduce(Fi, F, op=MPI.SUM)
        return E, F

    def training(self, m, Es, Fs, Gs, dGs):
        if hp.optimizer in ['sgd', 'adam']:
            batch_size = int(hp.batch_size * (1 + hp.batch_size_growth * m))
            if batch_size < 0 or batch_size > self.nsample:
                hp.batch_size = self.nsample

            niter = -(- self.nsample / batch_size)
            for m in range(niter):
                sampling = sample(range(self.nsample), batch_size)
                sampling = self.comm.bcast(sampling, root=0)

                weight_grads = [np.zeros_like(weight) for weight in self.nnp.weights]
                bias_grads = [np.zeros_like(bias) for bias in self.nnp.bias]
                w_para = [np.zeros_like(weight) for weight in self.nnp.weights]
                b_para = [np.zeros_like(bias) for bias in self.nnp.bias]
                for i in sampling:
                    E_pred, F_pred = self.inference(Gs[i][self.rank], dGs[i][self.rank])
                    E_error = E_pred - Es[i]
                    F_error = F_pred - Fs[i]
                    w_tmp, b_tmp = self.nnp.backprop(Gs[i][self.rank], dGs[i][self.rank], E_error, F_error)
                    for w_p, b_p, w_t, b_t in zip(w_para, b_para, w_tmp, b_tmp):
                        w_p += w_t / (batch_size * self.natom)
                        b_p += b_t / (batch_size * self.natom)
                for w_grad, b_grad, w_p, b_p in zip(weight_grads, bias_grads, w_para, b_para):
                    self.comm.Allreduce(w_p, w_grad, op=MPI.SUM)
                    self.comm.Allreduce(b_p, b_grad, op=MPI.SUM)

                self.nnp.weights, self.nnp.bias = self.optimizer.update_params(weight_grads, bias_grads)
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
        E_preds, F_preds = np.zeros_like(Es), np.zeros_like(Fs)
        for i, (G, dG) in enumerate(zip(Gs, dGs)):
            E_pred, F_pred = self.inference(G[self.rank], dG[self.rank])
            E_preds[i] = E_pred
            F_preds[i] = F_pred

        if bool_.SAVE_FIG:
            self.animator.set_pred(m, E_preds, F_preds)
        E_RMSE = rmse(E_preds, Es)
        F_RMSE = rmse(F_preds, Fs)
        RMSE = E_RMSE + hp.mixing_beta * F_RMSE
        return E_RMSE, F_RMSE, RMSE

    def save_fig(self):
        self.animator.save_fig()

    def save_w(self, datestr):
        weight_save_dir = path.join(file_.weight_dir, datestr)
        mkdir(weight_save_dir)
        if self.rank == 0:
            for i in range(self.nnp.nweight):
                np.save(path.join(weight_save_dir, '{}_weights{}.npy'.format(file_.name, i)),
                        self.nnp.weights[i])
                np.save(path.join(weight_save_dir, '{}_bias{}.npy'.format(file_.name, i)),
                        self.nnp.bias[i])

    def load_w(self):
        for i in range(self.nnp.nweight):
            self.nnp.weights[i] = np.load(path.join(file_.weight_dir, '{}_weights{}.npy'.format(file_.name, i)))
            self.nnp.bias[i] = np.load(path.join(file_.weight_dir, '{}_bias{}.npy'.format(file_.name, i)))

    def sync_w(self):
        for i in range(self.nnp.nweight):
            self.comm.Bcast(self.nnp.weights[i], root=0)
            self.comm.Bcast(self.nnp.bias[i], root=0)
