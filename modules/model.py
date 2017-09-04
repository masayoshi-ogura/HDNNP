# -*- coding: utf-8 -*-

from os import path
from os import mkdir
from math import sqrt
from random import sample
import numpy as np
from mpi4py import MPI

from config import hp
from config import file_
from activation_function import ACTIVATIONS, DERIVATIVES, SECOND_DERIVATIVES
from optimizer import OPTIMIZERS


class SingleNNP(object):
    def __init__(self, comm, nsample, ninput):
        self.comm = comm
        self.natom = hp.natom
        self.nsample = nsample
        self.shape = (ninput,) + hp.hidden_layer + (1,)
        self.nlayer = len(hp.hidden_layer)
        self.nweight = len(hp.hidden_layer) + 1
        self.learning_rate = hp.learning_rate
        self.beta = hp.beta
        self.activation = ACTIVATIONS[hp.activation]
        self.deriv_activation = DERIVATIVES[hp.activation]
        self.second_deriv_activation = SECOND_DERIVATIVES[hp.activation]

        self.weights, self.bias = [], []
        for i in range(self.nweight):
            self.weights.append(np.random.normal(0.0, 0.5, (self.shape[i+1], self.shape[i])))
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
        loss = (E_pred - E_true)**2 + hp.beta * (F_pred - F_true)**2 / (3 * hp.natom)
        return loss

    def feedforward(self, Gi, dGi):
        inputs = [None]
        outputs = [Gi]
        deriv_inputs = [None]
        deriv_outputs = [dGi]
        for i in range(self.nlayer):
            inputs.append(np.dot(self.weights[i], outputs[i]) + self.bias[i])
            outputs.append(self.activation(inputs[i+1]))
            deriv_inputs.append(np.tensordot(deriv_outputs[i], self.weights[i], ((1,), (1,))))
            deriv_outputs.append(self.deriv_activation(inputs[i+1])[None, :] * deriv_inputs[i+1])
        Ei = np.dot(self.weights[-1], outputs[-1]) + self.bias[-1]
        Fi = - np.tensordot(deriv_outputs[-1], self.weights[-1], ((1,), (1,)))
        inputs.append(Ei)
        outputs.append(None)
        deriv_inputs.append(Fi)
        deriv_outputs.append(None)
        return inputs, outputs, deriv_inputs, deriv_outputs

    def backprop(self, Gi, dGi, E_error, F_error):
        # feedforward
        inputs, outputs, deriv_inputs, deriv_outputs = self.feedforward(Gi, dGi)

        # backprop
        weight_grads = [np.zeros_like(weight) for weight in self.weights]
        bias_grads = [np.zeros_like(bias) for bias in self.bias]

        for i in reversed(range(self.nweight)):
            # energy
            e_delta = self.deriv_activation(inputs[i+1]) * np.dot(e_delta, self.weights[i+1]) \
                if 'e_delta' in locals() else E_error
            weight_grads[i] += e_delta[:, None] * outputs[i][None, :]
            bias_grads[i] += e_delta

            # force
            f_delta = (self.second_deriv_activation(inputs[i+1]) * np.dot(self.weights[i], outputs[i]) +
                       self.deriv_activation(inputs[i+1]))[None, :] * np.dot(f_delta, self.weights[i+1]) \
                if 'f_delta' in locals() else F_error
            weight_grads[i] += hp.beta * np.tensordot(f_delta, -deriv_outputs[i], ((0,), (0,))) / (3 * hp.natom)

        return weight_grads, bias_grads


class HDNNP(object):
    def __init__(self, comm, rank, nsample, ninput):
        self.comm = comm
        self.rank = rank
        self.nnp = SingleNNP(comm, nsample, ninput)
        self.optimizer = OPTIMIZERS[hp.optimizer](self.nnp.weights, self.nnp.bias)
        self.nsample = nsample

    def inference(self, Gi, dGi):
        inputs, _, deriv_inputs, _ = self.nnp.feedforward(Gi, dGi)
        Ei = inputs[-1]
        Fi = deriv_inputs[-1]
        E, F = np.zeros_like(Ei), np.zeros_like(Fi)
        self.comm.Allreduce(Ei, E, op=MPI.SUM)
        self.comm.Allreduce(Fi, F, op=MPI.SUM)
        return E, F

    def training(self, dataset):
        if hp.optimizer in ['sgd', 'adam']:
            for m in range(self.nsample / hp.batch_size + 1):
                subdataset = sample(dataset, hp.batch_size)
                self.comm.bcast(subdataset, root=0)

                weight_grads = [np.zeros_like(weight) for weight in self.nnp.weights]
                bias_grads = [np.zeros_like(bias) for bias in self.nnp.bias]
                w_para = [np.zeros_like(weight) for weight in self.nnp.weights]
                b_para = [np.zeros_like(bias) for bias in self.nnp.bias]
                for E_true, F_true, G, dG in subdataset:
                    E_pred, F_pred = self.inference(G[self.rank], dG[self.rank])
                    E_error = E_pred - E_true
                    F_error = F_pred - F_true
                    w_tmp, b_tmp = self.nnp.backprop(G[self.rank], dG[self.rank], E_error, F_error)
                    for w_p, b_p, w_t, b_t in zip(w_para, b_para, w_tmp, b_tmp):
                        w_p += w_t / (self.nsample * hp.natom)
                        b_p += b_t / (self.nsample * hp.natom)
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

    def calc_RMSE(self, dataset):
        E_MSE = 0.0
        F_MSE = 0.0
        for E_true, F_true, G, dG in dataset:
            E_pred, F_pred = self.inference(G[self.rank], dG[self.rank])
            E_MSE += np.sum((E_pred - E_true) ** 2)
            F_MSE += np.sum((F_pred - F_true)**2)
        E_RMSE = sqrt(E_MSE / self.nsample)
        F_RMSE = sqrt(F_MSE / (self.nsample * hp.natom * 3))
        RMSE = E_RMSE + hp.beta * F_RMSE
        return E_RMSE, F_RMSE, RMSE

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
