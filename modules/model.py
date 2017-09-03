# -*- coding: utf-8 -*-

from os import path
from os import mkdir
from math import sqrt
from random import sample
import numpy as np
from mpi4py import MPI

from config import hp
from config import file_
from activation_function import ACTIVATIONS, DERIVATIVES
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
        Ei = np.dot(self.weights[-1], outputs) + self.bias[-1]
        Fi = - np.tensordot(deriv_outputs, self.weights[-1], ((1,), (1,)))
        inputs.append(Ei)
        outputs.append(None)
        deriv_inputs.append(Fi)
        deriv_outputs.append(None)
        return inputs, outputs, deriv_inputs, deriv_outputs

    def backprop(self, Gi, dGi, E_error, F_error):
        # feedforward
        inputs, outputs, deriv_inputs, deriv_outputs = self.feedforward(Gi, dGi)

        # backprop
        # energy
        weight_grads = [np.zeros_like(weights) for weights in self.weights]
        bias_grads = [np.zeros_like(bias) for bias in self.bias]
        for i in reversed(range(self.nweight)):
            delta = self.deriv_activation(inputs[i+1]) * np.dot(delta, self.weights[i+1]) \
                    if 'delta' in locals() else np.array([E_error])
            weight_grads[i] += delta[:, None] * outputs[i][None, :]
            bias_grads[i] += delta
    #
    #     # forces
    #     f_output_errors = np.zeros(1)
    #     f_hidden2_errors = np.zeros(self.hidden2_nodes)
    #     f_hidden1_errors = np.zeros(self.hidden1_nodes)
    #     f_grad_output_cost = np.zeros((self.output_nodes, self.hidden2_nodes))
    #     f_grad_hidden2_cost = np.zeros((self.hidden2_nodes, self.hidden1_nodes))
    #     f_grad_hidden1_cost = np.zeros((self.hidden1_nodes, self.input_nodes))
    #     for r in range(3*self.natom):
    #         f_output_error = np.array([F_errors[r]])
    #         coef = np.dot(self.w[1], self.dif_activation_func(self.hidden1_inputs) * np.dot(self.w[0], dGi[r]))
    #         f_hidden2_error = self.dif_activation_func(self.hidden2_inputs) * \
    #             np.dot(- self.w[2], (1 - 2 * self.hidden2_outputs) * coef) * f_output_error
    #         f_hidden1_error = self.dif_activation_func(self.hidden1_inputs) * \
    #             np.dot(self.w[1].T, f_hidden2_error)
    #
    #         f_output_errors += f_output_error
    #         f_hidden2_errors += f_hidden2_error
    #         f_hidden1_errors += f_hidden1_error
    #         f_grad_output_cost += np.dot(f_output_error[:, None], (- self.dif_activation_func(self.hidden2_inputs) * coef)[None, :])
    #         f_grad_hidden2_cost += np.dot(f_hidden2_error[:, None], self.hidden1_outputs[None, :])
    #         f_grad_hidden1_cost += np.dot(f_hidden1_error[:, None], Gi[None, :])
        return weight_grads, bias_grads


class HDNNP(object):
    def __init__(self, comm, rank, nsample, ninput):
        self.comm = comm
        self.rank = rank
        self.nnp = SingleNNP(comm, nsample, ninput)
        self.optimizer = OPTIMIZERS[hp.optimizer](self.nnp.weights, self.nnp.bias)
        self.nsample = nsample

    def inference(self):
        inputs, _, deriv_inputs, _ = self.nnp.feedforward()
        Ei = inputs[-1]
        Fi = deriv_inputs[-1]
        E, F = np.zeros_like(Ei), np.zeros_like(Fi)
        self.comm.Allreduce(Ei, E, op=MPI.SUM)
        self.comm.Allreduce(Fi, F, op=MPI.SUM)
        return E, F

    def training(self, alldataset):
        if hp.optimizer in ['SGD', 'Adam']:
            # TODO: random sample from alldataset
            subdataset = sample(alldataset, hp.batch_size)
            self.comm.bcast(subdataset, root=0)
            for n in range():
                # TODO: calc grad in subset and get average and update
                E_true, F_true, G, dG = subdataset[n]
                E_pred, F_pred = self.inference(G[self.rank], dG[self.rank])
                E_error = E_pred - E_true
                F_error = F_pred - F_true
                weight_grads, bias_grads = self.nnp.backprop(G[self.rank], dG[self.rank], E_error, F_error)
                self.optimizer.update_params(weight_grads, bias_grads)
        elif hp.optimizer == 'BFGS':
            print 'Info: BFGS is off-line learning method. "batch_size" is ignored.'
            # TODO: BFGS optimize
            self.optimizer.update_params()
        else:
            raise ValueError('invalid optimizer: select SGD or Adam or BFGS')

    def calc_RMSE(self, dataset):
        E_MSE = 0.0
        F_MSE = 0.0
        for n in range(self.nsample):
            E_true, F_true, G, dG = dataset[n]
            E_pred, F_pred = self.inference(G[self.rank], dG[self.rank])
            E_MSE += (E_pred - E_true) ** 2
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
