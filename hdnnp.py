# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
import scipy.special as sp

class single_nnp:
    def __init__(self, input_n, hidden1_n, hidden2_n, output_n, learning, name):
        # set number of nodes of each layers and learning rate
        self.input_n = input_n
        self.hidden1_n = hidden1_n
        self.hidden2_n = hidden2_n
        self.output_n = output_n
        self.learning = learning
        self.name = name
        
        # initialize weight parameters
        self.w = []
        self.b = []
        # input-hidden1
        self.w.append(np.random.normal(0.0, 0.5, (hidden1_n, input_n)))
        self.b.append(np.random.normal(0.0, 0.5, (hidden1_n, 1)))
        # hidden1-hidden2
        self.w.append(np.random.normal(0.0, 0.5, (hidden2_n, hidden1_n)))
        self.b.append(np.random.normal(0.0, 0.5, (hidden2_n, 1)))
        # hidden2-output
        self.w.append(np.random.normal(-0.1, 0.5, (output_n, hidden2_n)))
        self.b.append(np.random.normal(-0.1, 0.5, (output_n, 1)))
        
        # define activation function and derivative
        self.activation_func = lambda x: sp.expit(x)
        self.dif_activation_func = lambda x: sp.expit(x) * (1 - sp.expit(x))
        pass
    
    def train(self, var_list, dvar_list, energy_error, force_errors, beta):
        # feed_forward
        self.query(var_list)
        
        # back_prop
        # energy
        e_output_errors = np.array(energy_error, ndmin=2).T
        e_hidden2_errors = self.dif_activation_func(self.hidden2_inputs) * np.dot(self.w[2].T, e_output_errors)
        e_hidden1_errors = self.dif_activation_func(self.hidden1_inputs) * np.dot(self.w[1].T, e_hidden2_errors)
        
        e_grad_output_cost = np.dot(e_output_errors, np.transpose(self.hidden2_outputs))
        e_grad_hidden2_cost = np.dot(e_hidden2_errors, np.transpose(self.hidden1_outputs))
        e_grad_hidden1_cost = np.dot(e_hidden1_errors, np.array(var_list, ndmin=2))
        
        # forces
        size = len(dvar_list)
        f_grad_output_cost = np.zeros((self.output_n, self.hidden2_n))
        f_grad_hidden2_cost = np.zeros((self.hidden2_n, self.hidden1_n))
        f_grad_hidden1_cost = np.zeros((self.hidden1_n, self.input_n))
        for r in range(size):
            f_output_errors = np.array(force_errors[r], ndmin=2).T
            coef = np.dot(self.w[1], self.dif_activation_func(self.hidden1_inputs)) * np.dot(self.w[0], np.array([dvar_list[r]], ndmin=2).T)
            f_hidden2_errors = self.dif_activation_func(self.hidden2_inputs) * np.dot(- self.w[2].T * (1 - 2 * self.hidden2_outputs) * coef, f_output_errors)
            f_hidden1_errors = self.dif_activation_func(self.hidden1_inputs) * np.dot(self.w[1].T, f_hidden2_errors)
            
            f_grad_output_cost += np.dot(f_output_errors, np.transpose(- self.dif_activation_func(self.hidden2_inputs) * coef))
            f_grad_hidden2_cost += np.dot(f_hidden2_errors, np.transpose(self.hidden2_outputs))
            f_grad_hidden1_cost += np.dot(f_hidden1_errors, np.array(var_list, ndmin=2))
        
        # modify weight parameters
        w_grad,b_grad = [],[]
        w_grad.append(self.learning * (e_grad_hidden1_cost - beta * f_grad_hidden1_cost / size))
        w_grad.append(self.learning * (e_grad_hidden2_cost - beta * f_grad_hidden2_cost / size))
        w_grad.append(self.learning * (e_grad_output_cost - beta * f_grad_output_cost / size))
        b_grad.append(self.learning * (e_hidden1_errors - beta * f_hidden1_errors / size))
        b_grad.append(self.learning * (e_hidden2_errors - beta * f_hidden2_errors / size))
        b_grad.append(self.learning * (e_output_errors - beta * f_output_errors / size))
        return w_grad, b_grad
    
    def query(self, var_list):
        # feed_forward
        inputs = np.array(var_list, ndmin=2).T
        bias = np.array([1], ndmin=2)
        self.hidden1_inputs = np.dot(self.w[0], inputs) + np.dot(self.b[0], bias)
        self.hidden1_outputs = self.activation_func(self.hidden1_inputs)
        
        self.hidden2_inputs = np.dot(self.w[1], self.hidden1_outputs) + np.dot(self.b[1], bias)
        self.hidden2_outputs = self.activation_func(self.hidden2_inputs)
        
        self.final_inputs = np.dot(self.w[2], self.hidden2_outputs) + np.dot(self.b[2], bias)
        final_outputs = self.final_inputs
        
        return final_outputs[0]
        
    def differentiate(self, var_list, dvar_list):
        self.query(var_list)
        
        hidden1_inputs = np.array(dvar_list, ndmin=2).T
        hidden1_outputs = np.dot(self.w[0], hidden1_inputs)
        
        hidden2_inputs = self.dif_activation_func(self.hidden1_inputs) * hidden1_outputs
        hidden2_outputs = np.dot(self.w[1], hidden2_inputs)
        
        final_inputs = self.dif_activation_func(self.hidden2_inputs) * hidden2_outputs
        final_outputs = -1 * np.dot(self.w[2], final_inputs)
        
        return final_outputs
    
    def save_w(self, dire):
        np.save(dire+self.name+'_wih1.npy', self.w[0])
        np.save(dire+self.name+'_wh1h2.npy', self.w[1])
        np.save(dire+self.name+'_wh2o.npy', self.w[2])
        np.save(dire+self.name+'_bih1.npy', self.b[0])
        np.save(dire+self.name+'_bh1h2.npy', self.b[1])
        np.save(dire+self.name+'_bh2o.npy', self.b[2])
        
    def load_w(self, dire):
        self.w[0] = np.load(dire+self.name+'_wih1.npy')
        self.w[1] = np.load(dire+self.name+'_wh1h2.npy')
        self.w[2] = np.load(dire+self.name+'_wh2o.npy')
        self.b[0] = np.load(dire+self.name+'_bih1.npy')
        self.b[1] = np.load(dire+self.name+'_bh1h2.npy')
        self.b[2] = np.load(dire+self.name+'_bh2o.npy')


def train(comm, rank, nnp, natom, subnum, subdataset, beta):
    w_grad_sum = [np.zeros_like(nnp.w[0]),np.zeros_like(nnp.w[1]),np.zeros_like(nnp.w[2])]
    b_grad_sum = [np.zeros_like(nnp.b[0]),np.zeros_like(nnp.b[1]),np.zeros_like(nnp.b[2])]
    for n in range(subnum):
        Et = subdataset[n][0]
        Frt = subdataset[n][1]
        G = subdataset[n][2]
        dG = subdataset[n][3]
        E = query_E(comm, nnp, G[rank], natom)
        Fr = query_F(comm, nnp, G[rank], dG[rank], natom)
        E_error = (Et - E[0])
        F_errors = (Frt - Fr)

        w_grad,b_grad = nnp.train([G[rank]], [dGr for atom in dG[rank] for dGr in atom], [E_error], [force for atom in F_errors for force in atom], beta)

        for i in range(3):
            tmp = np.zeros_like(w_grad[i])
            comm.Allreduce(w_grad[i], tmp, op=MPI.SUM)
            w_grad_sum[i] += tmp
        for i in range(3):
            tmp = np.zeros_like(b_grad[i])
            comm.Allreduce(b_grad[i], tmp, op=MPI.SUM)
            b_grad_sum[i] += tmp
    
    for i in range(3):
        nnp.w[i] += w_grad_sum[i] / (subnum * natom)
    for i in range(3):
        nnp.b[i] += b_grad_sum[i] / (subnum * natom)

def query_E(comm, nnp, Gi, natom):
    Ei = nnp.query([Gi])
    E = np.zeros(1)
    comm.Allreduce(Ei, E, op=MPI.SUM)
    
    return E

def query_F(comm, nnp, Gi, dGi, natom):
    Fir = np.zeros((natom, 3))
    for k in range(natom):
        for l in range(3):
            Fir[k][l] = nnp.differentiate([Gi], [dGi[k][l]])
    Fr = np.zeros((natom, 3))
    comm.Allreduce(Fir, Fr, op=MPI.SUM)
    
    return Fr
