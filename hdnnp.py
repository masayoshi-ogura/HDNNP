# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
import scipy.special as sp
import math

class single_nnp:
    def __init__(self, input_n, hidden1_n, hidden2_n, output_n, learning, beta, gamma, name):
        # set number of nodes of each layers and learning rate
        # beta: mixing parameter of error between energy and force
        # gamma: NAG parameter. rate of accumulation
        self.input_n = input_n
        self.hidden1_n = hidden1_n
        self.hidden2_n = hidden2_n
        self.output_n = output_n
        self.learning = learning
        self.beta = beta
        self.gamma = gamma
        self.name = name
        
        # initialize weight parameters
        self.w = []
        self.b = []
        # input-hidden1
        self.w.append(np.random.normal(0.0, 0.5, (hidden1_n, input_n)))
        self.b.append(np.random.normal(0.0, 0.5, (hidden1_n)))
        # hidden1-hidden2
        self.w.append(np.random.normal(0.0, 0.5, (hidden2_n, hidden1_n)))
        self.b.append(np.random.normal(0.0, 0.5, (hidden2_n)))
        # hidden2-output
        self.w.append(np.random.normal(-0.1, 0.5, (output_n, hidden2_n)))
        self.b.append(np.random.normal(-0.1, 0.5, (output_n)))
        
        # accumulation of weight parameters and bias parameters
        self.v_w = [np.zeros_like(self.w[0]),np.zeros_like(self.w[1]),np.zeros_like(self.w[2])]
        self.v_b = [np.zeros_like(self.b[0]),np.zeros_like(self.b[1]),np.zeros_like(self.b[2])]
        
        # define activation function and derivative
        self.activation_func = lambda x: sp.expit(x)
        self.dif_activation_func = lambda x: sp.expit(x) * (1 - sp.expit(x))
    
    ### input
    # Gi: numpy array (gnum)
    # dGi: numpy array (3*natom x gnum)
    # E_error: float
    # F_errors: numpy array (3*natom)
    ### output
    # w_grad: list of weight_parameters(numpy array)
    # b_grad: list of bias_parameters(numpy array)
    def gradient(self, Gi, dGi, E_error, F_errors):
        # feed_forward
        self.energy(Gi)
        
        # back_prop
        # energy
        e_output_errors = np.array([E_error])
        e_hidden2_errors = self.dif_activation_func(self.hidden2_inputs) * np.dot(self.w[2].T, e_output_errors)
        e_hidden1_errors = self.dif_activation_func(self.hidden1_inputs) * np.dot(self.w[1].T, e_hidden2_errors)
        
        e_grad_output_cost = np.matrix(e_output_errors).T * (self.hidden2_outputs)
        e_grad_hidden2_cost = np.matrix(e_hidden2_errors).T * (self.hidden1_outputs)
        e_grad_hidden1_cost = np.matrix(e_hidden1_errors).T * Gi
        
        # forces
        R = len(dGi)
        f_output_errors = np.zeros(1)
        f_hidden2_errors = np.zeros(self.hidden2_n)
        f_hidden1_errors = np.zeros(self.hidden1_n)
        f_grad_output_cost = np.zeros((self.output_n, self.hidden2_n))
        f_grad_hidden2_cost = np.zeros((self.hidden2_n, self.hidden1_n))
        f_grad_hidden1_cost = np.zeros((self.hidden1_n, self.input_n))
        for r in range(R):
            f_output_error = F_errors[r]
            coef = np.dot(self.w[1], self.dif_activation_func(self.hidden1_inputs) * np.dot(self.w[0], dGi[r]))
            f_hidden2_error = self.dif_activation_func(self.hidden2_inputs) * np.dot(- self.w[2], (1 - 2 * self.hidden2_outputs) * coef) * f_output_errors
            f_hidden1_error = self.dif_activation_func(self.hidden1_inputs) * np.dot(self.w[1].T, f_hidden2_errors)
            
            f_output_errors += f_output_error
            f_hidden2_errors += f_hidden2_error
            f_hidden1_errors += f_hidden1_error
            f_grad_output_cost += np.matrix(f_output_error).T * (- self.dif_activation_func(self.hidden2_inputs) * coef)
            f_grad_hidden2_cost += np.matrix(f_hidden2_error).T * self.hidden1_outputs
            f_grad_hidden1_cost += np.matrix(f_hidden1_error).T * Gi
        
        # modify weight parameters
        w_grad,b_grad = [],[]
        w_grad.append(self.learning * (e_grad_hidden1_cost - self.beta * f_grad_hidden1_cost / R))
        w_grad.append(self.learning * (e_grad_hidden2_cost - self.beta * f_grad_hidden2_cost / R))
        w_grad.append(self.learning * (e_grad_output_cost - self.beta * f_grad_output_cost / R))
        b_grad.append(self.learning * (e_hidden1_errors - self.beta * f_hidden1_errors / R))
        b_grad.append(self.learning * (e_hidden2_errors - self.beta * f_hidden2_errors / R))
        b_grad.append(self.learning * (e_output_errors - self.beta * f_output_errors / R))
        return w_grad, b_grad
    
    def energy(self, Gi):
        # feed_forward
        bias = np.ones(1)
        self.hidden1_inputs = np.dot(self.w[0], Gi) + (self.b[0] * bias)
        self.hidden1_outputs = self.activation_func(self.hidden1_inputs)
        
        self.hidden2_inputs = np.dot(self.w[1], self.hidden1_outputs) + (self.b[1] * bias)
        self.hidden2_outputs = self.activation_func(self.hidden2_inputs)
        
        self.final_inputs = np.dot(self.w[2], self.hidden2_outputs) + (self.b[2] * bias)
        final_outputs = self.final_inputs
        
        return final_outputs
    
    def force(self, Gi, dGi):
        self.energy(Gi)
        
        hidden1_outputs = np.dot(self.w[0], dGi.T)
        
        hidden2_inputs = self.dif_activation_func(self.hidden1_inputs) * hidden1_outputs
        hidden2_outputs = np.dot(self.w[1], hidden2_inputs)
        
        final_inputs = self.dif_activation_func(self.hidden2_inputs) * hidden2_outputs
        final_outputs = -1 * np.dot(self.w[2], final_inputs)
        
        return final_outputs.reshape(-1)
    
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

    ### input
    # comm, rank: MPI communicator, rank of the processor
    # subdataset: list of following 4 objects
    #             energy: float
    #             forces: numpy array (3*natom)
    #             G: numpy array (natom x gnum)
    #             dG: numpy array (natom x 3*natom x gnum)
    def train(self, comm, rank, natom, subnum, subdataset):
        w_grad_sum = [np.zeros_like(self.w[0]),np.zeros_like(self.w[1]),np.zeros_like(self.w[2])]
        b_grad_sum = [np.zeros_like(self.b[0]),np.zeros_like(self.b[1]),np.zeros_like(self.b[2])]
        
        # before calculating grad_sum, renew weight and bias parameters with old v_w and v_b
        for i in range(3):
            self.w[i] += self.gamma * self.v_w[i]
            self.b[i] += self.gamma * self.v_b[i]
        
        # calculate grad_sum
        for n in range(subnum):
            Et = subdataset[n][0]
            Frt = subdataset[n][1]
            G = subdataset[n][2]
            dG = subdataset[n][3]
            E = self.query_E(comm, G[rank], natom)
            Fr = self.query_F(comm, G[rank], dG[rank], natom)
            E_error = (Et - E)
            F_errors = (Frt - Fr)

            w_grad,b_grad = self.gradient(G[rank], dG[rank], E_error, F_errors)

            for i in range(3):
                w_recv = np.zeros_like(w_grad[i])
                b_recv = np.zeros_like(b_grad[i])
                comm.Allreduce(w_grad[i], w_recv, op=MPI.SUM)
                comm.Allreduce(b_grad[i], b_recv, op=MPI.SUM)
                w_grad_sum[i] += w_recv
                b_grad_sum[i] += b_recv
        
        # renew weight and bias parameters with calculated gradient
        for i in range(3):
            self.w[i] += w_grad_sum[i] / (subnum * natom)
            self.b[i] += b_grad_sum[i] / (subnum * natom)
            self.v_w[i] = (self.gamma * self.v_w[i]) + (w_grad_sum[i] / (subnum * natom))
            self.v_b[i] = (self.gamma * self.v_b[i]) + (b_grad_sum[i] / (subnum * natom))

    # calculte energy
    ### input
    # comm: MPI communicator
    # Gi: numpy array (gnum)
    ### output
    # E: float
    def query_E(self, comm, Gi, natom):
        Ei = self.energy(Gi)
        E = np.zeros(1)
        comm.Allreduce(Ei, E, op=MPI.SUM)
        
        return E[0]

    # calculate force
    ### input
    # comm: MPI communicator
    # Gi: numpy array (gnum)
    # dGi: numpy array (3*natom x gnum)
    ### output
    # Fr: numpy array (3*natom)
    def query_F(self, comm, Gi, dGi, natom):
        Fir = np.zeros(3*natom)
        for r in range(3*natom):
            Fir[r] = self.force(Gi, dGi[r])
        Fr = np.zeros(3*natom)
        comm.Allreduce(Fir, Fr, op=MPI.SUM)
        
        return Fr
        
    # calculate RMSE
    ### input
    # comm, rank: MPI communicator, rank of the processor
    # dataset: list of following 4 objects
    #          energy: float
    #          forces: numpy array (3*natom)
    #          G: numpy array (natom x gnum)
    #          dG: numpy array (natom x 3*natom x gnum)
    ### output
    # E_RMSE: float
    # F_RMSE: float
    def calc_RMSE(self, comm, rank, natom, nsample, dataset):
        E_MSE = 0.0
        F_MSE = 0.0
        for n in range(nsample):
            Et = dataset[n][0]
            Frt = dataset[n][1]
            G = dataset[n][2]
            dG = dataset[n][3]
            E_out = self.query_E(comm, G[rank], natom)
            F_rout = self.query_F(comm, G[rank], dG[rank], natom)
            E_MSE += (Et - E_out) ** 2
            F_MSE += np.sum((Frt - F_rout)**2)
        E_RMSE = math.sqrt(E_MSE / nsample)
        F_RMSE = math.sqrt(F_MSE / (nsample * natom * 3))
        RMSE = E_RMSE + beta * F_RMSE
        
        return E_RMSE, F_RMSE, RMSE
