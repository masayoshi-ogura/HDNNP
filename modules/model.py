# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
import scipy.special as sp
import math
from os import path

class single_nnp:
    def __init__(self, comm, rank, network_shape, learning_rate, beta, gamma, natom, nsample):
        self.comm = comm
        self.rank = rank
        self.natom = natom
        self.nsample = nsample
        # set number of nodes of each layers and learning rate
        # beta: mixing parameter of error between energy and force
        # gamma: NAG parameter. rate of accumulation
        self.input_nodes,self.hidden1_nodes,self.hidden2_nodes,self.output_nodes = network_shape
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        
        # initialize weight parameters
        self.w = []
        self.b = []
        # input-hidden1
        self.w.append(np.random.normal(0.0, 0.5, (self.hidden1_nodes, self.input_nodes)))
        self.b.append(np.random.normal(0.0, 0.5, (self.hidden1_nodes)))
        # hidden1-hidden2
        self.w.append(np.random.normal(0.0, 0.5, (self.hidden2_nodes, self.hidden1_nodes)))
        self.b.append(np.random.normal(0.0, 0.5, (self.hidden2_nodes)))
        # hidden2-output
        self.w.append(np.random.normal(-0.1, 0.5, (self.output_nodes, self.hidden2_nodes)))
        self.b.append(np.random.normal(-0.1, 0.5, (self.output_nodes)))
        
        # accumulation of weight parameters and bias parameters
        self.v_w = [np.zeros_like(self.w[0]),np.zeros_like(self.w[1]),np.zeros_like(self.w[2])]
        self.v_b = [np.zeros_like(self.b[0]),np.zeros_like(self.b[1]),np.zeros_like(self.b[2])]
        
        # define activation function and derivative
        self.activation_func = lambda x: sp.expit(x)
        self.dif_activation_func = lambda x: sp.expit(x) * (1 - sp.expit(x))

    ### input
    # subdataset: list of following 4 objects
    #             energy: float
    #             forces: numpy array (3*natom)
    #             G: numpy array (natom x ninput)
    #             dG: numpy array (natom x 3*natom x ninput)
    def train(self, nsubset, subdataset):
        w_grad_sum = [np.zeros_like(self.w[0]),np.zeros_like(self.w[1]),np.zeros_like(self.w[2])]
        b_grad_sum = [np.zeros_like(self.b[0]),np.zeros_like(self.b[1]),np.zeros_like(self.b[2])]
        
        # before calculating grad_sum, renew weight and bias parameters with old v_w and v_b
        for i in range(3):
            self.w[i] += self.gamma * self.v_w[i]
            self.b[i] += self.gamma * self.v_b[i]
        
        # calculate grad_sum
        for n in range(nsubset):
            Et = subdataset[n][0]
            Frt = subdataset[n][1]
            G = subdataset[n][2]
            dG = subdataset[n][3]
            E = self.__query_E(G[self.rank])
            Fr = self.__query_F(G[self.rank], dG[self.rank])
            E_error = (Et - E)
            F_errors = (Frt - Fr)

            w_grad,b_grad = self.__gradient(G[self.rank], dG[self.rank], E_error, F_errors)

            for i in range(3):
                w_recv = np.zeros_like(w_grad[i])
                b_recv = np.zeros_like(b_grad[i])
                self.comm.Allreduce(w_grad[i], w_recv, op=MPI.SUM)
                self.comm.Allreduce(b_grad[i], b_recv, op=MPI.SUM)
                w_grad_sum[i] += w_recv
                b_grad_sum[i] += b_recv
        
        # renew weight and bias parameters with calculated gradient
        for i in range(3):
            self.w[i] += w_grad_sum[i] / (nsubset * self.natom)
            self.b[i] += b_grad_sum[i] / (nsubset * self.natom)
            self.v_w[i] = (self.gamma * self.v_w[i]) + (w_grad_sum[i] / (nsubset * self.natom))
            self.v_b[i] = (self.gamma * self.v_b[i]) + (b_grad_sum[i] / (nsubset * self.natom))
    
    def save_w(self, dire, name):
        np.save(path.join(dire, name+'_wih1.npy'), self.w[0])
        np.save(path.join(dire, name+'_wh1h2.npy'), self.w[1])
        np.save(path.join(dire, name+'_wh2o.npy'), self.w[2])
        np.save(path.join(dire, name+'_bih1.npy'), self.b[0])
        np.save(path.join(dire, name+'_bh1h2.npy'), self.b[1])
        np.save(path.join(dire, name+'_bh2o.npy'), self.b[2])
    
    def load_w(self, dire, name):
        self.w[0] = np.load(path.join(dire, name+'_wih1.npy'))
        self.w[1] = np.load(path.join(dire, name+'_wh1h2.npy'))
        self.w[2] = np.load(path.join(dire, name+'_wh2o.npy'))
        self.b[0] = np.load(path.join(dire, name+'_bih1.npy'))
        self.b[1] = np.load(path.join(dire, name+'_bh1h2.npy'))
        self.b[2] = np.load(path.join(dire, name+'_bh2o.npy'))
    
    # calculate RMSE
    ### input
    # dataset: list of following 4 objects
    #          energy: float
    #          forces: numpy array (3*natom)
    #          G: numpy array (natom x ninput)
    #          dG: numpy array (natom x 3*natom x ninput)
    ### output
    # E_RMSE: float
    # F_RMSE: float
    def calc_RMSE(self, dataset):
        E_MSE = 0.0
        F_MSE = 0.0
        for n in range(self.nsample):
            Et = dataset[n][0]
            Frt = dataset[n][1]
            G = dataset[n][2]
            dG = dataset[n][3]
            E_out = self.__query_E(G[self.rank])
            F_rout = self.__query_F(G[self.rank], dG[self.rank])
            E_MSE += (Et - E_out) ** 2
            F_MSE += np.sum((Frt - F_rout)**2)
        E_RMSE = math.sqrt(E_MSE / self.nsample)
        F_RMSE = math.sqrt(F_MSE / (self.nsample * self.natom * 3))
        RMSE = E_RMSE + self.beta * F_RMSE
        
        return E_RMSE, F_RMSE, RMSE
    
    ### input
    # Gi: numpy array (ninput)
    # dGi: numpy array (3*natom x ninput)
    # E_error: float
    # F_errors: numpy array (3*natom)
    ### output
    # w_grad: list of weight_parameters(numpy array)
    # b_grad: list of bias_parameters(numpy array)
    
    def __gradient(self, Gi, dGi, E_error, F_errors):
        # feed_forward
        self.__energy(Gi)
        
        # back_prop
        # energy
        e_output_errors = np.array([E_error])
        e_hidden2_errors = self.dif_activation_func(self.hidden2_inputs) * np.dot(self.w[2].T, e_output_errors)
        e_hidden1_errors = self.dif_activation_func(self.hidden1_inputs) * np.dot(self.w[1].T, e_hidden2_errors)
        
        e_grad_output_cost = np.dot(e_output_errors.reshape((-1,1)), self.hidden2_outputs.reshape((1,-1)))
        e_grad_hidden2_cost = np.dot(e_hidden2_errors.reshape((-1,1)), self.hidden1_outputs.reshape((1,-1)))
        e_grad_hidden1_cost = np.dot(e_hidden1_errors.reshape((-1,1)), Gi.reshape((1,-1)))
        
        # forces
        R = len(dGi)
        f_output_errors = np.zeros(1)
        f_hidden2_errors = np.zeros(self.hidden2_nodes)
        f_hidden1_errors = np.zeros(self.hidden1_nodes)
        f_grad_output_cost = np.zeros((self.output_nodes, self.hidden2_nodes))
        f_grad_hidden2_cost = np.zeros((self.hidden2_nodes, self.hidden1_nodes))
        f_grad_hidden1_cost = np.zeros((self.hidden1_nodes, self.input_nodes))
        for r in range(R):
            f_output_error = F_errors[r]
            coef = np.dot(self.w[1], self.dif_activation_func(self.hidden1_inputs) * np.dot(self.w[0], dGi[r]))
            f_hidden2_error = self.dif_activation_func(self.hidden2_inputs) * np.dot(- self.w[2], (1 - 2 * self.hidden2_outputs) * coef) * f_output_errors
            f_hidden1_error = self.dif_activation_func(self.hidden1_inputs) * np.dot(self.w[1].T, f_hidden2_errors)
            
            f_output_errors += f_output_error
            f_hidden2_errors += f_hidden2_error
            f_hidden1_errors += f_hidden1_error
            f_grad_output_cost += np.dot(f_output_error.reshape((-1,1)), (- self.dif_activation_func(self.hidden2_inputs) * coef).reshape((1,-1)))
            f_grad_hidden2_cost += np.dot(f_hidden2_error.reshape((-1,1)), self.hidden1_outputs.reshape((1,-1)))
            f_grad_hidden1_cost += np.dot(f_hidden1_error.reshape((-1,1)), Gi.reshape((1,-1)))
        
        # modify weight parameters
        w_grad,b_grad = [],[]
        w_grad.append(self.learning_rate * (e_grad_hidden1_cost - self.beta * f_grad_hidden1_cost / R))
        w_grad.append(self.learning_rate * (e_grad_hidden2_cost - self.beta * f_grad_hidden2_cost / R))
        w_grad.append(self.learning_rate * (e_grad_output_cost - self.beta * f_grad_output_cost / R))
        b_grad.append(self.learning_rate * (e_hidden1_errors - self.beta * f_hidden1_errors / R))
        b_grad.append(self.learning_rate * (e_hidden2_errors - self.beta * f_hidden2_errors / R))
        b_grad.append(self.learning_rate * (e_output_errors - self.beta * f_output_errors / R))
        return w_grad, b_grad
    
    def __energy(self, Gi):
        # feed_forward
        self.hidden1_inputs = np.dot(self.w[0], Gi) + self.b[0]
        self.hidden1_outputs = self.activation_func(self.hidden1_inputs)
        
        self.hidden2_inputs = np.dot(self.w[1], self.hidden1_outputs) + self.b[1]
        self.hidden2_outputs = self.activation_func(self.hidden2_inputs)
        
        self.final_inputs = np.dot(self.w[2], self.hidden2_outputs) + self.b[2]
        final_outputs = self.final_inputs
        
        return final_outputs
    
    def __force(self, Gi, dGi):
        self.__energy(Gi)
        
        hidden1_outputs = np.dot(self.w[0], dGi.T)
        
        hidden2_inputs = self.dif_activation_func(self.hidden1_inputs) * hidden1_outputs
        hidden2_outputs = np.dot(self.w[1], hidden2_inputs)
        
        final_inputs = self.dif_activation_func(self.hidden2_inputs) * hidden2_outputs
        final_outputs = -1 * np.dot(self.w[2], final_inputs)
        
        return final_outputs.reshape(-1)

    # calculte energy
    ### input
    # Gi: numpy array (ninput)
    ### output
    # E: float
    def __query_E(self, Gi):
        Ei = self.__energy(Gi)
        E = np.zeros(1)
        self.comm.Allreduce(Ei, E, op=MPI.SUM)
        
        return E[0]

    # calculate force
    ### input
    # Gi: numpy array (ninput)
    # dGi: numpy array (3*natom x ninput)
    ### output
    # Fr: numpy array (3*natom)
    def __query_F(self, Gi, dGi):
        Fir = np.zeros(3*self.natom)
        for r in range(3*self.natom):
            Fir[r] = self.__force(Gi, dGi[r])
        Fr = np.zeros(3*self.natom)
        self.comm.Allreduce(Fir, Fr, op=MPI.SUM)
        
        return Fr
