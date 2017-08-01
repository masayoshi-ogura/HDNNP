import numpy as np
from quippy import AtomsReader
import matplotlib.pyplot as plt
import scipy.special as sp
import math
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

Gs = np.load('training_data/npy/0802-000035/Ge-Gs.npy')
Es = np.load('training_data/npy/0802-000035/Ge-Es.npy')

class single_nnp:
    def __init__(self, input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate):
        # set number of nodes of each layers and learning rate
        # beta: mixing parameter of error between energy and force
        # gamma: NAG parameter. rate of accumulation
        self.input_nodes = input_nodes
        self.hidden1_nodes = hidden1_nodes
        self.hidden2_nodes = hidden2_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        
        # initialize weight parameters
        self.w = []
        self.b = []
        # input-hidden1
        self.w.append(np.random.normal(0.0, 0.5, (hidden1_nodes, input_nodes)))
        self.b.append(np.random.normal(0.0, 0.5, (hidden1_nodes)))
        # hidden1-hidden2
        self.w.append(np.random.normal(0.0, 0.5, (hidden2_nodes, hidden1_nodes)))
        self.b.append(np.random.normal(0.0, 0.5, (hidden2_nodes)))
        # hidden2-output
        self.w.append(np.random.normal(0.0, 1.0, (output_nodes, hidden2_nodes)))
        self.b.append(np.random.normal(0.0, 1.0, (output_nodes)))
        
        # define activation function and derivative
        self.activation_func = lambda x: sp.expit(x)
        self.dif_activation_func = lambda x: sp.expit(x) * (1 - sp.expit(x))
    
    def gradient(self, Gi, E_error):
        # feed_forward
        self.energy(Gi)
        
        # back_prop
        # energy
        e_output_errors = np.array([E_error])
        e_hidden2_errors = self.dif_activation_func(self.hidden2_inputs) * np.dot(self.w[2].T, e_output_errors)
        e_hidden1_errors = self.dif_activation_func(self.hidden1_inputs) * np.dot(self.w[1].T, e_hidden2_errors)
        
        e_grad_output_cost = np.dot(e_output_errors.reshape((-1,1)), self.hidden2_outputs.reshape((1,-1)))
        e_grad_hidden2_cost = np.dot(e_hidden2_errors.reshape((-1,1)), self.hidden1_outputs.reshape((1,-1)))
        e_grad_hidden1_cost = np.dot(e_hidden1_errors.reshape((-1,1)), Gi.reshape((1,-1)))
        
        
        # modify weight parameters
        w_grad,b_grad = [],[]
        w_grad.append(self.learning_rate * (e_grad_hidden1_cost))
        w_grad.append(self.learning_rate * (e_grad_hidden2_cost ))
        w_grad.append(self.learning_rate * (e_grad_output_cost))
        b_grad.append(self.learning_rate * (e_hidden1_errors ))
        b_grad.append(self.learning_rate * (e_hidden2_errors ))
        b_grad.append(self.learning_rate * (e_output_errors ))
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

    def query_E(self, comm, Gi):
        E = np.zeros(1)
        Ei = self.energy(Gi)
        comm.Allreduce(Ei,E,op=MPI.SUM)
        
        return E[0]

input_n = 2
hidden_n = 10
output_n = 1
learning = 0.01
natom = 8
nsample = 203

nn_ge = single_nnp(input_n, hidden_n, hidden_n, output_n, learning)
for k in range(3):
    comm.Bcast(nn_ge.w[k], root=0)
    comm.Bcast(nn_ge.b[k], root=0)

# training
import random
nloop = 10000

for hoge in range(10):
    for n in range(nloop):
        i = random.sample(range(nsample), 1)[0]
        i = comm.bcast(i, root=0)
        total = nn_ge.query_E(comm, Gs[i][rank])
        
        error = Es[i] - total
        w_grad = [np.zeros((10,2)),np.zeros((10,10)),np.zeros((1,10))]
        b_grad = [np.zeros(10),np.zeros(10),np.zeros(1)]
        w_grad_ser = [np.zeros((10,2)),np.zeros((10,10)),np.zeros((1,10))]
        b_grad_ser = [np.zeros(10),np.zeros(10),np.zeros(1)]
        # serial
        for j in range(natom):
            if hoge == 0 and n == 0 and j == 0:
                print rank,i,Gs[i],error/natom
            w_ser,b_ser = nn_ge.gradient(Gs[i][j], error / natom)
            for k in range(3):
                w_grad_ser[k] += w_ser[k]
                b_grad_ser[k] += b_ser[k]
        # parallel
        w,b = nn_ge.gradient(Gs[i][rank], error/natom)
        for k in range(3):
            comm.Allreduce(w[k], w_grad[k], op=MPI.SUM)
            comm.Allreduce(b[k], b_grad[k], op=MPI.SUM)
            nn_ge.w[k] += w_grad[k] / natom
            nn_ge.b[k] += b_grad[k] / natom

    MSE = 0.0
    for i in range(nsample):
        Eout = nn_ge.query_E(comm, Gs[i][rank])
        MSE += (Es[i] - Eout) ** 2
    RMSE = math.sqrt(MSE / nsample)
    if rank == 0:
        print RMSE
