# -*- coding: utf-8 -*-

import time
from mpi4py import MPI
import numpy as np
import random

import hdnnp
import my_func

# set MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# on root proc, read data from file and calculate symmetric functions
if rank == 0:
    stime = time.time()
    Es = np.load('Ge-Es.npy')
    Fs = np.load('Ge-Fs.npy')
    Gs = np.load('Ge-Gs.npy')
    dGs = np.load('Ge-dGs.npy')
    nsample = 181
    natom = 8
    dataset = [[Es[i],Fs[i],Gs[i],dGs[i]] for i in range(nsample)]
else:
    nsample,natom,dataset = None,None,[None,np.empty((8,3)),np.empty(8),np.empty((8,8,3))] * 181

[nsample,natom,dataset] = comm.bcast([nsample,natom,dataset], root=0)

# initialize single NNP
learning = 0.1
nnp = hdnnp.single_nnp(1, 10, 10, 1, learning, name='Ge')
nnp.w[0] = comm.bcast(nnp.w[0], root=0)
nnp.w[1] = comm.bcast(nnp.w[1], root=0)
nnp.w[2] = comm.bcast(nnp.w[2], root=0)
nnp.b[0] = comm.bcast(nnp.b[0], root=0)
nnp.b[1] = comm.bcast(nnp.b[1], root=0)
nnp.b[2] = comm.bcast(nnp.b[2], root=0)
# ロードする場合
#nnp.load_w('weight_params/')

# training
# 重複ありで全データセットからランダムにsubnum個取り出し、それをサブセットとしてトレーニングする。
nepoch = 1000
# サブセット１つにデータをいくつ含めるか
subnum = 10
beta = 0.0
if rank == 0:
    print 'learning_rate: '+str(learning)
    print 'nepoch: '+str(nepoch)
    print 'data_num_of_subset: '+str(subnum)
    print 'beta: '+str(beta)+'\n'
for m in range(nepoch):
    subdataset = random.sample(dataset, subnum)
    hdnnp.train(comm, rank, nnp, natom, subnum, subdataset, beta)
#    if (m+1) % 10 == 0:
#        E_RMSE,F_RMSE = my_func.calc_RMSE(comm, rank, nnp, natom, nsample, dataset)
#        if rank == 0:
#            print 'iteration: '+str(m+1)
#            print E_RMSE
#            print F_RMSE

if rank == 0:
    E_RMSE,F_RMSE = my_func.calc_RMSE(comm, rank, nnp, natom, nsample, dataset)
    print 'iteration: '+str(m+1)
    print E_RMSE
    print F_RMSE
    etime = time.time()
    print '\nspent time: '+str(etime-stime)
    nnp.save_w('weight_params/')
