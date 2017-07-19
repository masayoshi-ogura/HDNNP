# -*- coding: utf-8 -*-

import time
from datetime import datetime
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
    file = open('progress'+str(datetime.now().time())+'.out', 'w')
    stime = time.time()
    Es = np.load('Ge-Es.npy') # nsample
    Fs = np.load('Ge-Fs.npy') # nsample x 3*natom
    Gs = np.load('Ge-Gs.npy') # nsample x natom x gnum
    dGs = np.load('Ge-dGs.npy') # nsample x natom x 3*natom x gnum
    nsample = 181
    natom = 8
    gnum = len(Rcs)*len(Rss)*len(etas)
else:
    nsample,natom,gnum = None,None,None
[nsample,natom,gnum] = comm.bcast([nsample,natom,gnum], root=0)
if rank != 0:
    Es,Fs,Gs,dGs = np.empty(nsample),np.empty((nsample,3*natom)),np.empty(nsample,natom,gnum),np.empty((nsample,natom,3*natom,gnum))
comm.Bcast(Es, root=0)
comm.Bcast(Fs, root=0)
comm.Bcast(Gs, root=0)
comm.Bcast(dGs, root=0)
dataset = [[Es[i],Fs[i],Gs[i],dGs[i]] for i in range(nsample)]

# initialize single NNP
learning = 0.1
nnp = hdnnp.single_nnp(gnum, 10, 10, 1, learning, name='Ge')
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
nepoch = 30000
# サブセット１つにデータをいくつ含めるか
subnum = 10
beta = 0.5
if rank == 0:
    file.write('learning_rate: '+str(learning))
    file.write('\n')
    file.write('nepoch: '+str(nepoch))
    file.write('\n')
    file.write('data_num_of_subset: '+str(subnum))
    file.write('\n')
    file.write('beta: '+str(beta)+'\n')
    file.write('\n')
for m in range(nepoch):
    subdataset = random.sample(dataset, subnum)
    hdnnp.train(comm, rank, nnp, natom, subnum, subdataset, beta)
    if (m+1) % 1000 == 0:
        E_RMSE,F_RMSE = my_func.calc_RMSE(comm, rank, nnp, natom, nsample, dataset)
        if rank == 0:
            file.write('iteration: '+str(m+1))
            file.write('\n')
            file.write('energy RMSE: '+str(E_RMSE))
            file.write('\n')
            file.write('force RMSE: '+str(F_RMSE))
            file.write('\n')
            file.write('spent time: '+str(time.time()-stime))
            file.write('\n')
            file.flush()

if rank == 0:
    nnp.save_w('weight_params/')
    file.close()
