# -*- coding: utf-8 -*-

import time
from datetime import datetime
from mpi4py import MPI
import numpy as np
import random
#from quippy import *

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
    ## read from xyz file
    ## for only quippy-available computer
    #datadir = '/home/ogura/m1/structure_data_SiGe/'
    #datafile = 'AllSiGe.xyz'
    #alldataset = AtomsReader(datadir + datafile)
    #rawdataset = [data for data in alldataset if data.config_type == 'CrystalSi0Ge8' and data.cohesive_energy < 0.0]
    #cordinates = [data for data in rawdataset]
    #nsample = len(rawdataset)
    #natom = 8
    #Es = np.array([data.cohesive_energy for data in rawdataset])
    #Fs = np.array([np.array(data.force).T for data in rawdataset]).reshape((nsample,3*natom))
    #Rcs = [cordinates[0].lattice[1][1]]
    #Rss = [1.0,2.0]
    #etas = [0.1,1.0]
    #gnum = len(Rcs)*len(Rss)*len(etas)
    #Gs,dGs = my_func.symmetric_func(cordinates, natom, nsample, gnum, Rcs, Rss, etas)
    
    ## read from numpy file
    Es = np.load('Ge-Es-Elt0.npy') # nsample
    Fs = np.load('Ge-Fs-Elt0.npy') # nsample x 3*natom
    Gs = np.load('Ge-Gs-Elt0-gnum4.npy') # nsample x natom x gnum
    dGs = np.load('Ge-dGs-Elt0-gnum4.npy') # nsample x natom x 3*natom x gnum
    nsample = len(Es)
    natom = len(Gs[0])
    gnum = len(Gs[0][0])
else:
    nsample,natom,gnum = None,None,None
[nsample,natom,gnum] = comm.bcast([nsample,natom,gnum], root=0)
if rank != 0:
    Es,Fs,Gs,dGs = np.empty(nsample),np.empty((nsample,3*natom)),np.empty((nsample,natom,gnum)),np.empty((nsample,natom,3*natom,gnum))
comm.Bcast(Es, root=0)
comm.Bcast(Fs, root=0)
comm.Bcast(Gs, root=0)
comm.Bcast(dGs, root=0)
dataset = [[Es[i],Fs[i],Gs[i],dGs[i]] for i in range(nsample)]

# initialize single NNP
learning = 0.001
beta = 0.5
gamma = 0.9
hidden_n = 3
nnp = hdnnp.single_nnp(gnum, hidden_n, hidden_n, 1, learning, beta, gamma, name='Ge')
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
nepoch = 8000
# サブセット１つにデータをいくつ含めるか
subnum = 10
if rank == 0:
    #file.write('Rc: '+','.join(map(str,Rcs)))
    #file.write('\n')
    #file.write('Rs: '+','.join(map(str,Rss)))
    #file.write('\n')
    #file.write('eta: '+','.join(map(str,etas)))
    #file.write('\n')
    file.write('NN_figure: '+str(gnum)+'x'+str(hidden_n)+'x'+str(hidden_n)+'x1')
    file.write('\n')
    file.write('learning_rate: '+str(learning))
    file.write('\n')
    file.write('nepoch: '+str(nepoch))
    file.write('\n')
    file.write('data_num_of_subset: '+str(subnum))
    file.write('\n')
    file.write('beta: '+str(beta))
    file.write('\n')
    file.write('gamma: '+str(gamma)+'\n')
    file.write('\n')
    file.flush()
for m in range(nepoch):
    subdataset = random.sample(dataset, subnum)
    nnp.train(comm, rank, natom, subnum, subdataset)
    if (m+1) % 10 == 0:
        E_RMSE,F_RMSE = nnp.calc_RMSE(comm, rank, natom, nsample, dataset)
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
    #np.save('Ge-Es.npy', Es)
    #np.save('Ge-Fs.npy', Fs)
    #np.save('Ge-Gs.npy', Gs)
    #np.save('Ge-dGs.npy', dGs)
