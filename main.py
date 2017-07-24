# -*- coding: utf-8 -*-

# set computer name
from os import uname
from re import match
if match(r'Masayoshi', uname()[1]):
    cname = 'local'
elif match(r'forte', uname()[1]):
    cname = 'forte'
elif match(r'iris', uname()[1]):
    cname = 'iris'

# import python modules
import time
from datetime import datetime
from mpi4py import MPI
import numpy as np
import random
if not cname == 'forte':
    from quippy import AtomsReader

# import own modules
import hdnnp
import my_func

# set MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set variables to all procs
weight_dir = 'weight_params/'
learning = 0.001
beta = 0.5
gamma = 0.9
hidden_n = 3
nepoch = 10000
subnum = 10
output_interval = 100
name = 'Ge'

# on root proc,
# 1. prepare output file
# 2. set variables unrelated to calculation
# 3. read training data set from .xyz and calculate symmetric functions, or read from .npy
# 4. write parameters to output file
if rank == 0:
    now = datetime.now()
    file = open('progress'+now.strftime('-%m%d-%H%M%S')+'.out', 'w')
    stime = time.time()
    
    if cname == 'forte':
        train_dir = 'training_data/npy/'
        Es = np.load(train_dir+name+'-Es.npy') # nsample
        Fs = np.load(train_dir+name+'-Fs.npy') # nsample x 3*natom
        Gs = np.load(train_dir+name+'-Gs.npy') # nsample x natom x gnum
        dGs = np.load(train_dir+name+'-dGs.npy') # nsample x natom x 3*natom x gnum
        nsample = len(Es)
        natom = len(Gs[0])
        gnum = len(Gs[0][0])
    else:
        train_dir = 'training_data/xyz/'
        alldataset = AtomsReader(datadir+'AllSiGe.xyz')
        rawdataset = [data for data in alldataset if data.config_type == 'CrystalSi0Ge8' and data.cohesive_energy < 0.0]
        cordinates = [data for data in rawdataset]
        nsample = len(rawdataset)
        natom = 8
        Es = np.array([data.cohesive_energy for data in rawdataset])
        Fs = np.array([np.array(data.force).T for data in rawdataset]).reshape((nsample,3*natom))
        a = cordinates[0].lattice[1][1]
        Rcs = [a]
        Rss = [1.0]
        etas = [0.0]
        gnum = len(Rcs)*len(Rss)*len(etas)
        Gs,dGs = my_func.symmetric_func(cordinates, natom, nsample, gnum, Rcs, Rss, etas)
        file.write('Rc: '+','.join(map(str,Rcs))+'\n')
        file.write('Rs: '+','.join(map(str,Rss))+'\n')
        file.write('eta: '+','.join(map(str,etas))+'\n')
    file.write('NN_figure: '+str(gnum)+'x'+str(hidden_n)+'x'+str(hidden_n)+'x1\n')
    file.write('learning_rate: '+str(learning)+'\n')
    file.write('beta: '+str(beta)+'\n')
    file.write('gamma: '+str(gamma)+'\n')
    file.write('nepoch: '+str(nepoch)+'\n')
    file.write('data_num_of_subset: '+str(subnum)+'\n\n')
    file.flush()
else:
    nsample,natom,gnum = None,None,None
# broadcast training data set to other procs
[nsample,natom,gnum] = comm.bcast([nsample,natom,gnum], root=0)
if rank != 0:
    Es,Fs,Gs,dGs = np.empty(nsample),np.empty((nsample,3*natom)),np.empty((nsample,natom,gnum)),np.empty((nsample,natom,3*natom,gnum))
comm.Bcast(Es, root=0)
comm.Bcast(Fs, root=0)
comm.Bcast(Gs, root=0)
comm.Bcast(dGs, root=0)
dataset = [[Es[i],Fs[i],Gs[i],dGs[i]] for i in range(nsample)]

# initialize single NNP
nnp = hdnnp.single_nnp(gnum, hidden_n, hidden_n, 1, learning, beta, gamma, name)
for i in range(3):
    nnp.w[i] = comm.bcast(nnp.w[i], root=0)
    nnp.b[i] = comm.bcast(nnp.b[i], root=0)
# load weight parameters when restart
#nnp.load_w('weight_params/')

# training
for m in range(nepoch):
    subdataset = random.sample(dataset, subnum)
    nnp.train(comm, rank, natom, subnum, subdataset)
    if (m+1) % output_interval == 0:
        E_RMSE,F_RMSE,RMSE = nnp.calc_RMSE(comm, rank, natom, nsample, dataset)
        if rank == 0:
            file.write('iteration: '+str(m+1)+'\n')
            file.write('energy RMSE: '+str(E_RMSE)+'\n')
            file.write('force RMSE: '+str(F_RMSE)+'\n')
            file.write('RMSE: '+str(RMSE)+'\n')
            file.write('spent time: '+str(time.time()-stime)+'\n')
            file.flush()

# save
if rank == 0:
    nnp.save_w(weight_dir)
    file.close()
    if not cname == 'forte':
        np.save(train_dir+name+'-Es.npy', Es)
        np.save(train_dir+name+'-Fs.npy', Fs)
        np.save(train_dir+name+'-Gs.npy', Gs)
        np.save(train_dir+name+'-dGs.npy', dGs)
