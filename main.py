# -*- coding: utf-8 -*-

# define variables
import hyperparameters

# import python modules
import time
import os
from datetime import datetime
from mpi4py import MPI
import numpy as np
import random
if IMPORT_QUIPPY:
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
train_dir = 'training_data/'

# on root proc,
# 1. prepare output file
# 2. set variables unrelated to calculation
# 3. read training data set from .xyz and calculate symmetric functions, or read from .npy
# 4. write parameters to output file
if rank == 0:
    datestr = datetime.now().strftime('%m%d-%H%M%S')
    file = open('progress-'+datestr+'.out', 'w')
    stime = time.time()
    
    if LOAD_TRAINING_DATA:
        train_npy_dir = train_dir+'npy/'
        Es = np.load(train_npy_dir+name+'-Es.npy') # NSAMPLE
        Fs = np.load(train_npy_dir+name+'-Fs.npy') # NSAMPLE x 3*NATOM
        Gs = np.load(train_npy_dir+name+'-Gs.npy') # NSAMPLE X NATOM X NINPUT
        dGs = np.load(train_npy_dir+name+'-dGs.npy') # NSAMPLE x NATOM x 3*NATOM x NINPUT
        NSAMPLE = len(Es)
        NINPUT = len(Gs[0][0])
    else:
        train_xyz_dir = train_dir+'xyz/'
        train_npy_dir = train_dir+'npy/'
        alldataset = AtomsReader(train_xyz_dir+'AllSiGe.xyz')
        rawdataset = [data for data in alldataset if data.config_type == 'CrystalSi0Ge8' and data.cohesive_energy < 0.0]
        cordinates = [data for data in rawdataset]
        NSAMPLE = len(rawdataset)
        Es = np.array([data.cohesive_energy for data in rawdataset])
        Fs = np.array([np.array(data.force).T for data in rawdataset]).reshape((NSAMPLE,3*NATOM))
        a = cordinates[0].lattice[1][1]
        Rcs = [a]
        NINPUT = len(Rcs)*len(Rss)*len(etas)
        Gs,dGs = my_func.symmetric_func(cordinates, NATOM, NSAMPLE, NINPUT, Rcs, Rss, etas)
        file.write('Rc: '+','.join(map(str,Rcs))+'\n')
        file.write('Rs: '+','.join(map(str,Rss))+'\n')
        file.write('eta: '+','.join(map(str,etas))+'\n')
    file.write('NN_figure: '+str(NINPUT)+'x'+str(HIDDEN_NODES)+'x'+str(HIDDEN_NODES)+'x1\n')
    file.write('learning_rate: '+str(LEARNING)+'\n')
    file.write('beta: '+str(BETA)+'\n')
    file.write('gamma: '+str(GAMMA)+'\n')
    file.write('nepoch: '+str(NEPOCH)+'\n')
    file.write('data_num_of_subset: '+str(NSUBSET)+'\n\n')
    file.flush()

# broadcast training data set to other procs
NSAMPLE = comm.bcast(NSAMPLE, root=0)
NINPUT = comm.bcast(NINPUT, root=0)
if rank != 0:
    Es,Fs,Gs,dGs = np.empty(NSAMPLE),np.empty((NSAMPLE,3*NATOM)),np.empty((NSAMPLE,NATOM,NINPUT)),np.empty((NSAMPLE,NATOM,3*NATOM,NINPUT))
comm.Bcast(Es, root=0)
comm.Bcast(Fs, root=0)
comm.Bcast(Gs, root=0)
comm.Bcast(dGs, root=0)
dataset = [[Es[i],Fs[i],Gs[i],dGs[i]] for i in range(NSAMPLE)]

# initialize single NNP
nnp = hdnnp.single_nnp(NINPUT, HIDDEN_NODES, HIDDEN_NODES, 1, LEARNING, BETA, GAMMA, name)
# load weight parameters when restart
if LOAD_WEIGHT_PARAMS:
    nnp.load_w('weight_params/')
else:
    for i in range(3):
        comm.Bcast(nnp.w[i], root=0)
        comm.Bcast(nnp.b[i], root=0)

# training
for m in range(NEPOCH):
    subdataset = random.sample(dataset, NSUBSET)
    nnp.train(comm, rank, NATOM, NSUBSET, subdataset)
    if (m+1) % OUTPUT_INTERVAL == 0:
        E_RMSE,F_RMSE,RMSE = nnp.calc_RMSE(comm, rank, NATOM, NSAMPLE, dataset)
        if rank == 0:
            file.write('iteration: '+str(m+1)+'\n')
            file.write('energy RMSE: '+str(E_RMSE)+'\n')
            file.write('force RMSE: '+str(F_RMSE)+'\n')
            file.write('RMSE: '+str(RMSE)+'\n')
            file.write('spent time: '+str(time.time()-stime)+'\n')
            file.flush()

# save
if rank == 0:
    file.close()
    if SAVE_WEIGHT_PARAMS:
        weight_save_dir = weight_dir+datestr+'/'
        os.mkdir(weight_save_dir)
        nnp.save_w(weight_save_dir)
    if SAVE_TRAINING_DATA:
        train_save_dir = train_npy_dir+datestr+'/'
        os.mkdir(train_save_dir)
        np.save(train_save_dir+name+'-Es.npy', Es)
        np.save(train_save_dir+name+'-Fs.npy', Fs)
        np.save(train_save_dir+name+'-Gs.npy', Gs)
        np.save(train_save_dir+name+'-dGs.npy', dGs)
