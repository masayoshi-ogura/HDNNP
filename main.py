# -*- coding: utf-8 -*-

# define variables
from config import hp,bool,other

# import python modules
import time
import os
from datetime import datetime
from mpi4py import MPI
import numpy as np
import random
if bool.IMPORT_QUIPPY:
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

if rank == 0:
    datestr = datetime.now().strftime('%m%d-%H%M%S')
    file = open('progress-'+datestr+'.out', 'w')
    stime = time.time()

if bool.LOAD_TRAINING_DATA:
    train_npy_dir = train_dir+'npy/'
    Es = np.load(train_npy_dir+other.name+'-Es.npy') # nsample
    Fs = np.load(train_npy_dir+other.name+'-Fs.npy') # nsample x 3*natom
    Gs = np.load(train_npy_dir+other.name+'-Gs.npy') # nsample x natom x ninput
    dGs = np.load(train_npy_dir+other.name+'-dGs.npy') # nsample x natom x 3*natom x ninput
    hp.nsample = len(Es)
    hp.ninput = len(Gs[0][0])
else:
    train_xyz_dir = train_dir+'xyz/'
    train_npy_dir = train_dir+'npy/'
    alldataset = AtomsReader(train_xyz_dir+'AllSiGe.xyz')
    rawdataset = [data for data in alldataset if data.config_type == 'CrystalSi0Ge8' and data.cohesive_energy < 0.0]
    cordinates = [data for data in rawdataset]
    hp.nsample = len(rawdataset)
    Es = np.array([data.cohesive_energy for data in rawdataset])
    Fs = np.array([np.array(data.force).T for data in rawdataset]).reshape((hp.nsample,3*hp.natom))
    hp.ninput = len(hp.Rcs) + len(hp.Rcs)*len(hp.Rss)*len(hp.etas) + len(hp.Rcs)*len(hp.etas)*len(hp.lams)*len(hp.zetas)
    Gs,dGs = my_func.symmetric_func(comm, rank, cordinates, hp.natom, hp.nsample, hp.ninput, hp.Rcs, hp.Rss, hp.etas, hp.lams, hp.zetas)
dataset = [[Es[i],Fs[i],Gs[i],dGs[i]] for i in range(hp.nsample)]

if rank == 0:
    if not bool.LOAD_TRAINING_DATA:
        file.write('Rc: '+','.join(map(str,hp.Rcs))+'\n')
        file.write('Rs: '+','.join(map(str,hp.Rss))+'\n')
        file.write('eta: '+','.join(map(str,hp.etas))+'\n')
        file.write('lam: '+','.join(map(str,hp.lams))+'\n')
        file.write('zeta: '+','.join(map(str,hp.zetas))+'\n')
    file.write('NN_figure: '+str(hp.ninput)+'x'+str(hp.hidden_nodes)+'x'+str(hp.hidden_nodes)+'x1\n')
    file.write('learning_rate: '+str(hp.learning_rate)+'\n')
    file.write('beta: '+str(hp.beta)+'\n')
    file.write('gamma: '+str(hp.gamma)+'\n')
    file.write('nepoch: '+str(hp.nepoch)+'\n')
    file.write('data_num_of_subset: '+str(hp.nsubset)+'\n\n')
    file.write('iteration      spent time     energy RMSE    force RMSE     RMSE\n')
    file.flush()

# initialize single NNP
nnp = hdnnp.single_nnp(hp.ninput, hp.hidden_nodes, hp.hidden_nodes, 1, hp.learning_rate, hp.beta, hp.gamma)
# load weight parameters when restart
if bool.LOAD_WEIGHT_PARAMS:
    nnp.load_w(weight_dir, other.name)
else:
    for i in range(3):
        comm.Bcast(nnp.w[i], root=0)
        comm.Bcast(nnp.b[i], root=0)

# training
for m in range(hp.nepoch):
    subdataset = random.sample(dataset, hp.nsubset)
    subdataset = comm.bcast(subdataset, root=0)
    nnp.train(comm, rank, hp.natom, hp.nsubset, subdataset)
    if (m+1) % other.output_interval == 0:
        E_RMSE,F_RMSE,RMSE = nnp.calc_RMSE(comm, rank, hp.natom, hp.nsample, dataset, hp.beta)
        if rank == 0:
            file.write('%-15i%-15f%-15f%-15f%-15f\n' % (m+1, time.time()-stime, E_RMSE, F_RMSE, RMSE))
            file.flush()

# save
if rank == 0:
    file.close()
    if bool.SAVE_WEIGHT_PARAMS:
        weight_save_dir = weight_dir+datestr+'/'
        os.mkdir(weight_save_dir)
        nnp.save_w(weight_save_dir, other.name)
    if bool.SAVE_TRAINING_DATA:
        train_save_dir = train_npy_dir+datestr+'/'
        os.mkdir(train_save_dir)
        np.save(train_save_dir+other.name+'-Es.npy', Es)
        np.save(train_save_dir+other.name+'-Fs.npy', Fs)
        np.save(train_save_dir+other.name+'-Gs.npy', Gs)
        np.save(train_save_dir+other.name+'-dGs.npy', dGs)
