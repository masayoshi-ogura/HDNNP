# -*- coding: utf-8 -*-

# define variables
from config import hp, bool, other

# import own modules
from modules.input import Generator
from modules.model import SingleNNP

# import python modules
import time
from os import path, mkdir
from datetime import datetime
from mpi4py import MPI
import random
if bool.IMPORT_QUIPPY:
    from quippy import AtomsReader

# set MPI variables
allcomm = MPI.COMM_WORLD
allrank = allcomm.Get_rank()
allsize = allcomm.Get_size()

# set variables to all procs
weight_dir = 'weight_params'
train_dir = 'training_data'
train_xyz_file = path.join(train_dir, 'xyz', other.xyzfile)
train_npy_dir = path.join(train_dir, 'npy', other.name)
if not path.exists(train_npy_dir):
    mkdir(train_npy_dir)

generator = Generator(train_npy_dir, other.name, hp.Rcs, hp.etas, hp.Rss, hp.lams, hp.zetas)

if allrank == 0:
    datestr = datetime.now().strftime('%m%d-%H%M%S')
    file = open('progress-'+datestr+'.out', 'w')
    stime = time.time()

if bool.LOAD_TRAINING_XYZ_DATA:
    alldataset = AtomsReader(train_xyz_file)
    coordinates = []
    for data in alldataset:
        if data.config_type == other.name and data.cohesive_energy < 0.0:
            coordinates.append(data)
    hp.nsample = len(coordinates)
    Es, Fs = generator.calc_EF(coordinates, hp.natom, hp.nsample)
    hp.ninput = len(hp.Rcs) + \
        len(hp.Rcs)*len(hp.etas)*len(hp.Rss) + \
        len(hp.Rcs)*len(hp.etas)*len(hp.lams)*len(hp.zetas)
    Gs, dGs = generator.calc_G(allcomm, allsize, allrank, coordinates, hp.natom, hp.nsample, hp.ninput)
else:
    Es, Fs = generator.load_EF()
    Gs, dGs = generator.load_G()
    hp.nsample = len(Es)
    hp.ninput = len(Gs[0][0])
dataset = [[Es[i], Fs[i], Gs[i], dGs[i]] for i in range(hp.nsample)]

if allrank == 0:
    file.write('Rc:   '+','.join(map(str, hp.Rcs))+'\n')
    file.write('eta:  '+','.join(map(str, hp.etas))+'\n')
    file.write('Rs:   '+','.join(map(str, hp.Rss))+'\n')
    file.write('lam:  '+','.join(map(str, hp.lams))+'\n')
    file.write('zeta: '+','.join(map(str, hp.zetas))+'\n')
    file.write('NN_figure:     '+str(hp.ninput)+'x'+str(hp.hidden_nodes)+'x'+str(hp.hidden_nodes)+'x1\n')
    file.write('learning_rate: '+str(hp.learning_rate)+'\n')
    file.write('beta:    '+str(hp.beta)+'\n')
    file.write('gamma:   '+str(hp.gamma)+'\n')
    file.write('nepoch:  '+str(hp.nepoch)+'\n')
    file.write('nsample: '+str(hp.nsample)+'\n')
    file.write('data_num_of_subset: '+str(hp.nsubset)+'\n\n')
    file.write('iteration      spent time     energy RMSE    force RMSE     RMSE\n')
    file.flush()

# use only "natom" nodes for NN
allgroup = allcomm.Get_group()
NNcomm = allcomm.Create(allgroup.Incl(range(hp.natom)))
if allrank < hp.natom:
    NNrank = NNcomm.Get_rank()

    # initialize single NNP
    nnp = SingleNNP(NNcomm, NNrank, (hp.ninput, hp.hidden_nodes, hp.hidden_nodes, 1), hp.learning_rate, hp.beta, hp.gamma, hp.natom, hp.nsample)
    # load weight parameters when restart
    if bool.LOAD_WEIGHT_PARAMS:
        nnp.load_w(weight_dir, other.name)
    else:
        for i in range(3):
            NNcomm.Bcast(nnp.w[i], root=0)
            NNcomm.Bcast(nnp.b[i], root=0)

    # training
    for m in range(hp.nepoch):
        subdataset = random.sample(dataset, hp.nsubset)
        subdataset = NNcomm.bcast(subdataset, root=0)
        nnp.train(hp.nsubset, subdataset)
        if (m+1) % other.output_interval == 0:
            E_RMSE, F_RMSE, RMSE = nnp.calc_RMSE(dataset)
            if allrank == 0:
                file.write('%-15i%-15f%-15f%-15f%-15f\n' % (m+1, time.time()-stime, E_RMSE, F_RMSE, RMSE))
                file.flush()

    # save
    if allrank == 0:
        file.close()
        if bool.SAVE_WEIGHT_PARAMS:
            weight_save_dir = path.join(weight_dir, datestr)
            mkdir(weight_save_dir)
            nnp.save_w(weight_save_dir, other.name)
