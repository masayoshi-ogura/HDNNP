# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import bool
from config import other

# import python modules
from os import path
from os import mkdir
from time import time
from datetime import datetime
from random import sample
from mpi4py import MPI

# import own modules
from modules.generator import make_dataset
from modules.model import SingleNNP

allcomm = MPI.COMM_WORLD
allrank = allcomm.Get_rank()
allsize = allcomm.Get_size()

weight_dir = 'weight_params'

if allrank == 0:
    datestr = datetime.now().strftime('%m%d-%H%M%S')
    file = open('progress-'+datestr+'.out', 'w')
    stime = time()

dataset, hp.nsample, hp.ninput = make_dataset(allcomm, allrank, allsize)

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
    file.write('ninput:  '+str(hp.ninput)+'\n')
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
        subdataset = sample(dataset, hp.nsubset)
        subdataset = NNcomm.bcast(subdataset, root=0)
        nnp.train(hp.nsubset, subdataset)
        if (m+1) % other.output_interval == 0:
            E_RMSE, F_RMSE, RMSE = nnp.calc_RMSE(dataset)
            if allrank == 0:
                file.write('%-15i%-15f%-15f%-15f%-15f\n' % (m+1, time()-stime, E_RMSE, F_RMSE, RMSE))
                file.flush()

    # save
    if allrank == 0:
        file.close()
        if bool.SAVE_WEIGHT_PARAMS:
            weight_save_dir = path.join(weight_dir, datestr)
            mkdir(weight_save_dir)
            nnp.save_w(weight_save_dir, other.name)
