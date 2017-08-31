# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import bool_
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

dataset, nsample, ninput = make_dataset(allcomm, allrank, allsize)

if allrank == 0:
    file.write("""
Rc:   {}
eta:  {}
Rs:   {}
lam:  {}
zeta: {}
NN_figure:     {}x{}x{}x{}
learning_rate: {}
beta:          {}
gamma:         {}
nepoch:        {}
nsample:       {}
ninput:        {}
data_num_of_subset: {}

iteration      spent time     energy RMSE    force RMSE     RMSE
""".format(','.join(map(str, hp.Rcs)), ','.join(map(str, hp.etas)), ','.join(map(str, hp.Rss)), ','.join(map(str, hp.lams)),
           ','.join(map(str, hp.zetas)), ninput, hp.hidden_nodes, hp.hidden_nodes, 1, hp.learning_rate, hp.beta, hp.gamma,
           hp.nepoch, nsample, ninput, hp.nsubset))
    file.flush()

# use only "natom" nodes for NN
allgroup = allcomm.Get_group()
NNcomm = allcomm.Create(allgroup.Incl(range(hp.natom)))
if allrank < hp.natom:
    NNrank = NNcomm.Get_rank()

    # initialize single NNP
    nnp = SingleNNP(NNcomm, NNrank, nsample, ninput)
    # load weight parameters when restart
    if bool_.LOAD_WEIGHT_PARAMS:
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
                file.write('{:<15}{:<15}{:<15}{:<15}{:<15}\n'.format(m+1, time()-stime, E_RMSE, F_RMSE, RMSE))
                file.flush()

    # save
    if allrank == 0:
        file.close()
        if bool_.SAVE_WEIGHT_PARAMS:
            weight_save_dir = path.join(weight_dir, datestr)
            mkdir(weight_save_dir)
            nnp.save_w(weight_save_dir, other.name)
