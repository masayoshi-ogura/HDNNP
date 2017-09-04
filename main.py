# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import bool_

# import python modules
from time import time
from datetime import datetime
from mpi4py import MPI

# import own modules
from modules.generator import make_dataset
from modules.model import HDNNP

allcomm = MPI.COMM_WORLD
allrank = allcomm.Get_rank()
allsize = allcomm.Get_size()

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
NN_figure:     {}x{}x{}
learning_rate: {}
beta:          {}
momentum1:     {}
momentum2:     {}
nepoch:        {}
nsample:       {}
ninput:        {}
batch_size:    {}
optimizer:     {}
activation:    {}

epoch          spent time     energy RMSE    force RMSE     RMSE
""".format(','.join(map(str, hp.Rcs)), ','.join(map(str, hp.etas)), ','.join(map(str, hp.Rss)),
           ','.join(map(str, hp.lams)), ','.join(map(str, hp.zetas)),
           ninput, 'x'.join(map(str, hp.hidden_layer)), 1,
           hp.learning_rate, hp.beta, hp.momentum1, hp.momentum2,
           hp.nepoch, nsample, ninput, hp.batch_size,
           hp.optimizer, hp.activation))
    file.flush()

# use only "natom" nodes for NN
allgroup = allcomm.Get_group()
NNcomm = allcomm.Create(allgroup.Incl(range(hp.natom)))
if allrank < hp.natom:
    NNrank = NNcomm.Get_rank()

    # initialize HDNNP
    hdnnp = HDNNP(NNcomm, NNrank, nsample, ninput)
    # load weight parameters when restart
    if bool_.LOAD_WEIGHT_PARAMS:
        hdnnp.load_w()
    else:
        hdnnp.sync_w()

    # training
    for m in range(hp.nepoch):
        hdnnp.training(dataset)
        # E_RMSE, F_RMSE, RMSE = hdnnp.calc_RMSE(dataset)
        E_RMSE, F_RMSE, RMSE, plt = hdnnp.calc_RMSE(dataset)  # debug
        plt.savefig('F0_RMSE_{}.png'.format(m+1))  # debug
        if allrank == 0:
            file.write('{:<15}{:<15}{:<15}{:<15}{:<15}\n'.format(m+1, time()-stime, E_RMSE, F_RMSE, RMSE))
            file.flush()

    # save
    if allrank == 0:
        file.close()
        if bool_.SAVE_WEIGHT_PARAMS:
            hdnnp.save_w(datestr)
