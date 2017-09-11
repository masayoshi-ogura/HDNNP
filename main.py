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

Es, Fs, Gs, dGs, natom, nsample, ninput, composition = make_dataset(allcomm, allrank, allsize)

if allrank == 0:
    file.write("""
Rc:   {}
eta:  {}
Rs:   {}
lam:  {}
zeta: {}
NN_figure:           {}x{}x{}
learning_rate:       {}
learning_rate_decay: {}
mixing_beta:         {}
momentum:            {}
adam_beta1:          {}
adam_beta2:          {}
epsilon:             {}
natom:               {}
composition:         {}
nepoch:              {}
nsample:             {}
ninput:              {}
batch_size:          {}
batch_size_growth:   {}
optimizer:           {}
activation:          {}

epoch          spent time     energy RMSE    force RMSE     RMSE
""".format(','.join(map(str, hp.Rcs)), ','.join(map(str, hp.etas)), ','.join(map(str, hp.Rss)),
           ','.join(map(str, hp.lams)), ','.join(map(str, hp.zetas)),
           ninput, 'x'.join(map(str, hp.hidden_layer)), 1,
           hp.learning_rate, hp.learning_rate_decay, hp.mixing_beta,
           hp.momentum, hp.adam_beta1, hp.adam_beta2, hp.epsilon,
           natom, dict(composition), hp.nepoch, nsample, ninput,
           hp.batch_size, hp.batch_size_growth, hp.optimizer, hp.activation))
    file.flush()

# use only "natom" nodes for NN
allgroup = allcomm.Get_group()
NNcomm = allcomm.Create(allgroup.Incl(range(natom)))
if allrank < natom:
    NNrank = NNcomm.Get_rank()

    # initialize HDNNP
    hdnnp = HDNNP(NNcomm, NNrank, natom, nsample, ninput, composition)
    # load weight parameters when restart
    if bool_.LOAD_WEIGHT_PARAMS:
        hdnnp.load_w()
    else:
        hdnnp.sync_w()

    # training
    for m in range(hp.nepoch):
        hdnnp.training(m, Es, Fs, Gs, dGs)
        E_RMSE, F_RMSE, RMSE = hdnnp.calc_RMSE(m, Es, Fs, Gs, dGs)
        if allrank == 0:
            file.write('{:<14} {:<14.9f} {:<14.9f} {:<14.9f} {:<14.9f}\n'.format(m+1, time()-stime, E_RMSE, F_RMSE, RMSE))
            file.flush()

    # save
    if allrank == 0:
        file.close()
        if bool_.SAVE_WEIGHT_PARAMS:
            hdnnp.save_w(datestr)
        if bool_.SAVE_FIG:
            hdnnp.save_fig()
