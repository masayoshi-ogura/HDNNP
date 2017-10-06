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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

datestr = datetime.now().strftime('%m%d-%H%M%S')
if rank == 0:
    file = open('progress-'+datestr+'.out', 'w')
    stime = time()
    file.write("""
Rc:   {}
eta:  {}
Rs:   {}
lam:  {}
zeta: {}
learning_rate:       {}
learning_rate_decay: {}
mixing_beta:         {}
smooth_factor:       {}
batch_size:          {}
batch_size_growth:   {}
optimizer:           {}
""".format(','.join(map(str, hp.Rcs)), ','.join(map(str, hp.etas)), ','.join(map(str, hp.Rss)),
           ','.join(map(str, hp.lams)), ','.join(map(str, hp.zetas)),
           hp.learning_rate, hp.learning_rate_decay, hp.mixing_beta, hp.smooth_factor,
           hp.batch_size, hp.batch_size_growth, hp.optimizer))
    file.flush()

for ret in make_dataset(comm, rank, size, 'train'):
    config, Es, Fs, Gs, dGs, natom, nsample, ninput, composition = ret

    if rank == 0:
        file.write("""

-------------------------{}-----------------------------

natom:         {}
composition:   {}
ninput:        {}
hidden_layers: {}
nepoch:        {}
nsample:       {}

epoch          spent time     energy RMSE    force RMSE     RMSE
""".format(config, natom, dict(composition['number']), ninput,
           '\n\t\t'.join(map(str, hp.model)), hp.nepoch, nsample))
        file.flush()

    # initialize HDNNP
    hdnnp = HDNNP(natom, nsample)
    # if size > natom, unnnecessary node return False and do nothing.
    if hdnnp.initialize(comm, rank, size, ninput, composition):
        hdnnp.load_w(datestr)

        # training
        for m in range(hp.nepoch):
            hdnnp.training(m, Es, Fs, Gs, dGs)
            E_RMSE, F_RMSE, RMSE = hdnnp.calc_RMSE(m, Es, Fs, Gs, dGs)
            if rank == 0:
                file.write('{:<14} {:<14.9f} {:<14.9f} {:<14.9f} {:<14.9f}\n'.format(m+1, time()-stime, E_RMSE, F_RMSE, RMSE))
                file.flush()

        # save
        if bool_.SAVE_FIG:
            hdnnp.save_fig(datestr, config, 'gif')
        hdnnp.save_w(datestr)
    comm.Barrier()
if rank == 0:
    file.close()
