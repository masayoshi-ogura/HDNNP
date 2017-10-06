# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import file_

# import python modules
from time import time
from os import path
from datetime import datetime
from mpi4py import MPI

# import own modules
from modules.generator import make_dataset
from modules.model import HDNNP
from modules.animator import Animator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

datestr = datetime.now().strftime('%m%d-%H%M%S')
if rank == 0:
    file = open(path.join(file_.progress_dir, 'progress-{}.out'.format(datestr)), 'w')
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

    for ret in make_dataset(comm, rank, size, 'training'):
        config, training_data, validation_data, natom, nsample, ninput, composition = ret

        file.write("""

-------------------------{}-----------------------------

natom:         {}
composition:   {}
ninput:        {}
hidden_layers:
\t{}
nepoch:        {}
nsample:       {}

epoch          spent time     energy RMSE    force RMSE     RMSE
""".format(config, natom, dict(composition['number']), ninput,
           '\n\t\t'.join(map(str, hp.model)), hp.nepoch, nsample))
        file.flush()

        training_animator = Animator('training')
        validation_animator = Animator('validation')
        hdnnp = HDNNP(ninput)
        hdnnp.initialize(comm, rank, size, ninput, composition)
        hdnnp.load(datestr)

        for m, training_RMSE, validation_RMSE in hdnnp.fit(training_data, validation_data, training_animator, validation_animator):
            t_RMSE, t_dRMSE, t_tRMSE = training_RMSE
            v_RMSE, v_dRMSE, v_tRMSE = validation_RMSE
            file.write('{:<7} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f}\n'
                       .format(m+1, time()-stime, t_RMSE, t_dRMSE, t_tRMSE, v_RMSE, v_dRMSE, v_tRMSE))
            file.flush()

        training_animator.save_fig(datestr, config)
        validation_animator.save_fig(datestr, config)
        hdnnp.save(datestr)
        comm.Barrier()
    file.close()
else:
    for ret in make_dataset(comm, rank, size, 'training'):
        config, training_data, validation_data, natom, nsample, ninput, composition = ret

        hdnnp = HDNNP(ninput)
        if hdnnp.initialize(comm, rank, size, ninput, composition):
            hdnnp.load(datestr)

            for m, training_RMSE, validation_RMSE in hdnnp.fit(training_data, validation_data):
                t_RMSE, t_dRMSE, t_tRMSE = training_RMSE
                v_RMSE, v_dRMSE, v_tRMSE = validation_RMSE
        comm.Barrier()
