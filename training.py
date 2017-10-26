# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import file_
from config import mpi

# import python modules
from time import time
from os import path
from datetime import datetime
from mpi4py import MPI

# import own modules
from modules.data import DataGenerator
from modules.model import HDNNP
from modules.animator import Animator
from modules.util import mpimkdir
from modules.util import mpisave

stime = time()
datestr = datetime.now().strftime('%m%d-%H%M%S')
save_dir = path.join(file_.save_dir, datestr)
mpimkdir(save_dir)
file = MPI.File.Open(mpi.comm, path.join(file_.progress_dir, 'progress-{}.out'.format(datestr)), MPI.MODE_CREATE | MPI.MODE_WRONLY)
file.Write("""\
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
optimizer:           {}\
""".format(','.join(map(str, hp.Rcs)), ','.join(map(str, hp.etas)), ','.join(map(str, hp.Rss)),
           ','.join(map(str, hp.lams)), ','.join(map(str, hp.zetas)),
           hp.learning_rate, hp.learning_rate_decay, hp.mixing_beta, hp.smooth_factor,
           hp.batch_size, hp.batch_size_growth, hp.optimizer))
file.Sync()

generator = DataGenerator('training', precond='pca')
for config, training_data, validation_data in generator:
    natom = training_data.natom
    ninput = training_data.ninput
    nsample = training_data.nsample
    composition = training_data.composition
    file.Write("""


-------------------------{}-----------------------------

composition:   {}
natom:         {}
ninput:        {}
hidden_layers:
\t\t{}
nepoch:        {}
nsample:       {}

epoch   spent time        training_RMSE     training_dRMSE    training_tRMSE    validation_RMSE   validation_dRMSE  validation_tRMSE\
""".format(config, dict(composition['number']), natom, ninput,
           '\n\t\t'.join(map(str, hp.hidden_layers)), hp.nepoch, nsample))
    file.Sync()

    training_animator = Animator()
    validation_animator = Animator()
    hdnnp = HDNNP(natom, ninput, composition)
    hdnnp.load(save_dir)

    for m, training_RMSE, validation_RMSE in hdnnp.fit(training_data, validation_data, training_animator, validation_animator):
        t_RMSE, t_dRMSE, t_tRMSE = training_RMSE
        v_RMSE, v_dRMSE, v_tRMSE = validation_RMSE
        file.Write("""
{:<7} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f}\
""".format(m+1, time()-stime, t_RMSE, t_dRMSE, t_tRMSE, v_RMSE, v_dRMSE, v_tRMSE))
        file.Sync()

    mpisave(training_animator, datestr, config, 'training')
    mpisave(validation_animator, datestr, config, 'validation')
    hdnnp.save(save_dir)
    mpi.comm.Barrier()
mpisave(generator, save_dir)
file.Close()
