# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import file_
from config import mpi

# import python modules
from time import time
from os import path
from datetime import datetime

# import own modules
from modules.data import DataGenerator
from modules.model import HDNNP
from modules.animator import Animator

datestr = datetime.now().strftime('%m%d-%H%M%S')
generator = DataGenerator('training')
if mpi.rank == 0:
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

    for config, training_data, validation_data in generator:
        file.write("""

-------------------------{}-----------------------------

natom:         {}
composition:   {}
ninput:        {}
hidden_layers:
\t{}
nepoch:        {}
nsample:       {}

epoch   spent time        training_RMSE     training_dRMSE    training_tRMSE    validation_RMSE   validation_dRMSE  validation_tRMSE
""".format(config, training_data.natom, dict(training_data.composition['number']), training_data.ninput,
           '\n\t\t'.join(map(str, hp.hidden_layers)), hp.nepoch, training_data.nsample))
        file.flush()

        training_animator = Animator('training')
        validation_animator = Animator('validation')
        hdnnp = HDNNP(training_data.ninput, training_data.composition)
        hdnnp.initialize()
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
        mpi.comm.Barrier()
    file.close()
else:
    for config, training_data, validation_data in generator:
        hdnnp = HDNNP(training_data.ninput, training_data.composition)
        if hdnnp.initialize():
            hdnnp.load(datestr)
            for m, training_RMSE, validation_RMSE in hdnnp.fit(training_data, validation_data):
                pass
        mpi.comm.Barrier()
