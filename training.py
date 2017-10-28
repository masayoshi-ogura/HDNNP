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
import numpy as np

# import own modules
from modules.data import DataGenerator
from modules.model import HDNNP
from modules.util import mpimkdir
from modules.util import mpisave

stime = time()
datestr = datetime.now().strftime('%m%d-%H%M%S')
save_dir = path.join(file_.save_dir, datestr)
out_dir = path.join(file_.out_dir, datestr)
mpimkdir(save_dir)
mpimkdir(out_dir)
progress = MPI.File.Open(mpi.comm, path.join(out_dir, 'progress.dat'), MPI.MODE_CREATE | MPI.MODE_WRONLY)
progress.Write("""\
Rc:   {}
eta:  {}
Rs:   {}
lam:  {}
zeta: {}
learning_rate:       {}
learning_rate_decay: {}
mixing_beta:         {}
l1_norm:             {}
l2_norm:             {}
batch_size:          {}
optimizer:           {}\
""".format(','.join(map(str, hp.Rcs)), ','.join(map(str, hp.etas)), ','.join(map(str, hp.Rss)),
           ','.join(map(str, hp.lams)), ','.join(map(str, hp.zetas)),
           hp.learning_rate, hp.learning_rate_decay, hp.mixing_beta, hp.l1_norm, hp.l2_norm, hp.batch_size, hp.optimizer))

generator = DataGenerator('training', precond='pca')
for config, training_data, validation_data in generator:
    output_data = {
                   'training_energy': np.empty((hp.nepoch+1,) + training_data.label.shape),
                   'training_force': np.empty((hp.nepoch+1,) + training_data.dlabel.shape),
                   'validation_energy': np.empty((hp.nepoch+1,) + validation_data.label.shape),
                   'validation_force': np.empty((hp.nepoch+1,) + validation_data.dlabel.shape),
                   }
    output_data['training_energy'][0] = training_data.label
    output_data['training_force'][0] = training_data.dlabel
    output_data['validation_energy'][0] = validation_data.label
    output_data['validation_force'][0] = validation_data.dlabel

    natom = training_data.natom
    ninput = training_data.ninput
    nsample = training_data.nsample
    composition = training_data.composition
    progress.Write("""


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

    hdnnp = HDNNP(natom, ninput, composition)
    hdnnp.load(save_dir)

    for m, tr_result, val_result in hdnnp.fit(training_data, validation_data):
        tr_output, tr_doutput, tr_RMSE, tr_dRMSE, tr_tRMSE = tr_result
        val_output, val_doutput, val_RMSE, val_dRMSE, val_tRMSE = val_result
        progress.Write("""
{:<7} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f}\
""".format(m+1, time()-stime, tr_RMSE, tr_dRMSE, tr_tRMSE, val_RMSE, val_dRMSE, val_tRMSE))
        output_data['training_energy'][m+1] = tr_output
        output_data['training_force'][m+1] = tr_doutput
        output_data['validation_energy'][m+1] = val_output
        output_data['validation_force'][m+1] = val_doutput

    hdnnp.save(save_dir)
    np.savez(path.join(out_dir, '{}.npz'.format(config)), **output_data)
    mpi.comm.Barrier()
mpisave(generator, save_dir)
progress.Close()
