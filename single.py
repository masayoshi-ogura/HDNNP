# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import file_

# import python modules
from sys import argv
from time import time
from os import path
from os import makedirs
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# import own modules
from modules.data import FunctionData
from modules.model import SingleNNP

stime = time()
datestr = datetime.now().strftime('%m%d-%H%M%S')
save_dir = path.join(file_.save_dir, datestr)
out_dir = path.join(file_.out_dir, datestr)
makedirs(save_dir)
makedirs(out_dir)
progress = open(path.join(out_dir, 'progress.dat'), 'w')
progress.write("""\
learning_rate:       {}
learning_rate_decay: {}
mixing_beta:         {}
smooth_factor:       {}
batch_size:          {}
batch_size_growth:   {}
optimizer:           {}\
""".format(hp.learning_rate, hp.learning_rate_decay, hp.mixing_beta, hp.smooth_factor,
           hp.batch_size, hp.batch_size_growth, hp.optimizer))

training_data = FunctionData(argv[1], 'training')
validation_data = FunctionData(argv[1], 'validation')
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

progress.write("""


-------------------{}-----------------------

ninput:        {}
hidden_layers:
\t\t{}
nepoch:        {}
nsample:       {}

epoch   spent time        training_RMSE     training_dRMSE    training_tRMSE    validation_RMSE   validation_dRMSE  validation_tRMSE\
""".format(argv[1], training_data.ninput, '\n\t\t'.join(map(str, hp.hidden_layers)), hp.nepoch, training_data.nsample))

# initialize NNP
nnp = SingleNNP(training_data.ninput)
nnp.load(save_dir)

for m, tr_result, val_result in nnp.fit(training_data, validation_data):
    tr_output, tr_doutput, tr_RMSE, tr_dRMSE, tr_tRMSE = tr_result
    val_output, val_doutput, val_RMSE, val_dRMSE, val_tRMSE = val_result
    progress.write("""
{:<7} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f}\
""".format(m+1, time()-stime, tr_RMSE, tr_dRMSE, tr_tRMSE, val_RMSE, val_dRMSE, val_tRMSE))
    output_data['training_energy'][m+1] = tr_output
    output_data['training_force'][m+1] = tr_doutput
    output_data['validation_energy'][m+1] = val_output
    output_data['validation_force'][m+1] = val_doutput

nnp.save(save_dir)
np.savez(path.join(out_dir, '{}.npz'.format(argv[1])), **output_data)
progress.close()
