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

# import own modules
from modules.data import FunctionData
from modules.model import SingleNNP
from modules.animator import Animator

datestr = datetime.now().strftime('%m%d-%H%M%S')
save_dir = path.join(file_.save_dir, datestr)
makedirs(save_dir)
file = open(path.join(file_.progress_dir, 'progress-{}.out'.format(datestr)), 'w')
stime = time()
file.write("""\
learning_rate:       {}
learning_rate_decay: {}
mixing_beta:         {}
smooth_factor:       {}
batch_size:          {}
batch_size_growth:   {}
optimizer:           {}\
""".format(hp.learning_rate, hp.learning_rate_decay, hp.mixing_beta, hp.smooth_factor,
           hp.batch_size, hp.batch_size_growth, hp.optimizer))
file.flush()

training_data = FunctionData(argv[1], 'training')
validation_data = FunctionData(argv[1], 'validation')

file.write("""


-------------------{}-----------------------

ninput:        {}
hidden_layers:
\t\t{}
nepoch:        {}
nsample:       {}

epoch   spent time        training_RMSE     training_dRMSE    training_tRMSE    validation_RMSE   validation_dRMSE  validation_tRMSE\
""".format(argv[1], training_data.ninput, '\n\t\t'.join(map(str, hp.hidden_layers)), hp.nepoch, training_data.nsample))
file.flush()

# initialize NNP
training_animator = Animator()
validation_animator = Animator()
nnp = SingleNNP(training_data.ninput)
nnp.load(save_dir)

for m, (t_RMSE, t_dRMSE, t_tRMSE), (v_RMSE, v_dRMSE, v_tRMSE) in nnp.fit(training_data, validation_data, training_animator, validation_animator):
    file.write("""
{:<7} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f}\
""".format(m+1, time()-stime, t_RMSE, t_dRMSE, t_tRMSE, v_RMSE, v_dRMSE, v_tRMSE))
    file.flush()

training_animator.save(datestr, argv[1], 'training')
validation_animator.save(datestr, argv[1], 'validation')
nnp.save(save_dir)
file.close()

if argv[1] in ['LJ', 'sin']:
    fig = plt.figure()
    plt.scatter(training_data.input, training_animator.true['energy'], c='blue')
    plt.scatter(validation_data.input, validation_animator.true['energy'], c='blue')
    plt.scatter(training_data.input, training_animator.preds['energy'][-1], c='red')
    plt.scatter(validation_data.input, validation_animator.preds['energy'][-1], c='yellow')
    fig.savefig(path.join(file_.fig_dir, datestr, 'original_func.png'))
    plt.close(fig)
    fig = plt.figure()
    plt.scatter(training_data.input, training_animator.true['force'], c='blue')
    plt.scatter(validation_data.input, validation_animator.true['force'], c='blue')
    plt.scatter(training_data.input, training_animator.preds['force'][-1], c='red')
    plt.scatter(validation_data.input, validation_animator.preds['force'][-1], c='yellow')
    fig.savefig(path.join(file_.fig_dir, datestr, 'derivative.png'))
    plt.close()
