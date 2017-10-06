# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import file_

# import python modules
from sys import argv
from time import time
from os import path
import math
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# import own modules
from modules.model import SingleNNP
from modules.animator import Animator

datestr = datetime.now().strftime('%m%d-%H%M%S')
file = open(path.join(file_.progress_dir, 'progress-{}.out'.format(datestr)), 'w')
stime = time()
file.write("""
learning_rate:       {}
learning_rate_decay: {}
mixing_beta:         {}
smooth_factor:       {}
batch_size:          {}
batch_size_growth:   {}
optimizer:           {}
""".format(hp.learning_rate, hp.learning_rate_decay, hp.mixing_beta, hp.smooth_factor,
           hp.batch_size, hp.batch_size_growth, hp.optimizer))
file.flush()

# ***データ生成***
if argv[1] == 'complicate':
    ninput = 3
    mesh = 10
    nsample = mesh**3
    lin = np.linspace(0.1, 1.0, mesh)
    x, y, z = np.meshgrid(lin, lin, lin)
    x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
    input = np.c_[x, y, z]
    label = (x**2 + np.sin(y) + 3.*np.exp(z) - np.log(x*y)/2 - y/z).reshape(nsample, 1)
    dinput = np.identity(3)[None, :, :].repeat(nsample, axis=0)
    dlabel = np.c_[2**x - 1/(2*x),
                   np.cos(y) - 1/(2*y) - 1/z,
                   3.*np.exp(z) + y/z**2].reshape(nsample, 3, 1)
elif argv[1] == 'LJ':
    ninput = 1
    nsample = 1000
    x = np.linspace(0.1, 1.0, nsample).reshape(nsample, 1)
    input = x.reshape(nsample, 1)
    label = (0.001/x**4 - 0.009/x**3).reshape(nsample, 1)
    dinput = np.ones(nsample).reshape(nsample, 1, 1)
    dlabel = (0.027/x**4 - 0.004/x**5).reshape(nsample, 1, 1)
elif argv[1] == 'sin':
    ninput = 1
    nsample = 1000
    x = np.linspace(-2*math.pi, 2*math.pi, nsample).reshape(nsample, 1)
    input = x
    label = np.sin(x)
    dinput = np.ones(nsample).reshape(nsample, 1, 1)
    dlabel = np.cos(x).reshape(nsample, 1, 1)
# データをシャッフル＆training,validationデータセットに分割(本来はさらにtestデータセットも別にしておく)
np.random.seed(0)
np.random.shuffle(input)
np.random.seed(0)
np.random.shuffle(label)
np.random.seed(0)
np.random.shuffle(dinput)
np.random.seed(0)
np.random.shuffle(dlabel)
sep = nsample * 7 / 10
training_data = (input[:sep], label[:sep], dinput[:sep], dlabel[:sep])
validation_data = (input[sep:], label[sep:], dinput[sep:], dlabel[sep:])

file.write("""
-------------------{}-----------------------

ninput:        {}
hidden_layers:
\t{}
nepoch:        {}
nsample:       {}

epoch   spent time        training_RMSE     training_dRMSE    training_tRMSE    validation_RMSE   validation_dRMSE  validation_tRMSE
""".format(argv[1], ninput, '\n\t'.join(map(str, hp.hidden_layers)), hp.nepoch, nsample))
file.flush()

# initialize NNP
training_animator = Animator('training')
validation_animator = Animator('validation')
nnp = SingleNNP(ninput)
nnp.load(datestr)

for m, (t_RMSE, t_dRMSE, t_tRMSE), (v_RMSE, v_dRMSE, v_tRMSE) in nnp.fit(training_data, validation_data, training_animator, validation_animator):
    file.write('{:<7} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f} {:<17.12f}\n'
               .format(m+1, time()-stime, t_RMSE, t_dRMSE, t_tRMSE, v_RMSE, v_dRMSE, v_tRMSE))
    file.flush()

training_animator.save_fig(datestr, argv[1])
validation_animator.save_fig(datestr, argv[1])
nnp.save(datestr)
file.close()

if argv[1] in ['LJ', 'sin']:
    fig = plt.figure()
    plt.scatter(input[:sep], training_animator.true['energy'], c='blue')
    plt.scatter(input[sep:], validation_animator.true['energy'], c='blue')
    plt.scatter(input[:sep], training_animator.preds['energy'][-1], c='red')
    plt.scatter(input[sep:], validation_animator.preds['energy'][-1], c='yellow')
    fig.savefig(path.join(file_.fig_dir, datestr, 'original_func.png'))
    plt.close(fig)
    fig = plt.figure()
    plt.scatter(input[:sep], training_animator.true['force'], c='blue')
    plt.scatter(input[sep:], validation_animator.true['force'], c='blue')
    plt.scatter(input[:sep], training_animator.preds['force'][-1], c='red')
    plt.scatter(input[sep:], validation_animator.preds['force'][-1], c='yellow')
    fig.savefig(path.join(file_.fig_dir, datestr, 'derivative.png'))
    plt.close()
