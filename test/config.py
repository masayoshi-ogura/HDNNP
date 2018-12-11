# -*- coding: utf-8 -*-

import matplotlib as mpl
from skopt.callbacks import DeltaYStopper
from skopt.space import (Real, Integer, Categorical)

from hdnnpy.settings import defaults as stg
from hdnnpy.skopt_callbacks import SamePointStopper

mpl.use('Agg')
mpl.rc('font', size=20)

# stg.file.out_dir = 'output'

stg.dataset.xyz_file = 'data/GaN.xyz'
# stg.dataset.tag = ['all']
stg.dataset.Rc = [5.0]
stg.dataset.eta = [0.01, 0.1, 1.0]
stg.dataset.Rs = [2.0, 3.2, 3.8]
stg.dataset.lambda_ = [-1, 1]
stg.dataset.zeta = [1, 2, 4]

stg.dataset.preproc = 'pca'
stg.dataset.nfeature = 20
# stg.dataset.ratio = 0.9
stg.dataset.batch_size = 5

stg.model.epoch = 10
# stg.model.interval = 10
# stg.model.patients = 5
# stg.model.init_lr = 1.0e-3
# stg.model.final_lr = 1.0e-6
# stg.model.lr_decay = 0.0e-6
# stg.model.l1_norm = 0.0e-4
# stg.model.l2_norm = 0.0e-4
stg.model.mixing_beta = 1.0
stg.model.layer = [
    {'node': 30, 'activation': 'tanh'},
    {'node': 30, 'activation': 'tanh'},
    {'node': 1, 'activation': 'identity'},
]
# stg.model.metrics = 'validation/main/tot_RMSE'

stg.skopt.kfold = 2
stg.skopt.init_num = 5
stg.skopt.max_num = 10
stg.skopt.space = [
    Real(1.0e-4, 1.0e-2, prior='log-uniform', name='init_lr'),
    Real(1.0e-6, 1.0e-4, prior='log-uniform', name='final_lr'),
    Real(1.0e-6, 1.0e-3, prior='log-uniform', name='lr_decay'),
]
stg.skopt.acq_func = 'LCB'
stg.skopt.callback = [
    SamePointStopper(),
    DeltaYStopper(1.0e-3, n_best=3),
]
