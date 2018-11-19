# -*- coding: utf-8 -*-

import matplotlib as mpl
from skopt.space import Real
from skopt.callbacks import DeltaYStopper

from modules.skopt_callbacks import SamePointStopper

mpl.use('Agg')
mpl.rc('font', size=20)

file = dict(
    xyz_file='test/data/GaN.xyz',
    out_dir='test/output',
)

sym_func = dict(
    Rc=[5.0],
    eta=[0.01, 0.1, 1.0],
    Rs=[2.0, 3.2, 3.8],
    lambda_=[-1, 1],
    zeta=[1, 2, 4],
)

model = dict(
    epoch=10,
    batch_size=5,
    preproc='pca',
    input_size=20,
    mixing_beta=1.0,
    layer=[
        {'node': 30, 'activation': 'tanh'},
        {'node': 30, 'activation': 'tanh'},
        {'node': 1, 'activation': 'identity'},
    ],
)

skopt = dict(
    kfold=2,
    init_num=5,
    max_num=10,
    space=[
        Real(1.0e-4, 1.0e-2, prior='log-uniform', name='init_lr'),
        Real(1.0e-6, 1.0e-4, prior='log-uniform', name='final_lr'),
        Real(1.0e-6, 1.0e-3, prior='log-uniform', name='lr_decay'),
    ],
    acq_func='LCB',
    callback=[
        SamePointStopper(),
        DeltaYStopper(1.0e-3, n_best=3),
    ],
)
