# -*- coding: utf-8 -*-

"""
This script defines the default settings for this program.
Some settings must be set, and all settings can be overwritten
by 'settings.py' on your working directory.
Please see 'test/settings.py' as a example.
"""

import os
import sys
from mpi4py import MPI
import chainermn
from chainermn.communicators.mpi_communicator_base import MpiCommunicatorBase

from .argparser import get_parser


def import_user_settings(args):
    if args['mode'] in ['test', 'phonon', 'optimize']:
        sys.path.insert(0, os.path.dirname(args['masters']))
    if not os.path.exists(os.path.join(sys.path[0], 'settings.py')):
        raise FileNotFoundError('`settings.py` is not found in {}'
                                .format(os.path.abspath(sys.path[0])))
    import settings as user_defined
    return user_defined


class defaults(object):
    file = dict(
        config=['all'],
        out_dir='output',
        ratio=0.9,
    )
    mpi = dict(
        comm=MPI.COMM_WORLD,
        rank=MPI.COMM_WORLD.Get_rank(),
        size=MPI.COMM_WORLD.Get_size(),
        gpu=-1,
        chainer_comm=chainermn.create_communicator('naive', MPI.COMM_WORLD),
    )
    model = dict(
        init_lr=1.0e-3,
        final_lr=1.0e-6,
        lr_decay=0.0e-6,
        l1_norm=0.0e-4,
        l2_norm=0.0e-4,
        metrics='validation/main/tot_RMSE',
    )


def assert_settings(args, file, mpi, sym_func, model, skopt):
    # file
    assert all(key in file for key in ['xyz_file', 'config', 'out_dir', 'ratio'])
    assert file['xyz_file'] is not None
    assert len(file['config']) > 0
    assert file['out_dir'] is not None
    assert 0.0 <= file['ratio'] <= 1.0

    # mpi
    assert all(key in mpi for key in ['comm', 'rank', 'size', 'gpu', 'chainer_comm'])
    assert mpi['comm'] is not None
    assert 0 <= mpi['rank'] < mpi['size']
    assert mpi['size'] > 0
    assert mpi['gpu'] is not None
    assert isinstance(mpi['chainer_comm'], MpiCommunicatorBase)

    # sym_func
    assert all(key in sym_func for key in ['Rc', 'eta', 'Rs', 'lambda_', 'zeta'])
    assert len(sym_func['Rc']) > 0
    assert len(sym_func['eta']) > 0
    assert len(sym_func['Rs']) > 0
    assert len(sym_func['lambda_']) > 0
    assert len(sym_func['zeta']) > 0

    # model
    assert all(key in model for key in ['epoch', 'batch_size', 'preproc', 'input_size'])
    assert all(key in model for key in ['init_lr', 'final_lr', 'lr_decay', 'mixing_beta'])
    assert all(key in model for key in ['l1_norm', 'l2_norm', 'layer', 'metrics'])
    assert model['epoch'] > 0
    assert model['batch_size'] >= mpi['size']
    assert model['preproc'] in [None, 'pca']
    assert model['input_size'] > 0
    assert 0.0 <= model['init_lr'] <= 1.0
    assert 0.0 <= model['final_lr'] <= model['init_lr']
    assert 0.0 <= model['lr_decay'] <= 1.0
    assert 0.0 <= model['mixing_beta'] <= 1.0
    assert 0.0 <= model['l1_norm'] <= 1.0
    assert 0.0 <= model['l2_norm'] <= 1.0
    assert len(model['layer']) > 0
    assert model['metrics'] is not None

    # skopt
    if args['mode'] == 'param_search':
        assert all(key in skopt for key in ['kfold', 'init_num', 'max_num'])
        assert all(key in skopt for key in ['space', 'acq_func', 'callback'])
        assert skopt['kfold'] > 0
        assert skopt['init_num'] > 0
        assert skopt['max_num'] > skopt['init_num']
        assert len(skopt['space']) > 0
        assert skopt['acq_func'] in ['LCB', 'EI', 'PI', 'gp_hedge', 'Elps', 'Plps']


args = vars(get_parser())
user_defined = import_user_settings(args)

file = {**defaults.file, **user_defined.file}
mpi = {**defaults.mpi, **user_defined.mpi} if 'mpi' in dir(user_defined) else defaults.mpi
sym_func = user_defined.sym_func
model = {**defaults.model, **user_defined.model}
skopt = user_defined.skopt if 'skopt' in dir(user_defined) else {}

assert_settings(args, file, mpi, sym_func, model, skopt)
