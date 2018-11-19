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


class defaults:
    class file:
        out_dir='output'
    class mpi:
        comm=MPI.COMM_WORLD
        rank=MPI.COMM_WORLD.Get_rank()
        size=MPI.COMM_WORLD.Get_size()
        gpu=-1
        chainer_comm=chainermn.create_communicator('naive', MPI.COMM_WORLD)
    class dataset:
        config=['all']
        ratio=0.9
    class model:
        init_lr=1.0e-3
        final_lr=1.0e-6
        lr_decay=0.0e-6
        l1_norm=0.0e-4
        l2_norm=0.0e-4
        metrics='validation/main/tot_RMSE'
    class skopt:
        pass


def import_user_settings(args):
    if args.mode in ['prediction', 'phonon']:
        sys.path.insert(0, os.path.dirname(args.masters))
    else:
        sys.path.insert(0, os.getcwd())

    if not os.path.exists(os.path.join(sys.path[0], 'settings.py')):
        raise FileNotFoundError('`settings.py` is not found in {}'
                                .format(os.path.abspath(sys.path[0])))
    from settings import stg
    return stg


def import_phonopy_settings():
    sys.path.insert(0, os.getcwd())
    import phonopy_settings
    return phonopy_settings


def assert_settings(args, stg):
    # file
    assert all(key in dir(stg.file) for key in ['out_dir'])
    assert stg.file.out_dir is not None

    # mpi
    assert all(key in dir(stg.mpi) for key in ['comm', 'rank', 'size', 'gpu', 'chainer_comm'])
    assert stg.mpi.comm is not None
    assert 0 <= stg.mpi.rank < stg.mpi.size
    assert stg.mpi.size > 0
    assert stg.mpi.gpu is not None
    assert isinstance(stg.mpi.chainer_comm, MpiCommunicatorBase)

    # dataset
    assert all(key in dir(stg.dataset) for key in ['Rc', 'eta', 'Rs', 'lambda_', 'zeta'])
    assert all(key in dir(stg.dataset) for key in ['xyz_file', 'config', 'preproc', 'ratio'])
    assert all(key in dir(stg.dataset) for key in ['nfeature', 'epoch', 'batch_size'])
    assert len(stg.dataset.Rc) > 0
    assert len(stg.dataset.eta) > 0
    assert len(stg.dataset.Rs) > 0
    assert len(stg.dataset.lambda_) > 0
    assert len(stg.dataset.zeta) > 0
    assert stg.dataset.xyz_file is not None
    assert len(stg.dataset.config) > 0
    assert stg.dataset.preproc in [None, 'pca']
    assert 0.0 <= stg.dataset.ratio <= 1.0
    assert stg.dataset.nfeature > 0
    assert stg.dataset.epoch > 0
    assert stg.dataset.batch_size >= stg.mpi.size

    # model
    assert all(key in dir(stg.model) for key in ['init_lr', 'final_lr', 'lr_decay', 'mixing_beta'])
    assert all(key in dir(stg.model) for key in ['l1_norm', 'l2_norm', 'layer', 'metrics'])
    assert 0.0 <= stg.model.init_lr <= 1.0
    assert 0.0 <= stg.model.final_lr <= stg.model.init_lr
    assert 0.0 <= stg.model.lr_decay <= 1.0
    assert 0.0 <= stg.model.mixing_beta <= 1.0
    assert 0.0 <= stg.model.l1_norm <= 1.0
    assert 0.0 <= stg.model.l2_norm <= 1.0
    assert len(stg.model.layer) > 0
    assert stg.model.metrics is not None

    # skopt
    if args.mode == 'param_search':
        assert all(key in dir(stg.skopt) for key in ['kfold', 'init_num', 'max_num'])
        assert all(key in dir(stg.skopt) for key in ['space', 'acq_func', 'callback'])
        assert stg.skopt.kfold > 0
        assert stg.skopt.init_num > 0
        assert stg.skopt.max_num > stg.skopt.init_num
        assert len(stg.skopt.space) > 0
        assert all(space.name in ['Rc', 'eta', 'Rs', 'lambda_', 'zeta', 'preproc', 'nfeature',
                                  'epoch', 'batch_size', 'init_lr', 'final_lr', 'lr_decay',
                                  'l1_norm', 'l2_norm', 'mixing_beta', 'node', 'activation']
                   for space in stg.skopt.space)
        assert stg.skopt.acq_func in ['LCB', 'EI', 'PI', 'gp_hedge', 'Elps', 'Plps']


def assert_phonopy_settings(stg):
    assert all(key in dir(stg) for key in ['dimensions', 'options', 'distance', 'callback'])
    assert len(stg.dimensions) == 3 and all(len(d) == 3 for d in stg.dimensions)
    assert isinstance(stg.options, dict)
    assert stg.distance > 0.0


args = get_parser()
stg = import_user_settings(args)
assert_settings(args, stg)

if args.mode == 'phonon':
    phonopy = import_phonopy_settings()
    assert_phonopy_settings(phonopy)

file = stg.file
mpi = stg.mpi
dataset = stg.dataset
model = stg.model
skopt = stg.skopt
