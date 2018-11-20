# -*- coding: utf-8 -*-

"""
This script defines the default settings for this program.
Some settings must be set, and all settings can be overwritten
by 'settings.py' on your working directory.
Please see 'test/settings.py' as a example.
"""

from os import path
import os
import sys
from mpi4py import MPI
import chainermn

from .argparser import get_parser


class defaults:
    class file:
        out_dir = 'output'
    class mpi:
        comm = MPI.COMM_WORLD
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        chainer_comm = chainermn.create_communicator('naive', MPI.COMM_WORLD)
    class dataset:
        config = ['all']
        ratio = 0.9
    class model:
        interval = 10
        patients = 5
        init_lr = 1.0e-3
        final_lr = 1.0e-6
        lr_decay = 0.0e-6
        l1_norm = 0.0e-4
        l2_norm = 0.0e-4
        metrics = 'validation/main/tot_RMSE'
    class skopt:
        pass


def import_user_settings(args):
    if args.mode == 'training' and args.resume:
        sys.path.insert(0, path.dirname(args.resume))
    elif args.mode in ['prediction', 'phonon']:
        sys.path.insert(0, path.dirname(args.masters))
    else:
        sys.path.insert(0, os.getcwd())

    if not path.exists(path.join(sys.path[0], 'settings.py')):
        raise FileNotFoundError('`settings.py` is not found in {}'
                                .format(path.abspath(sys.path[0])))
    from settings import stg
    return stg


def import_phonopy_settings():
    sys.path.insert(0, os.getcwd())
    import phonopy_settings
    return phonopy_settings


args = get_parser()

stg = import_user_settings(args)
if args.mode == 'phonon':
    phonopy = import_phonopy_settings()

if not args.debug and stg.mpi.rank != 0:
    sys.stdout = open(os.devnull, 'w')

file = stg.file
mpi = stg.mpi
dataset = stg.dataset
model = stg.model
skopt = stg.skopt
