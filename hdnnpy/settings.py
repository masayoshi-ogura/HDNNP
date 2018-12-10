# -*- coding: utf-8 -*-

"""
This script defines the default settings for this program.
Some settings must be set, and all settings can be overwritten
by 'config.py' on your working directory.
Please see 'test/config.py' as a example.
"""

import importlib.util
from pathlib import Path
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
        tag = ['all']
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


def import_user_configurations(args):
    if args.mode == 'train' and args.is_resume:
        file_path = args.resume_dir.absolute().with_name('config.py')
    elif args.mode == 'predict':
        file_path = args.masters.absolute().with_name('config.py')
    else:
        file_path = Path.cwd()/'config.py'

    spec = importlib.util.spec_from_file_location('config', file_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    stg = config.stg
    if stg.mpi.rank == 0:
        print('Loaded user configuration file: {}'.format(file_path))

    # convert path string to pathlib.Path object
    stg.file.out_dir = Path(stg.file.out_dir)
    stg.dataset.xyz_file = Path(stg.dataset.xyz_file)
    return stg


args = get_parser()

stg = import_user_configurations(args)

file = stg.file
mpi = stg.mpi
dataset = stg.dataset
model = stg.model
skopt = stg.skopt

# Hide stdout from MPI subprocesses
if mpi.rank != 0:
    sys.stdout = Path(os.devnull).open('w')
    # sys.stderr = Path(os.devnull).open('w')
