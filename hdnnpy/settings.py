# coding: utf-8

"""This script defines the default settings for this program.
Some settings must be set, and all settings can be overwritten
by 'config.py' on your working directory.
Please see 'test/config.py' as a example.
"""

__all__ = [
    'defaults',
    'stg',
    ]

import importlib.util
import os
from pathlib import Path
import sys

from hdnnpy.argparser import get_parser
from hdnnpy.utils import MPI


class defaults:
    class file:
        out_dir = 'output'

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
        metrics = 'validation/main/total_RMSE'

    class skopt:
        pass


def import_user_configurations(args):
    if args.mode == 'train' and args.is_resume:
        file_path = args.resume_dir.with_name('config.py')
    elif args.mode == 'predict':
        file_path = args.masters.with_name('config.py')
    else:
        file_path = Path.cwd()/'config.py'

    spec = importlib.util.spec_from_file_location('config', file_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    stg = config.stg
    if MPI.rank == 0:
        print('Loaded user configuration from {}'.format(file_path))

    # convert path string to pathlib.Path object
    if args.mode == 'train' and args.is_resume:
        stg.file.out_dir = args.resume_dir.parent
    else:
        stg.file.out_dir = Path(stg.file.out_dir).absolute()
    stg.dataset.xyz_file = Path(stg.dataset.xyz_file).absolute()

    # add args in `stg` namespace
    stg.args = args
    return stg


stg = import_user_configurations(get_parser())

# Hide stdout from MPI subprocesses
if MPI.rank != 0:
    sys.stdout = Path(os.devnull).open('w')
    # sys.stderr = Path(os.devnull).open('w')
