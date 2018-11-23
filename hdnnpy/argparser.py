# -*- coding: utf-8 -*-

import argparse
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description='High Dimensional Neural Network Potential',
                                     fromfile_prefix_chars='@')
    parser.add_argument(
        '--debug', 
        '-d',
        action='store_true',
        default=False,
        help='enables verbose progress and debug output'
        )

     # TODO: Fix help messages to acutual "help message"
    subparsers = parser.add_subparsers(dest='mode')

    vasp2xyz_parser = subparsers.add_parser('vasp2xyz', help='converts vasp OUTCAR into a xyz file extention')
    training_parser = subparsers.add_parser('training', help='train the network')
    prediction_parser = subparsers.add_parser('prediction', help='make prediction from trained network')
    ps_parser = subparsers.add_parser('param_search', help='see `param_search -h`')
    sf_parser = subparsers.add_parser('sym_func', help='see `sym_func -h`')
    phonon_parser = subparsers.add_parser('phonon', help='see `phonon -h`')

    set_up_vasp2xyz_parser(vasp2xyz_parser)
    set_up_traning_parser(training_parser)
    set_up_prediction_and_phonon_parser(prediction_parser, phonon_parser)

    return parser.parse_args()

def set_up_vasp2xyz_parser(vasp2xyz_parser):
    """Setups vasp2xyz parser"""

    vasp2xyz_parser.add_argument(
        'prefix',
        help='prefix uses to distinguish files'
    )

    vasp2xyz_parser.add_argument(
        'OUTCAR',
        type=Path,
        help='path to vasp output file OUTCAR'
    )

    vasp2xyz_parser.add_argument(
        'filename',
        type=Path,
        help="path to output file" 
    )

def set_up_traning_parser(training_parser):
    """Setups training parser"""

    training_parser.add_argument(
        '--verbose',
        '-v', 
        action='store_true', 
        default=False,
        help='this flag may increase processing time.'
        )

    training_parser.add_argument(
        '--resume', 
        '-r', 
        type=Path,
        help='resume training from given config directory.\n'
             'the given directory must contain '
             '`trainer_snapshot.npz`, `interim_result.pickle`.')

def set_up_prediction_and_phonon_parser(prediction_parser, phonon_parser):
    """Setups param prediction and phonon parser"""

    for parser in [prediction_parser, phonon_parser]:
        parser.add_argument(
            '--poscar', 
            '-p', 
            required=True, 
            type=Path,
            help='POSCAR file used for postprocess calculation.'
            )
        parser.add_argument(
            '--masters', 
            '-m', 
            required=True, 
            type=Path,
            help='trained masters model used for postprocess calculation.'
            )
