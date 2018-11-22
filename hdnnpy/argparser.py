# -*- coding: utf-8 -*-

import argparse
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description='High Dimensional Neural Network Potential',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--debug', '-d',
                         action='store_true',
                         default=False,
                         help='enables verbose progress and debug output')

    subparsers = parser.add_subparsers(dest='mode')

    training_parser = subparsers.add_parser('training', help='train the network')
    prediction_parser = subparsers.add_parser('prediction', help='make prediction from trained network')

    # TODO: Fix help messages to acutual "help message"
    ps_parser = subparsers.add_parser('param_search', help='see `param_search -h`')
    sf_parser = subparsers.add_parser('sym_func', help='see `sym_func -h`')
    phonon_parser = subparsers.add_parser('phonon', help='see `phonon -h`')

    # training mode
    training_parser.add_argument('--verbose', '-v', 
                                action='store_true', 
                                default=False,
                                help='this flag may increase processing time.')
    training_parser.add_argument('--resume', '-r', 
                                type=Path,
                                help='resume training from given config directory.\n'
                                      'the given directory must contain '
                                      '`trainer_snapshot.npz`, `interim_result.pickle`.')

    # test mode
    for p in [prediction_parser, phonon_parser]:
        p.add_argument('--poscar', '-p', 
                        required=True, 
                        type=Path,
                        help='POSCAR file used for postprocess calculation.')
        p.add_argument('--masters', '-m', 
                        required=True, 
                        type=Path,
                        help='trained masters model used for postprocess calculation.')

    return parser.parse_args()
