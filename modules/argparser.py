# -*- coding: utf-8 -*-

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='High Dimensional Neural Network Potential',
                                     fromfile_prefix_chars='@')
    subparsers = parser.add_subparsers(dest='mode')

    training_parser = subparsers.add_parser('training', help='see `training -h`')
    ps_parser = subparsers.add_parser('param_search', help='see `param_search -h`')
    sf_parser = subparsers.add_parser('sym_func', help='see `sym_func -h`')
    prediction_parser = subparsers.add_parser('prediction', help='see `prediction -h`')
    phonon_parser = subparsers.add_parser('phonon', help='see `phonon -h`')

    parser.add_argument('--debug', '-d', action='store_true', default=False,
                        help='if this flag is set, MPI subprocesses also print their stdout')

    # training mode
    training_parser.add_argument('--verbose', '-v', action='store_true', default=False,
                                 help='this flag may increase processing time.')
    training_parser.add_argument('--resume', '-r', type=str,
                                 help='resume training from given config directory.\n'
                                      'do not add "/" at the end of path.'
                                      'the given directory must contain '
                                      '`trainer_snapshot.npz`, `interim_result.pickle`.')

    # param_search mode
    ps_parser.add_argument('--verbose', '-v', action='store_true', default=False,
                           help='this flag may increase processing time.')

    # test mode
    for p in [prediction_parser, phonon_parser]:
        p.add_argument('--poscar', '-p', required=True, type=str,
                       help='POSCAR file used for postprocess calculation.')
        p.add_argument('--masters', '-m', required=True, type=str,
                       help='trained masters model used for postprocess calculation.')

    return parser.parse_args()
