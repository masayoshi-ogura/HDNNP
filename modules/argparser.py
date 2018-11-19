# -*- coding: utf-8 -*-

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='High Dimensional Neural Network Potential',
                                     fromfile_prefix_chars='@')
    subparsers = parser.add_subparsers(dest='mode')

    training_parser = subparsers.add_parser('training', help='see `training -h`')
    ps_parser = subparsers.add_parser('param_search', help='see `param_search -h`')
    sf_parser = subparsers.add_parser('sym_func', help='see `sym_func -h`')
    test_parser = subparsers.add_parser('test', help='see `test -h`')
    phonon_parser = subparsers.add_parser('phonon', help='see `phonon -h`')
    optimize_parser = subparsers.add_parser('optimize', help='see `optimize -h`')

    # training mode
    training_parser.add_argument('--verbose', '-v', action='store_true',
                                 help='trainer extensions "PlotReport snapshot_object" is set.\n'
                                      'this flag increases processing time.')
    training_parser.set_defaults(verbose=False)

    # parameter search mode
    ps_parser.add_argument('--verbose', '-v', action='store_true',
                           help='trainer extensions "PlotReport snapshot_object" is set.\n'
                                'this flag increases processing time.')
    ps_parser.set_defaults(verbose=False)

    # test mode
    for p in [test_parser, phonon_parser, optimize_parser]:
        p.add_argument('--poscar', '-p', required=True, type=str,
                       help='POSCAR file used for postprocess calculation.')
        p.add_argument('--masters', '-m', required=True, type=str,
                       help='trained masters model used for postprocess calculation.')

    return parser.parse_args()
