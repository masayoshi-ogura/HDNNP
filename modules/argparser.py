# -*- coding: utf-8 -*-

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='High Dimensional Neural Network Potential',
                                     fromfile_prefix_chars='@')
    subparsers = parser.add_subparsers(dest='mode')

    training_parser = subparsers.add_parser('training', help='see `training -h`')
    ps_parser = subparsers.add_parser('param_search', help='see `training -h`')
    sf_parser = subparsers.add_parser('sym_func', help='see `sf -h`')
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
    ps_parser.add_argument('--kfold', '-k', type=int,
                           help='# of block in k-fold cross validation\n'
                                'default value is 5')
    ps_parser.add_argument('--init', '-i', type=int,
                           help='# of initial data in Bayesian Optimization\n'
                                'default value is 10')
    ps_parser.add_argument('--max-iter', '-m', type=int,
                           help='max # of iteration in Bayesian Optimization\n'
                                'default value is 30')
    ps_parser.set_defaults(verbose=False, kfold=5, init=10, max_iter=30)

    # test mode
    test_parser.add_argument('--masters', '-m', nargs='*', type=str,
                             help='trained master models for phonon calculation.\n'
                                  'default: "masters.npz" in the latest output directory')
    phonon_parser.add_argument('--masters', '-m', nargs='*', type=str,
                               help='trained master models for phonon calculation.\n'
                                    'default: "masters.npz" in the latest output directory')
    optimize_parser.add_argument('--masters', '-m', nargs='*', type=str,
                                 help='trained master models for phonon calculation.\n'
                                      'default: "masters.npz" in the latest output directory')

    return parser.parse_args()
