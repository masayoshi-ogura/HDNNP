# -*- coding: utf-8 -*-

from pprint import pprint as pretty_print
import yaml
from os import makedirs
from sys import stdout
import numpy as np
from chainer import Variable

from . import settings as stg


def pprint(data, root_only=True, flush=True, **options):
    if stg.mpi.rank == 0 or not root_only:
        if isinstance(data, list) or isinstance(data, dict):
            pretty_print(data, **options)
        else:
            print(data, **options)
        if flush:
            stdout.flush()


def mkdir(path):
    if stg.mpi.rank == 0:
        makedirs(path, exist_ok=True)


def flatten_dict(dic):
    return {k: v.data.item() if isinstance(v, Variable)
            else v.item() if isinstance(v, np.float64)
            else v for k, v in dic.items()}


def dump_result(file_path, result):
    args = {k:v for k,v in vars(stg.args).items() if not k.startswith('_')}
    file = {k:v for k,v in vars(stg.file).items() if not k.startswith('_')}
    dataset = {k:v for k,v in vars(stg.dataset).items() if not k.startswith('_')}
    model = {k:v for k,v in vars(stg.model).items() if not k.startswith('_')}

    with open(file_path, 'w') as f:
        yaml.dump({
            'args': args,
            'file': file,
            'dataset': dataset,
            'model': model,
            'result': result,
        }, f, default_flow_style=False)


def dump_lammps(file_path, preproc, masters):
    nelements = len(masters)
    depth = len(masters[0])
    with open(file_path, 'w') as f:
        f.write('# title\nneural network potential trained by HDNNP\n\n')
        f.write('# symmetry function parameters\n{}\n{}\n{}\n{}\n{}\n\n'
                .format(' '.join(map(str, stg.dataset.Rc)),
                        ' '.join(map(str, stg.dataset.eta)),
                        ' '.join(map(str, stg.dataset.Rs)),
                        ' '.join(map(str, stg.dataset.lambda_)),
                        ' '.join(map(str, stg.dataset.zeta))))

        if stg.dataset.preproc is None:
            f.write('# preprocess parameters\n0\n\n')
        elif stg.dataset.preproc == 'pca':
            f.write('# preprocess parameters\n1\npca\n\n')
            for i in range(nelements):
                element = masters[i].element
                components = preproc.components[element]
                mean = preproc.mean[element]
                f.write('{} {} {}\n'.format(element, components.shape[1], components.shape[0]))
                f.write('# components\n')
                for row in components.T:
                    f.write('{}\n'.format(' '.join(map(str, row))))
                f.write('# mean\n')
                f.write('{}\n\n'.format(' '.join(map(str, mean))))

        f.write('# neural network parameters\n{}\n\n'.format(depth))
        for i in range(nelements):
            for j in range(depth):
                W = getattr(masters[i], 'l{}'.format(j)).W.data
                b = getattr(masters[i], 'l{}'.format(j)).b.data
                f.write('{} {} {} {} {}\n'
                        .format(masters[i].element, j + 1, W.shape[1], W.shape[0], stg.model.layer[j]['activation']))
                f.write('# weight\n')
                for row in W.T:
                    f.write('{}\n'.format(' '.join(map(str, row))))
                f.write('# bias\n')
                f.write('{}\n\n'.format(' '.join(map(str, b))))
