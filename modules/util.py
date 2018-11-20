# -*- coding: utf-8 -*-

from pprint import pprint as pretty_print
import os
import sys
import signal
import pickle
import yaml
import numpy as np
import chainer
from chainer import Variable

from . import settings as stg


def pprint(data, flush=True, **options):
    if isinstance(data, list) or isinstance(data, dict):
        pretty_print(data, **options)
    else:
        print(data, **options)
    if flush:
        sys.stdout.flush()


def mkdir(path):
    if stg.mpi.rank == 0:
        os.makedirs(path, exist_ok=True)


def flatten_dict(dic):
    return {k: v.data.item() if isinstance(v, Variable)
            else v.item() if isinstance(v, np.float64)
            else v for k, v in dic.items()}


def set_hyperparameter(key, value):
    value = value if isinstance(value, str) else value.item()
    if key in ['node', 'activation']:
        for layer in stg.model.layer[:-1]:
            layer[key] = value
    elif key in dir(stg.dataset):
        setattr(stg.dataset, key, value)
    elif key in dir(stg.model):
        setattr(stg.model, key, value)


# signal handler of SIGINT and SIGTERM
class ChainerSafelyTerminate(object):
    def __init__(self, config, trainer, result):
        self.config = config
        self.trainer = trainer
        self.result = result
        self.signum = None

    def __enter__(self):
        stg.mpi.comm.Barrier()
        self.old_sigint_handler = signal.signal(signal.SIGINT, self._snapshot)
        self.old_sigterm_handler = signal.signal(signal.SIGTERM, self._snapshot)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_sigint_handler)
        signal.signal(signal.SIGTERM, self.old_sigterm_handler)
        if not self.signum:
            chainer.serializers.save_npz(os.path.join(self.trainer.out, 'masters.npz'),
                                         self.trainer.updater.get_optimizer('master').target)
            ### comment out: output lammps.nnp at end of training for each config
            # preproc = PREPROC[stg.dataset.preproc](stg.dataset.nfeature)
            # preproc.load(path.join(stg.file.out_dir, 'preproc.npz'))
            # dump_lammps(os.path.join(self.trainer.out, 'lammps.nnp'), preproc,
            #                          self.trainer.updater.get_optimizer('master').target)
            self.result['training_time'] += self.trainer.elapsed_time
            self.result['observation'].append({'config': self.config, **flatten_dict(self.trainer.observation)})

    def _snapshot(self, signum, frame):
        self.signum = signal.Signals(signum)
        if stg.args.mode == 'training' and stg.mpi.rank == 0:
            pprint('Stop {} training by signal: {}!\n'
                   'Take trainer snapshot at epoch: {}'
                   .format(self.config, self.signum.name, self.trainer.updater.epoch))
            chainer.serializers.save_npz(os.path.join(self.trainer.out, 'trainer_snapshot.npz'), self.trainer)
            with open(os.path.join(self.trainer.out, 'interim_result.pickle'), 'wb') as f:
                pickle.dump(self.result, f)
        # must raise any Exception to stop trainer.run()
        raise InterruptedError('Chainer training loop is interrupted by {}'.format(self.signum.name))


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
