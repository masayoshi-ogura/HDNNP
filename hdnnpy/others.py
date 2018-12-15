# -*- coding: utf-8 -*-

__all__ = [
    'ChainerSafelyTerminate',
    'assert_settings',
    'dump_config',
    'dump_lammps',
    'dump_skopt_result',
    'dump_training_result',
    ]

import csv
from pathlib import Path
import pickle
import signal

import chainer
import yaml

from hdnnpy.settings import stg
from hdnnpy.utils import (MPI,
                          flatten_dict,
                          pprint,
                          )


class ChainerSafelyTerminate(object):
    def __init__(self, tag, trainer, result):
        self.tag = tag
        self.trainer = trainer
        self.result = result
        self.signum = None

    def __enter__(self):
        self.old_sigint_handler = signal.signal(signal.SIGINT, self._snapshot)
        self.old_sigterm_handler = signal.signal(signal.SIGTERM, self._snapshot)
        MPI.comm.Barrier()

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_sigint_handler)
        signal.signal(signal.SIGTERM, self.old_sigterm_handler)
        if not self.signum:
            self.result['training_time'] += self.trainer.elapsed_time
            self.result['observation'].append({'tag': self.tag, **flatten_dict(self.trainer.observation)})

    def _snapshot(self, signum, frame):
        self.signum = signal.Signals(signum)
        if stg.args.mode == 'train' and MPI.rank == 0:
            pprint(f'Stop {self.tag} training by signal: '
                   f'{self.signum.name}!\n'
                   f'Take trainer snapshot at epoch: '
                   f'{self.trainer.updater.epoch}')
            chainer.serializers.save_npz(
                self.trainer.out/'trainer_snapshot.npz', self.trainer)
            (self.trainer.out/'interim_result.pickle').write_bytes(pickle.dumps(self.result))
        # must raise any Exception to stop trainer.run()
        raise InterruptedError(f'Chainer training loop is interrupted by '
                               f'{self.signum.name}')


def dump_lammps(file_path, preprocess, masters):
    nelements = len(masters)
    depth = len(masters[0])
    with file_path.open('w') as f:
        f.write('# title\nneural network potential trained by HDNNP\n\n')
        f.write('# symmetry function parameters\n'
                f'{" ".join(map(str, stg.dataset.Rc))}\n'
                f'{" ".join(map(str, stg.dataset.eta))}\n'
                f'{" ".join(map(str, stg.dataset.Rs))}\n'
                f'{" ".join(map(str, stg.dataset.lambda_))}\n'
                f'{" ".join(map(str, stg.dataset.zeta))}\n\n')

        if stg.dataset.preprocess is None:
            f.write('# preprocess parameters\n0\n\n')
        elif stg.dataset.preprocess == 'pca':
            f.write('# preprocess parameters\n1\npca\n\n')
            for i in range(nelements):
                element = masters[i].element
                transform = preprocess.transform[element]
                mean = preprocess.mean[element]
                f.write(f'{element} {transform.shape[1]} '
                        f'{transform.shape[0]}\n')
                f.write('# transformation matrix\n')
                for row in transform.T:
                    f.write(f'{" ".join(map(str, row))}\n')
                f.write('# mean\n')
                f.write(f'{" ".join(map(str, mean))}\n\n')

        f.write(f'# neural network parameters\n{depth}\n\n')
        for i in range(nelements):
            for j in range(depth):
                W = getattr(masters[i], f'l{j}').W.data
                b = getattr(masters[i], f'l{j}').b.data
                f.write(f'{masters[i].element} {j+1} {W.shape[1]} '
                        f'{W.shape[0]} {stg.model.layer[j]["activation"]}\n')
                f.write('# weight\n')
                for row in W.T:
                    f.write(f'{" ".join(map(str, row))}\n')
                f.write('# bias\n')
                f.write(f'{" ".join(map(str, b))}\n\n')


def dump_training_result(file_path, result):
    args = {k: v if not isinstance(v, Path) else str(v)
            for k, v in vars(stg.args).items() if not k.startswith('_')}
    file = {k: v if not isinstance(v, Path) else str(v)
            for k, v in vars(stg.file).items() if not k.startswith('_')}
    dataset = {k: v if not isinstance(v, Path) else str(v)
               for k, v in vars(stg.dataset).items() if not k.startswith('_')}
    model = {k: v if not isinstance(v, Path) else str(v)
             for k, v in vars(stg.model).items() if not k.startswith('_')}

    with file_path.open('w') as f:
        yaml.dump({
            'args': args,
            'file': file,
            'dataset': dataset,
            'model': model,
            'result': result,
            }, f, default_flow_style=False)


def dump_skopt_result(file_path, result):
    with file_path.open('w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([space.name for space in stg.skopt.space] + ['score'])
        writer.writerows([x + [fun] for x, fun in zip(result.x_iters, result.func_vals)])


def dump_config(file_path):
    with file_path.open('w') as f:
        f.write('# -*- coding: utf-8 -*-\n'
                'from hdnnpy.settings import defaults as stg\n\n')
        for k, v in vars(stg.dataset).items():
            if k.startswith('_'):
                continue
            f.write(f'stg.dataset.{k} = {v}\n')
        for k, v in vars(stg.model).items():
            if k.startswith('_'):
                continue
            f.write(f'stg.model.{k} = {v}\n')


def assert_settings(stg):
    # file
    assert all(key in dir(stg.file) for key in ['out_dir'])
    assert stg.file.out_dir is not None

    # dataset
    assert all(key in dir(stg.dataset) for key in ['Rc', 'eta', 'Rs', 'lambda_', 'zeta'])
    assert all(key in dir(stg.dataset) for key in ['xyz_file', 'tag', 'preprocess', 'ratio'])
    assert all(key in dir(stg.dataset) for key in ['nfeature', 'batch_size'])
    assert len(stg.dataset.Rc) > 0
    assert len(stg.dataset.eta) > 0
    assert len(stg.dataset.Rs) > 0
    assert len(stg.dataset.lambda_) > 0
    assert len(stg.dataset.zeta) > 0
    assert stg.dataset.xyz_file is not None
    assert len(stg.dataset.tag) > 0
    assert all(preprocess in ['normalization', 'pca', 'standardization']
               for preprocess in stg.dataset.preprocess)
    assert 0.0 <= stg.dataset.ratio <= 1.0
    assert stg.dataset.nfeature > 0
    assert stg.dataset.batch_size >= MPI.size

    # model
    assert all(key in dir(stg.model) for key in ['epoch', 'interval', 'patients'])
    assert all(key in dir(stg.model) for key in ['init_lr', 'final_lr', 'lr_decay', 'mixing_beta'])
    assert all(key in dir(stg.model) for key in ['l1_norm', 'l2_norm', 'layer', 'metrics'])
    assert stg.model.epoch > 0
    assert stg.model.interval > 0
    assert stg.model.patients > 0
    assert 0.0 <= stg.model.init_lr <= 1.0
    assert 0.0 <= stg.model.final_lr <= stg.model.init_lr
    assert 0.0 <= stg.model.lr_decay <= 1.0
    assert 0.0 <= stg.model.mixing_beta <= 1.0
    assert 0.0 <= stg.model.l1_norm <= 1.0
    assert 0.0 <= stg.model.l2_norm <= 1.0
    assert len(stg.model.layer) > 0
    assert stg.model.metrics is not None

    # skopt
    if stg.args.mode == 'param-search':
        assert all(key in dir(stg.skopt) for key in ['kfold', 'init_num', 'max_num'])
        assert all(key in dir(stg.skopt) for key in ['space', 'acq_func', 'callback'])
        assert stg.skopt.kfold > 0
        assert stg.skopt.init_num > 0
        assert stg.skopt.max_num > stg.skopt.init_num
        assert len(stg.skopt.space) > 0
        assert all(space.name in ['Rc', 'eta', 'Rs', 'lambda_', 'zeta', 'preprocess', 'nfeature',
                                  'epoch', 'batch_size', 'init_lr', 'final_lr', 'lr_decay',
                                  'l1_norm', 'l2_norm', 'mixing_beta', 'node', 'activation']
                   for space in stg.skopt.space)
        assert stg.skopt.acq_func in ['LCB', 'EI', 'PI', 'gp_hedge', 'Elps', 'Plps']