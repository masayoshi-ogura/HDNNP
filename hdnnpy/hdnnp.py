# -*- coding: utf-8 -*-

__all__ = [
    'main',
    ]

import os
from pathlib import Path
import pickle
import shutil
import sys

import chainer
from chainer.dataset import concat_examples
import chainer.training.extensions as ext
from chainer.training.triggers import EarlyStoppingTrigger
import chainermn
import numpy as np
from skopt import gp_minimize
from skopt.utils import use_named_args

from hdnnpy.chainer_extensions import (Evaluator,
                                       scatter_plot,
                                       set_log_scale,
                                       )
from hdnnpy.dataset import (DatasetGenerator,
                            construct_test_datasets,
                            construct_training_datasets,
                            )
from hdnnpy.format import (parse_poscars,
                           parse_xyz,
                           )
from hdnnpy.model import (HDNNP,
                          SingleNNP,
                          loss_func,
                          )
from hdnnpy.preproc import PREPROC
from hdnnpy.settings import stg
from hdnnpy.updater import HDUpdater
from hdnnpy.utils import (ChainerSafelyTerminate,
                          assert_settings,
                          dump_config,
                          dump_lammps,
                          dump_skopt_result,
                          dump_training_result,
                          mkdir,
                          pprint,
                          )


def main():
    assert_settings(stg)
    mkdir(stg.file.out_dir)

    if stg.args.mode == 'train':
        try:
            preproc = PREPROC[stg.dataset.preproc](stg.dataset.nfeature) \
                    if stg.mpi.rank == 0 else PREPROC[None]()
            if stg.args.is_resume:
                preproc.load(stg.args.resume_dir.with_name('preproc.npz'))
            datasets, elements = construct_training_datasets(parse_xyz(stg.dataset.xyz_file), preproc)
            preproc.save(stg.file.out_dir/'preproc.npz')
            dataset = DatasetGenerator(*datasets).holdout(stg.dataset.ratio)
            masters, result = train(dataset, elements)
            if stg.mpi.rank == 0:
                chainer.serializers.save_npz(stg.file.out_dir/'masters.npz', masters)
                dump_lammps(stg.file.out_dir/'lammps.nnp', preproc, masters)
                dump_training_result(stg.file.out_dir/'result.yaml', result)
        finally:
            shutil.copy('config.py', stg.file.out_dir/'config.py')

    elif stg.args.mode == 'param-search':
        try:
            seed = np.random.get_state()[1][0]
            seed = stg.mpi.comm.bcast(seed, root=0)
            result = gp_minimize(objective_func, stg.skopt.space,
                                 n_random_starts=stg.skopt.init_num,
                                 n_calls=stg.skopt.max_num,
                                 acq_func=stg.skopt.acq_func,
                                 random_state=seed,
                                 verbose=True,
                                 callback=stg.skopt.callback)
            if stg.mpi.rank == 0:
                dump_skopt_result(stg.file.out_dir/'skopt_result.csv', result)
                dump_config(stg.file.out_dir/'best_config.py')
        finally:
            shutil.copy('config.py', stg.file.out_dir/'config.py')

    elif stg.args.mode == 'sym-func':
        preproc = PREPROC[None]()
        construct_training_datasets(parse_xyz(stg.dataset.xyz_file), preproc)

    elif stg.args.mode == 'predict':
        if stg.args.is_write:
            stream = stg.args.prediction_file.open('w')
        else:
            stream = sys.stdout

        preproc = PREPROC[stg.dataset.preproc](stg.dataset.nfeature)
        preproc.load(stg.args.masters.with_name('preproc.npz'))
        tag_xyz_map, tag_poscars_map = parse_poscars(stg.args.poscars)
        datasets, elements = construct_test_datasets(tag_xyz_map, preproc)

        for xyz_path, poscars, dataset in zip(
                tag_xyz_map.values(), tag_poscars_map.values(),
                DatasetGenerator(*datasets).foreach()):
            xyz_path.unlink()
            energies, forces = predict(dataset, elements)

            for poscar, energy, force in zip(poscars, energies, forces):
                pprint('Atomic Structure File:', stream=stream)
                pprint(poscar, stream=stream)
                if 'energy' in stg.args.value or 'E' in stg.args.value:
                    pprint('Total Energy:', stream=stream)
                    np.savetxt(stream, energy)
                if 'force' in stg.args.value or 'F' in stg.args.value:
                    for f in force:
                        pprint('Forces:', stream=stream)
                        np.savetxt(stream, f)
                        pprint(stream=stream)


@use_named_args(stg.skopt.space)
def objective_func(**params):
    for key, value in params.items():
        value = value if isinstance(value, str) else value.item()
        if key in ['node', 'activation']:
            for layer in stg.model.layer[:-1]:
                layer[key] = value
        elif key in dir(stg.dataset):
            setattr(stg.dataset, key, value)
        elif key in dir(stg.model):
            setattr(stg.model, key, value)
    assert_settings(stg)

    results = []
    with Path(os.devnull).open('w') as devnull:
        stdout = sys.stdout
        sys.stdout = devnull
        preproc = PREPROC[stg.dataset.preproc](stg.dataset.nfeature) \
                if stg.mpi.rank == 0 else PREPROC[None]()
        datasets, elements = construct_training_datasets(parse_xyz(stg.dataset.xyz_file), preproc)
        for dataset in DatasetGenerator(*datasets).kfold(stg.skopt.kfold):
            _, result = train(dataset, elements)
            results.append(result['observation'][-1][stg.model.metrics])
        sys.stdout = stdout
    return sum(results) / stg.skopt.kfold


def train(dataset, elements):
    result = {'training_time': 0.0, 'observation': []}

    # model and optimizer
    masters = chainer.ChainList(*[SingleNNP(element) for element in elements])
    master_opt = chainer.optimizers.Adam(stg.model.init_lr)
    master_opt = chainermn.create_multi_node_optimizer(master_opt, stg.mpi.chainer_comm)
    master_opt.setup(masters)
    master_opt.add_hook(chainer.optimizer_hooks.Lasso(stg.model.l1_norm))
    master_opt.add_hook(chainer.optimizer_hooks.WeightDecay(stg.model.l2_norm))

    for train, test in dataset:
        tag = train.tag

        # iterators
        train_iter = chainer.iterators.SerialIterator(train, stg.dataset.batch_size // stg.mpi.size,
                                                      repeat=True, shuffle=True)
        test_iter = chainer.iterators.SerialIterator(test, stg.dataset.batch_size // stg.mpi.size,
                                                     repeat=False, shuffle=False)

        # model
        hdnnp = HDNNP(train.elemental_composition, loss_func)
        hdnnp.sync_param_with(masters)
        main_opt = chainer.Optimizer()
        main_opt = chainermn.create_multi_node_optimizer(main_opt, stg.mpi.chainer_comm)
        main_opt.setup(hdnnp)

        # triggers
        interval = (stg.model.interval, 'epoch')
        stop_trigger = EarlyStoppingTrigger(check_trigger=interval, monitor=stg.model.metrics,
                                            patients=stg.model.patients, mode='min',
                                            verbose=stg.args.mode == 'train',
                                            max_trigger=(stg.model.epoch, 'epoch'))

        # updater and trainer
        updater = HDUpdater(train_iter, optimizer={'main': main_opt, 'master': master_opt})
        out_dir = stg.file.out_dir/tag if stg.args.mode == 'train' else stg.file.out_dir
        trainer = chainer.training.Trainer(updater, stop_trigger, out_dir)

        # extensions
        trainer.extend(ext.ExponentialShift('alpha', 1 - stg.model.lr_decay,
                                            target=stg.model.final_lr, optimizer=master_opt))
        evaluator = Evaluator(iterator=test_iter, target=hdnnp)
        trainer.extend(chainermn.create_multi_node_evaluator(evaluator, stg.mpi.chainer_comm))
        if stg.args.mode == 'train' and stg.mpi.rank == 0:
            trainer.extend(ext.LogReport(log_name='training.log'))
            trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/RMSE', 'main/d_RMSE', 'main/tot_RMSE',
                                            'validation/main/RMSE', 'validation/main/d_RMSE',
                                            'validation/main/tot_RMSE']))
            trainer.extend(scatter_plot(hdnnp, test), trigger=interval)
            if stg.args.verbose:
                trainer.extend(ext.PlotReport(['main/tot_RMSE', 'validation/main/tot_RMSE'], 'epoch',
                                              file_name='RMSE.png', marker=None, postprocess=set_log_scale))

        # load trainer snapshot and resume training
        if stg.args.mode == 'train' and stg.args.is_resume:
            if tag != stg.args.resume_dir.name:
                continue
            pprint(f'Resume training loop from dataset tagged "{tag}"')
            trainer_snapshot = stg.args.resume_dir/'trainer_snapshot.npz'
            interim_result = stg.args.resume_dir/'interim_result.pickle'
            chainer.serializers.load_npz(trainer_snapshot, trainer)
            result = pickle.loads(interim_result.read_bytes())
            # remove snapshot
            stg.mpi.comm.Barrier()
            if stg.mpi.rank == 0:
                trainer_snapshot.unlink()
                interim_result.unlink()
            stg.args.is_resume = False

        with ChainerSafelyTerminate(tag, trainer, result):
            trainer.run()

    return masters, result


def predict(dataset, elements):
    masters = chainer.ChainList(*[SingleNNP(element) for element in elements])
    chainer.serializers.load_npz(stg.args.masters, masters)
    hdnnp = HDNNP(dataset.elemental_composition, loss_func)
    hdnnp.sync_param_with(masters)
    energies, forces = hdnnp.predict(*concat_examples(dataset[:]))
    return energies.data, forces.data


if __name__ == '__main__':
    main()
