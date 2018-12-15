# coding: utf-8

__all__ = [
    'main',
    ]

from itertools import product
import os
from pathlib import Path
import pickle
import shutil
import sys

import ase.io
import chainer
from chainer.dataset import concat_examples
import chainer.training.extensions as ext
from chainer.training.triggers import EarlyStoppingTrigger
import chainermn
import numpy as np
from skopt import gp_minimize
from skopt.utils import use_named_args

from hdnnpy.chainer import (Evaluator,
                            HighDimensionalNNP,
                            Manager,
                            MasterNNP,
                            Updater,
                            scatter_plot,
                            set_log_scale,
                            )
from hdnnpy.dataset import (AtomicStructure,
                            DatasetGenerator,
                            HDNNPDataset,
                            )
from hdnnpy.format import (parse_poscars,
                           parse_xyz,
                           )
from hdnnpy.others import (assert_settings,
                           dump_config,
                           dump_skopt_result,
                           dump_training_result,
                           )
from hdnnpy.preprocess import PREPROCESS
from hdnnpy.settings import stg
from hdnnpy.utils import (MPI,
                          mkdir,
                          pprint,
                          )


def main():
    assert_settings(stg)
    mkdir(stg.file.out_dir)

    if stg.args.mode == 'train':
        try:
            tag_xyz_map, elements = parse_xyz(stg.dataset.xyz_file)
            datasets = construct_training_datasets(tag_xyz_map, elements)
            dataset = DatasetGenerator(*datasets).holdout(stg.dataset.ratio)
            masters, result = train(dataset, elements)
            if MPI.rank == 0:
                chainer.serializers.save_npz(stg.file.out_dir/'masters.npz', masters)
                # dump_lammps(stg.file.out_dir/'lammps.nnp', preprocesses[0], masters)
                dump_training_result(stg.file.out_dir/'result.yaml', result)
        finally:
            shutil.copy('config.py', stg.file.out_dir/'config.py')

    elif stg.args.mode == 'param-search':
        try:
            seed = np.random.get_state()[1][0]
            seed = MPI.comm.bcast(seed, root=0)
            result = gp_minimize(objective_func, stg.skopt.space,
                                 n_random_starts=stg.skopt.init_num,
                                 n_calls=stg.skopt.max_num,
                                 acq_func=stg.skopt.acq_func,
                                 random_state=seed,
                                 verbose=True,
                                 callback=stg.skopt.callback)
            if MPI.rank == 0:
                dump_skopt_result(stg.file.out_dir/'skopt_result.csv', result)
                dump_config(stg.file.out_dir/'best_config.py')
        finally:
            shutil.copy('config.py', stg.file.out_dir/'config.py')

    elif stg.args.mode == 'sym-func':
        tag_xyz_map, elements = parse_xyz(stg.dataset.xyz_file)
        construct_training_datasets(tag_xyz_map, elements)

    elif stg.args.mode == 'predict':
        if stg.args.is_write:
            stream = stg.args.prediction_file.open('w')
        else:
            stream = sys.stdout

        tag_xyz_map, tag_poscars_map, elements = parse_poscars(stg.args.poscars)
        datasets = construct_test_datasets(tag_xyz_map, elements)

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
        tag_xyz_map, elements = parse_xyz(stg.dataset.xyz_file)
        datasets = construct_training_datasets(tag_xyz_map, elements)
        for dataset in DatasetGenerator(*datasets).kfold(stg.skopt.kfold):
            _, result = train(dataset, elements)
            results.append(result['observation'][-1][stg.model.metrics])
        sys.stdout = stdout
    return sum(results) / stg.skopt.kfold


def train(dataset, elements, comm=None):
    if comm is None:
        comm = chainermn.create_communicator('naive', MPI.comm)
    is_train = stg.args.mode == 'train'
    result = {'training_time': 0.0, 'observation': []}

    # model and optimizer
    masters = MasterNNP(elements, stg.model.layer)
    master_opt = chainer.optimizers.Adam(stg.model.init_lr)
    master_opt = chainermn.create_multi_node_optimizer(master_opt, comm)
    master_opt.setup(masters)
    master_opt.add_hook(chainer.optimizer_hooks.Lasso(stg.model.l1_norm))
    master_opt.add_hook(chainer.optimizer_hooks.WeightDecay(stg.model.l2_norm))

    for training, test in dataset:
        tag = training.tag

        # iterators
        train_iter = chainer.iterators.SerialIterator(training, stg.dataset.batch_size // MPI.size,
                                                      repeat=True, shuffle=True)
        test_iter = chainer.iterators.SerialIterator(test, stg.dataset.batch_size // MPI.size,
                                                     repeat=False, shuffle=False)

        # model
        hdnnp = HighDimensionalNNP(training.elemental_composition, stg.model.layer, order=1, mixing_beta=stg.model.mixing_beta)
        hdnnp.sync_param_with(masters)
        main_opt = chainer.Optimizer()
        main_opt = chainermn.create_multi_node_optimizer(main_opt, comm)
        main_opt.setup(hdnnp)

        # triggers
        interval = (stg.model.interval, 'epoch')
        stop_trigger = EarlyStoppingTrigger(check_trigger=interval, monitor=stg.model.metrics,
                                            patients=stg.model.patients, mode='min',
                                            verbose=is_train,
                                            max_trigger=(stg.model.epoch, 'epoch'))

        # updater and trainer
        updater = Updater(train_iter, optimizer={'main': main_opt, 'master': master_opt})
        out_dir = stg.file.out_dir/tag if is_train else stg.file.out_dir
        trainer = chainer.training.Trainer(updater, stop_trigger, out_dir)

        # extensions
        trainer.extend(ext.ExponentialShift('alpha', 1 - stg.model.lr_decay,
                                            target=stg.model.final_lr, optimizer=master_opt))
        evaluator = Evaluator(iterator=test_iter, target=hdnnp)
        trainer.extend(chainermn.create_multi_node_evaluator(evaluator, comm))
        if is_train and MPI.rank == 0:
            trainer.extend(ext.LogReport(log_name='training.log'))
            trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/0th_RMSE', 'main/1st_RMSE', 'main/total_RMSE',
                                            'validation/main/0th_RMSE', 'validation/main/1st_RMSE',
                                            'validation/main/total_RMSE']))
            trainer.extend(scatter_plot(hdnnp, test, order=1), trigger=interval)
            if stg.args.verbose:
                trainer.extend(ext.PlotReport(['main/total_RMSE', 'validation/main/total_RMSE'], 'epoch',
                                              file_name='RMSE.png', marker=None, postprocess=set_log_scale))

        # load trainer snapshot and resume training
        if is_train and stg.args.is_resume:
            if tag != stg.args.resume_dir.name:
                continue
            pprint(f'Resume training loop from dataset tagged "{tag}"')
            trainer_snapshot = stg.args.resume_dir/'trainer_snapshot.npz'
            interim_result = stg.args.resume_dir/'interim_result.pickle'
            chainer.serializers.load_npz(trainer_snapshot, trainer)
            result = pickle.loads(interim_result.read_bytes())
            # remove snapshot
            MPI.comm.Barrier()
            if MPI.rank == 0:
                trainer_snapshot.unlink()
                interim_result.unlink()
            stg.args.is_resume = False

        with Manager(tag, trainer, result, is_train):
            trainer.run()

    return masters, result


def predict(dataset, elements):
    masters = MasterNNP(elements, stg.model.layer)
    chainer.serializers.load_npz(stg.args.masters, masters)

    hdnnp = HighDimensionalNNP(dataset.elemental_composition, stg.model.layer, order=1, mixing_beta=stg.model.mixing_beta)
    hdnnp.sync_param_with(masters)
    predictions = hdnnp.predict(concat_examples(dataset[:]))
    return [prediction.data for prediction in predictions]


def construct_training_datasets(tag_xyz_map, elements):
    if 'all' in stg.dataset.tag:
        included_tags = sorted(tag_xyz_map)
    else:
        included_tags = stg.dataset.tag

    params = {
        'type1': list(product(stg.dataset.Rc)),
        'type2': list(product(stg.dataset.Rc, stg.dataset.eta,
                              stg.dataset.Rs)),
        'type4': list(product(stg.dataset.Rc, stg.dataset.eta,
                              stg.dataset.lambda_, stg.dataset.zeta)),
        }
    preprocesses = []
    preprocess_dir_path = stg.file.out_dir/'preprocess'
    mkdir(preprocess_dir_path)
    for preprocess_name in stg.dataset.preprocess:
        if preprocess_name == 'pca':
            preprocess = PREPROCESS[preprocess_name](stg.dataset.nfeature)
        else:
            preprocess = PREPROCESS[preprocess_name]()
        if stg.args.mode == 'train' and stg.args.is_resume:
            preprocess.load(preprocess_dir_path
                            / f'{preprocess.__class__.__name__}.npz')
        preprocesses.append(preprocess)

    datasets = []
    stg.dataset.nsample = 0
    for tag in included_tags:
        try:
            xyz_path = tag_xyz_map[tag]
        except KeyError:
            pprint(f'Sub dataset tagged as "{tag}" does not exist. Skipped.')
            continue

        pprint(f'Construct sub dataset tagged as "{tag}"')
        dataset = HDNNPDataset(descriptor='symmetry_function',
                               property_='interatomic_potential',
                               order=1)
        structures = [AtomicStructure(atoms) for atoms
                      in ase.io.iread(str(xyz_path), index=':', format='xyz')]

        descriptor_npz_path = xyz_path.with_name('SymmetryFunction.npz')
        if descriptor_npz_path.exists():
            dataset.descriptor_dataset.load(descriptor_npz_path, verbose=True)
        else:
            dataset.descriptor_dataset.make(structures, **params, verbose=True)
            dataset.descriptor_dataset.save(descriptor_npz_path, verbose=True)

        property_npz_path = xyz_path.with_name('InteratomicPotential.npz')
        if property_npz_path.exists():
            dataset.property_dataset.load(property_npz_path, verbose=True)
        else:
            dataset.property_dataset.make(structures, verbose=True)
            dataset.property_dataset.save(property_npz_path, verbose=True)

        dataset.construct(elements, preprocesses, shuffle=True)
        for preprocess in preprocesses:
            preprocess.save(preprocess_dir_path
                            / f'{preprocess.__class__.__name__}.npz')

        dataset.scatter()
        datasets.append(dataset)
        stg.dataset.nsample += dataset.total_size
    pprint()
    return datasets


def construct_test_datasets(tag_xyz_map, elements):
    params = {
        'type1': list(product(stg.dataset.Rc)),
        'type2': list(product(stg.dataset.Rc, stg.dataset.eta,
                              stg.dataset.Rs)),
        'type4': list(product(stg.dataset.Rc, stg.dataset.eta,
                              stg.dataset.lambda_, stg.dataset.zeta)),
        }
    preprocesses = []
    preprocess_dir_path = stg.file.out_dir / 'preprocess'
    for preprocess_name in stg.dataset.preprocess:
        if preprocess_name == 'pca':
            preprocess = PREPROCESS[preprocess_name](stg.dataset.nfeature)
        else:
            preprocess = PREPROCESS[preprocess_name]()
        preprocess.load(preprocess_dir_path
                        / f'{preprocess.__class__.__name__}.npz')
        preprocesses.append(preprocess)

    datasets = []
    for xyz_path in tag_xyz_map.values():
        dataset = HDNNPDataset(descriptor='symmetry_function',
                               order=1)
        structures = [AtomicStructure(atoms) for atoms
                      in ase.io.iread(str(xyz_path), index=':', format='xyz')]

        dataset.descriptor_dataset.make(structures, **params, verbose=False)

        dataset.construct(elements, preprocesses, shuffle=False)

        datasets.append(dataset)

    return datasets


if __name__ == '__main__':
    main()
