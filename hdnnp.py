# -*- coding: utf-8 -*-

# define variables
from config import mpi

# import python modules
from os import path
import dill
import numpy as np
import chainer
from chainer import Variable
import chainer.training.extensions as ext
from chainer.dataset import concat_examples

# import own modules
from modules.data import AtomicStructureDataset
from modules.data import DataGenerator
from modules.model import SingleNNP, HDNNP
from modules.updater import HDUpdater
from modules.extensions import Evaluator
from modules.extensions import set_logscale
from modules.extensions import scatterplot


def run(hp, out_dir):
    results = []
    # dataset and iterator
    generator = DataGenerator(hp)
    for i, (dataset, elements) in enumerate(generator):
        # model and optimizer
        masters = chainer.ChainList(*[SingleNNP(hp, element) for element in elements])
        master_opt = chainer.optimizers.Adam(hp.init_lr)
        master_opt.setup(masters)
        master_opt.add_hook(chainer.optimizer.Lasso(hp.l1_norm))
        master_opt.add_hook(chainer.optimizer.WeightDecay(hp.l2_norm))

        for train, val, config, composition in dataset:
            train_iter = chainer.iterators.SerialIterator(train, hp.batch_size)
            val_iter = chainer.iterators.SerialIterator(val, hp.batch_size, repeat=False, shuffle=False)

            hdnnp = HDNNP(hp, composition)
            hdnnp.sync_param_with(masters)
            main_opt = chainer.Optimizer()
            main_opt.setup(hdnnp)

            # updater and trainer
            updater = HDUpdater(iterator=train_iter, optimizer={'main': main_opt, 'master': master_opt}, device=mpi.gpu)
            trainer = chainer.training.Trainer(updater, (hp.epoch, 'epoch'), out=out_dir)

            # extensions
            log_name = 'cv_{}.log'.format(i) if hp.cross_validation else 'log'
            trainer.extend(ext.ExponentialShift('alpha', 1-hp.lr_decay, target=hp.final_lr, optimizer=master_opt))  # learning rate decay
            trainer.extend(Evaluator(iterator=val_iter, target=hdnnp, device=mpi.gpu))  # evaluate validation dataset
            trainer.extend(ext.LogReport(log_name=log_name))
            if not hp.cross_validation:
                trainer.extend(ext.PlotReport(['main/tot_RMSE', 'validation/main/tot_RMSE'], 'epoch',
                                              file_name='learning.png', marker=None, postprocess=set_logscale))
                trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/RMSE', 'main/d_RMSE', 'main/tot_RMSE',
                                                'validation/main/RMSE', 'validation/main/d_RMSE', 'validation/main/tot_RMSE']))
                # trainer.extend(ext.ProgressBar(update_interval=10))
                trainer.extend(scatterplot(hdnnp, val, config),
                               trigger=chainer.training.triggers.MinValueTrigger('validation/main/tot_RMSE', (100, 'epoch')))

            trainer.run()
        results.append(trainer.observation)

    # serialize
    if not hp.cross_validation:
        generator.save(out_dir)
        chainer.serializers.save_npz(path.join(out_dir, 'masters.npz'), masters)
        chainer.serializers.save_npz(path.join(out_dir, 'optimizer.npz'), master_opt)
        result = {k: v.data.item() if isinstance(v, Variable) else v.item() for k, v in results[0]}
    else:
        result = {k: sum([r[k].data.item() if isinstance(r[k], Variable) else r[k].item() for r in results]) / hp.cross_validation
                  for k in results[0].keys()}
        result['id'] = hp.id
    result['sample'] = len(generator)
    return result


def test(hp, model_dir, poscar):
    dataset = AtomicStructureDataset(hp)
    dataset.load_poscar(poscar)
    with open(path.join(model_dir, 'preconditioning.dill'), 'r') as f:
        precond = dill.load(f)
    precond.decompose(dataset)

    nsample = len(dataset)
    iter = chainer.iterators.SerialIterator(dataset, nsample, repeat=False, shuffle=False)
    batch = concat_examples(iter.next(), device=mpi.gpu)
    masters = chainer.ChainList(*[SingleNNP(hp, element) for element in set(dataset.composition.element)])
    chainer.serializers.load_npz(path.join(model_dir, 'masters.npz'), masters)
    hdnnp = HDNNP(hp, dataset.composition)
    hdnnp.sync_param_with(masters)
    energy, force = hdnnp.predict(*batch)
    sets_of_forces = force.data.reshape(nsample, 3, -1).transpose(0, 2, 1)
    phonon = dataset.phonopy
    phonon.set_forces(sets_of_forces)
    phonon.produce_force_constants()

    mesh = [8, 8, 8]
    point_symmetry = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]
    points = 101
    bands = [np.concatenate([np.linspace(si, ei, points).reshape(-1, 1) for si, ei in zip(s, e)], axis=1)
             for s, e in zip(point_symmetry[:-1], point_symmetry[1:])]
    phonon.set_mesh(mesh)
    phonon.set_band_structure(bands)
    return phonon
