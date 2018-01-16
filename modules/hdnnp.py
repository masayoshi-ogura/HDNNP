# -*- coding: utf-8 -*-

# define variables
from config import mpi

# import python modules
from os import path
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.training.extensions as ext

# import own modules
from .data import AtomicStructureDataset
from .data import DataGenerator
from .preconditioning import PRECOND
from .model import SingleNNP, HDNNP
from .updater import HDUpdater
from .util import pprint, flatten_dict
from .extensions import Evaluator
from .extensions import set_logscale
from .extensions import scatterplot


def run(hp, out_dir, log):
    results = []
    # dataset and iterator
    precond = PRECOND[hp.preconditioning](ncomponent=20)
    generator = DataGenerator(hp, precond)
    if hp.mode == 'training':
        precond.save(path.join(out_dir, 'preconditioning.npz'))
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
            log_name = 'cv_{}.log'.format(i) if hp.mode == 'cv' else 'log'
            trainer.extend(ext.ExponentialShift('alpha', 1-hp.lr_decay, target=hp.final_lr, optimizer=master_opt))
            trainer.extend(Evaluator(iterator=val_iter, target=hdnnp, device=mpi.gpu))
            trainer.extend(ext.LogReport(log_name=log_name))
            if log:
                trainer.extend(ext.observe_lr('master', 'learning rate'))
                trainer.extend(ext.PlotReport(['learning rate'], 'epoch',
                                              file_name='learning_rate.png', marker=None, postprocess=set_logscale))
                trainer.extend(ext.PlotReport(['main/tot_RMSE', 'validation/main/tot_RMSE'], 'epoch',
                                              file_name='RMSE.png', marker=None, postprocess=set_logscale))
                trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/RMSE', 'main/d_RMSE', 'main/tot_RMSE',
                                                'validation/main/RMSE', 'validation/main/d_RMSE', 'validation/main/tot_RMSE']))
                trainer.extend(scatterplot(hdnnp, val, config),
                               trigger=chainer.training.triggers.MinValueTrigger(hp.metrics, (100, 'epoch')))
                trainer.extend(ext.snapshot_object(masters, 'masters_snapshot_epoch_{.updater.epoch}.npz'), trigger=(100, 'epoch'))

            trainer.run()
        results.append(flatten_dict(trainer.observation))

    # serialize
    if hp.mode == 'cv':
        result = {k: sum([r[k] for r in results]) / hp.kfold
                  for k in results[0].keys()}
        result['id'] = hp.id
    elif hp.mode == 'training':
        chainer.serializers.save_npz(path.join(out_dir, 'masters.npz'), masters)
        chainer.serializers.save_npz(path.join(out_dir, 'optimizer.npz'), master_opt)
        result, = results
    result['input'] = train._dataset.input.shape[2]
    result['sample'] = len(generator)
    return result


def test(hp, *args):
    if hp.mode == 'optimize':
        scale = optimize(hp, *args, save=False)
        hp.mode = 'phonon'
        phonon(hp, *args, save=False, scale=scale)
    elif hp.mode == 'phonon':
        phonon(hp, *args, save=True)


def optimize(hp, masters_path, *args, **kwargs):
    dirname, basename = path.split(masters_path)
    root, _ = path.splitext(basename)
    energy, force = predict(hp, masters_path, *args, **kwargs)
    nsample = len(energy)
    energy = energy.data.reshape(-1)
    force = np.sqrt((force.data.reshape(nsample, -1)**2).mean(axis=1))
    x = np.linspace(0.9, 1.1, nsample)
    plt.plot(x, energy, label='energy')
    plt.plot(x, force, label='force')
    plt.legend()
    plt.savefig(path.join(dirname, 'optimization_{}.png'.format(root)))
    plt.close()
    pprint('energy-optimized lattice parameter: {}'.format(x[np.argmin(energy)]))
    pprint('force-optimized lattice parameter: {}'.format(x[np.argmin(force)]))
    scale = x[np.argmin(energy)]
    return scale


def phonon(hp, masters_path, *args, **kwargs):
    pprint('drawing phonon band structure ...')
    dirname, basename = path.split(masters_path)
    root, _ = path.splitext(basename)
    dataset, force = predict(hp, masters_path, *args, **kwargs)
    nsample = len(dataset)
    sets_of_forces = force.data.reshape(nsample, 3, -1).transpose(0, 2, 1)
    phonon = dataset.phonopy
    phonon.set_forces(sets_of_forces)
    phonon.produce_force_constants()

    mesh = [8, 8, 8]
    point_symmetry = [[0.0, 0.0, 0.0],  # Gamma
                      [1.0/3, 1.0/3, 0.0],  # K
                      [0.5, 0.0, 0.0],  # M
                      [0.0, 0.0, 0.0],  # Gamma
                      [0.0, 0.0, 0.5],  # A
                      [1.0/3, 1.0/3, 0.5],  # H
                      [0.5, 0.0, 0.5],  # L
                      [0.0, 0.0, 0.5],  # A
                      ]
    labels = ['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A']
    points = 101
    bands = [np.concatenate([np.linspace(si, ei, points).reshape(-1, 1) for si, ei in zip(s, e)], axis=1)
             for s, e in zip(point_symmetry[:-1], point_symmetry[1:])]
    phonon.set_mesh(mesh)
    phonon.set_band_structure(bands, is_band_connection=True)
    plt = phonon.plot_band_structure(labels)
    ax = plt.gca()
    xticks = ax.get_xticks()

    # experimental result measured by IXS and Raman is obtained from
    # Phonon Dispersion Curves in Wurtzite-Structure GaN Determined by Inelastic X-Ray Scattering
    # PRL vol.86 #5 2001/1/29
    # @Gamma, Raman
    ax.scatter([xticks[0]]*6 + [xticks[3]]*6,
               [144.2, 533.5, 560.0, 569.2, 739.3, 746.6]*2,
               marker='o', s=50, facecolors='none', edgecolors='blue')
    # @Gamma, IXS
    ax.scatter([xticks[0]]*3 + [xticks[3]]*3,
               [329, 692, 729]*2,
               marker='o', s=50, facecolors='none', edgecolors='red')
    # @A, IXS
    ax.scatter([xticks[4]]*2 + [xticks[7]]*2,
               [231, 711]*2,
               marker='o', s=50, facecolors='none', edgecolors='red')
    # @M, IXS
    ax.scatter([xticks[2]]*5,
               [137, 184, 193, 238, 576],
               marker='o', s=50, facecolors='none', edgecolors='red')
    # @K, IXS
    ax.scatter([xticks[1]]*2,
               [215, 614],
               marker='o', s=50, facecolors='none', edgecolors='red')

    ax.grid(axis='x')
    plt.savefig(path.join(dirname, 'ph_band_HDNNP_{}.png'.format(root)))
    plt.close()


def predict(hp, masters_path, *args, **kwargs):
    dataset = AtomicStructureDataset(hp)
    dataset.load_poscar(*args, **kwargs)
    precond = PRECOND[hp.preconditioning](ncomponent=20)
    precond.load(path.join(path.dirname(masters_path), 'preconditioning.npz'))
    precond.decompose(dataset)
    masters = chainer.ChainList(*[SingleNNP(hp, element) for element in set(dataset.composition.element)])
    chainer.serializers.load_npz(masters_path, masters)
    hdnnp = HDNNP(hp, dataset.composition)
    hdnnp.sync_param_with(masters)
    energy, force = hdnnp.predict(dataset.input, dataset.dinput)
    if hp.mode == 'optimize':
        return energy, force
    elif hp.mode == 'phonon':
        return dataset, force
