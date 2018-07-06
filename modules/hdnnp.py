# -*- coding: utf-8 -*-

# define variables
import settings as stg

# import python modules
from os import path
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.training.extensions as ext
import chainermn

# import own modules
from .data import AtomicStructureDataset
from .preproc import PREPROC
from .model import SingleNNP, HDNNP
from .updater import HDUpdater
from .util import pprint, flatten_dict
from .extensions import Evaluator
from .extensions import set_log_scale
from .extensions import scatter_plot


def training(model_hp, dataset, elements, out_dir):
    trainer = None
    time = 0

    # model and optimizer
    masters = chainer.ChainList(*[SingleNNP(model_hp, element) for element in elements])
    master_opt = chainer.optimizers.Adam(model_hp.init_lr)
    master_opt = chainermn.create_multi_node_optimizer(master_opt, stg.mpi.chainer_comm)
    master_opt.setup(masters)
    master_opt.add_hook(chainer.optimizer_hooks.Lasso(model_hp.l1_norm))
    master_opt.add_hook(chainer.optimizer_hooks.WeightDecay(model_hp.l2_norm))

    for train, test, composition in dataset:
        train_iter = chainer.iterators.SerialIterator(chainermn.scatter_dataset(train, stg.mpi.chainer_comm),
                                                      model_hp.batch_size / stg.mpi.size)
        test_iter = chainer.iterators.SerialIterator(chainermn.scatter_dataset(test, stg.mpi.chainer_comm),
                                                     model_hp.batch_size / stg.mpi.size,
                                                     repeat=False, shuffle=False)

        hdnnp = HDNNP(model_hp, composition)
        hdnnp.sync_param_with(masters)
        main_opt = chainer.Optimizer()
        main_opt = chainermn.create_multi_node_optimizer(main_opt, stg.mpi.chainer_comm)
        main_opt.setup(hdnnp)

        # updater and trainer
        updater = HDUpdater(iterator=train_iter, device=stg.mpi.gpu,
                            optimizer={'main': main_opt, 'master': master_opt})
        trainer = chainer.training.Trainer(updater, (model_hp.epoch, 'epoch'), out=out_dir)

        # extensions
        trainer.extend(ext.ExponentialShift('alpha', 1 - model_hp.lr_decay,
                                            target=model_hp.final_lr, optimizer=master_opt))
        evaluator = Evaluator(iterator=test_iter, target=hdnnp, device=stg.mpi.gpu)
        trainer.extend(chainermn.create_multi_node_evaluator(evaluator, stg.mpi.chainer_comm))
        if stg.mpi.rank == 0:
            config = train.config
            trainer.extend(ext.LogReport(log_name='{}.log'.format(config)))
            trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/RMSE', 'main/d_RMSE', 'main/tot_RMSE',
                                            'validation/main/RMSE', 'validation/main/d_RMSE',
                                            'validation/main/tot_RMSE']))
            trainer.extend(scatter_plot(hdnnp, test, config),
                           trigger=chainer.training.triggers.MinValueTrigger(model_hp.metrics, (5, 'epoch')))
            trainer.extend(ext.snapshot_object(masters, '{}_masters_snapshot.npz'.format(config)),
                           trigger=(stg.model.epoch, 'epoch'))
            if stg.args.verbose:
                # trainer.extend(ext.observe_lr('master', 'learning rate'))
                # trainer.extend(ext.PlotReport(['learning rate'], 'epoch',
                #                               file_name='learning_rate.png',
                #                               marker=None,
                #                               postprocess=set_log_scale))
                trainer.extend(ext.PlotReport(['main/tot_RMSE', 'validation/main/tot_RMSE'], 'epoch',
                                              file_name='{}_RMSE.png'.format(config), marker=None,
                                              postprocess=set_log_scale))
                trainer.extend(
                    ext.snapshot_object(masters, config + '_masters_snapshot_epoch_{.updater.epoch}.npz'),
                    trigger=(100, 'epoch'))

        trainer.run()
        time += trainer.elapsed_time

    result = {'input': masters[0].l0.W.shape[1], 'elapsed_time': time}
    result.update(flatten_dict(trainer.observation))
    return masters, result


def predict(sf_hp, model_hp, masters_path, poscar, save=True):
    dataset = AtomicStructureDataset(sf_hp, poscar, file_format='POSCAR', save=save)
    preproc = PREPROC[stg.model.preproc]()
    preproc.load(path.join(path.dirname(masters_path), 'preproc.npz'))
    preproc.decompose(dataset)
    masters = chainer.ChainList(*[SingleNNP(model_hp, element) for element in set(dataset.composition.element)])
    chainer.serializers.load_npz(masters_path, masters)
    hdnnp = HDNNP(model_hp, dataset.composition)
    hdnnp.sync_param_with(masters)
    energy, force = hdnnp.predict(dataset.input, dataset.dinput)
    return dataset, energy, force


def optimize(sf_hp, model_hp, masters_path, poscar):
    dirname, basename = path.split(masters_path)
    root, _ = path.splitext(basename)
    _, energy, force = predict(sf_hp, model_hp, masters_path, poscar, save=False)
    nsample = len(energy)
    energy = energy.data.reshape(-1)
    force = np.sqrt((force.data ** 2).mean(axis=(1, 2)))
    x = np.linspace(0.9, 1.1, nsample)
    plt.plot(x, energy, label='energy')
    plt.plot(x, force, label='force')
    plt.legend()
    plt.savefig(path.join(dirname, 'optimization_{}.png'.format(root)))
    plt.close()
    return x[np.argmin(energy)], x[np.argmin(force)]


def phonon(sf_hp, model_hp, masters_path, poscar):
    pprint('drawing phonon band structure ...')
    dirname, basename = path.split(masters_path)
    root, _ = path.splitext(basename)
    dataset, _, force = predict(sf_hp, model_hp, masters_path, poscar)
    sets_of_forces = force.data
    phonopy = dataset.phonopy
    phonopy.set_forces(sets_of_forces)
    phonopy.produce_force_constants()

    mesh = [8, 8, 8]
    point_symmetry = [[0.0, 0.0, 0.0],  # Gamma
                      [1.0 / 3, 1.0 / 3, 0.0],  # K
                      [0.5, 0.0, 0.0],  # M
                      [0.0, 0.0, 0.0],  # Gamma
                      [0.0, 0.0, 0.5],  # A
                      [1.0 / 3, 1.0 / 3, 0.5],  # H
                      [0.5, 0.0, 0.5],  # L
                      [0.0, 0.0, 0.5],  # A
                      ]
    labels = ['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A']
    points = 101
    bands = [np.concatenate([np.linspace(si, ei, points).reshape(-1, 1) for si, ei in zip(s, e)], axis=1)
             for s, e in zip(point_symmetry[:-1], point_symmetry[1:])]
    phonopy.set_mesh(mesh)
    phonopy.set_total_DOS()
    phonopy.set_band_structure(bands, is_band_connection=True)
    phonopy_plt = phonopy.plot_band_structure_and_dos(labels=labels)
    ax = phonopy_plt.gcf().axes[0]
    xticks = ax.get_xticks()

    # experimental result measured by IXS and Raman is obtained from
    # Phonon Dispersion Curves in Wurtzite-Structure GaN Determined by Inelastic X-Ray Scattering
    # PRL vol.86 #5 2001/1/29
    # @Gamma, Raman
    ax.scatter([xticks[0]] * 6 + [xticks[3]] * 6,
               [144.2, 533.5, 560.0, 569.2, 739.3, 746.6] * 2,
               marker='o', s=50, facecolors='none', edgecolors='blue')
    # @Gamma, IXS
    ax.scatter([xticks[0]] * 3 + [xticks[3]] * 3,
               [329, 692, 729] * 2,
               marker='o', s=50, facecolors='none', edgecolors='red')
    # @A, IXS
    ax.scatter([xticks[4]] * 2 + [xticks[7]] * 2,
               [231, 711] * 2,
               marker='o', s=50, facecolors='none', edgecolors='red')
    # @M, IXS
    ax.scatter([xticks[2]] * 5,
               [137, 184, 193, 238, 576],
               marker='o', s=50, facecolors='none', edgecolors='red')
    # @K, IXS
    ax.scatter([xticks[1]] * 2,
               [215, 614],
               marker='o', s=50, facecolors='none', edgecolors='red')

    ax.grid(axis='x')
    phonopy_plt.savefig(path.join(dirname, 'ph_band_HDNNP_{}.png'.format(root)))
    phonopy_plt.close()


def dump(file_path, preproc, masters):
    nelements = len(masters)
    depth = len(masters[0])
    with open(file_path, 'w') as f:
        f.write('# title\nneural network potential trained by HDNNP\n\n')
        f.write('# symmetry function parameters\n{}\n{}\n{}\n{}\n{}\n\n'
                .format(' '.join(map(str, stg.sym_func.Rc)),
                        ' '.join(map(str, stg.sym_func.eta)),
                        ' '.join(map(str, stg.sym_func.Rs)),
                        ' '.join(map(str, stg.sym_func.lambda_)),
                        ' '.join(map(str, stg.sym_func.zeta))))

        if stg.model.preproc is None:
            f.write('# preprocess parameters\n0\n\n')
        elif stg.model.preproc == 'pca':
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
                        .format(masters[i].element, j + 1, W.shape[1], W.shape[0], stg.model.layer[j].activation))
                f.write('# weight\n')
                for row in W.T:
                    f.write('{}\n'.format(' '.join(map(str, row))))
                f.write('# bias\n')
                f.write('{}\n\n'.format(' '.join(map(str, b))))
