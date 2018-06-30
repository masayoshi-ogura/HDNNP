# -*- coding: utf-8 -*-

# define variables
from config import mpi

# import python modules
from os import path
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.training.extensions as ext
import chainermn

# import own modules
from .data import AtomicStructureDataset
from .preconditioning import PRECOND
from .model import SingleNNP, HDNNP
from .updater import HDUpdater
from .util import pprint, flatten_dict
from .extensions import Evaluator
from .extensions import set_log_scale
from .extensions import scatter_plot


def run(hp, generator, out_dir, verbose, comm=None):
    trainer = masters = master_opt = None
    assert hp.mode in ['training', 'cv']
    result_list = []
    time = 0
    # dataset and iterator
    for i, (dataset, elements) in enumerate(generator):
        # model and optimizer
        masters = chainer.ChainList(*[SingleNNP(hp, element) for element in elements])
        master_opt = chainer.optimizers.Adam(hp.init_lr)
        if comm:
            master_opt = chainermn.create_multi_node_optimizer(master_opt, comm)
        master_opt.setup(masters)
        master_opt.add_hook(chainer.optimizer_hooks.Lasso(hp.l1_norm))
        master_opt.add_hook(chainer.optimizer_hooks.WeightDecay(hp.l2_norm))

        for train, val, config, composition in dataset:
            train_iter = chainer.iterators.SerialIterator(train, hp.batch_size / mpi.size)
            val_iter = chainer.iterators.SerialIterator(val, hp.batch_size / mpi.size, repeat=False, shuffle=False)

            hdnnp = HDNNP(hp, composition)
            hdnnp.sync_param_with(masters)
            main_opt = chainer.Optimizer()
            if comm:
                main_opt = chainermn.create_multi_node_optimizer(main_opt, comm)
            main_opt.setup(hdnnp)

            # updater and trainer
            updater = HDUpdater(iterator=train_iter, optimizer={'main': main_opt, 'master': master_opt}, device=mpi.gpu)
            trainer = chainer.training.Trainer(updater, (hp.epoch, 'epoch'), out=out_dir)

            # extensions
            log_name = '{}_cv_{}.log'.format(config, i) if hp.mode == 'cv' else '{}.log'.format(config)
            trainer.extend(ext.ExponentialShift('alpha', 1 - hp.lr_decay, target=hp.final_lr, optimizer=master_opt))
            evaluator = Evaluator(iterator=val_iter, target=hdnnp, device=mpi.gpu)
            if comm:
                evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
            trainer.extend(evaluator)
            if mpi.rank == 0:
                trainer.extend(ext.LogReport(log_name=log_name))
                trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/RMSE', 'main/d_RMSE', 'main/tot_RMSE',
                                                'validation/main/RMSE', 'validation/main/d_RMSE',
                                                'validation/main/tot_RMSE']))
                trainer.extend(scatter_plot(hdnnp, val, config),
                               trigger=chainer.training.triggers.MinValueTrigger(hp.metrics, (100, 'epoch')))
                if verbose:
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
            if hp.mode == 'training' and mpi.rank == 0:
                chainer.serializers.save_npz(path.join(out_dir, '{}_masters_snapshot.npz'.format(config)), masters)
                chainer.serializers.save_npz(path.join(out_dir, '{}_optimizer_snapshot.npz'.format(config)), master_opt)
                dump(hp, path.join(out_dir, '{}_snapshot_lammps.nnp'.format(config)), generator.precond, masters)
        else:
            result_list.append(flatten_dict(trainer.observation))

    else:
        result = {'input': masters[0].l0.W.shape[1], 'sample': len(generator), 'elapsed_time': time}
        if hp.mode == 'cv':
            result['id'] = hp.id
            result.update({k: sum([r[k] for r in result_list]) / hp.kfold
                           for k in result_list[0].keys()})
            return result

        elif hp.mode == 'training':
            if mpi.rank == 0:
                chainer.serializers.save_npz(path.join(out_dir, 'masters.npz'), masters)
                chainer.serializers.save_npz(path.join(out_dir, 'optimizer.npz'), master_opt)
                dump(hp, path.join(out_dir, 'lammps.nnp'), generator.precond, masters)
            result.update(result_list[0])
            return result


def test(hp, *args):
    if hp.mode == 'optimize':
        scale = optimize(hp, *args, save=False)
        hp.mode = 'phonon'
        phonon(hp, *args, save=False, scale=scale)
    elif hp.mode == 'phonon':
        phonon(hp, *args, save=True)
    elif hp.mode == 'test':
        energy, force = predict(hp, *args)
        pprint('energy:\n{}'.format(energy.data))
        pprint('force:\n{}'.format(force.data))


def optimize(hp, masters_path, *args, **kwargs):
    dirname, basename = path.split(masters_path)
    root, _ = path.splitext(basename)
    energy, force = predict(hp, masters_path, *args, **kwargs)
    nsample = len(energy)
    energy = energy.data.reshape(-1)
    force = np.sqrt((force.data ** 2).mean(axis=(1, 2)))
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


def predict(hp, masters_path, *args, **kwargs):
    dataset = AtomicStructureDataset(hp, *args, file_format='POSCAR', **kwargs)
    precond = PRECOND[hp.preconditioning]()
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
    elif hp.mode == 'test':
        return energy, force


def dump(hp, file_path, precond, masters):
    nelements = len(masters)
    depth = len(masters[0])
    with open(file_path, 'w') as f:
        f.write('# title\nneural network potential trained by HDNNP\n\n')
        f.write('# symmetry function parameters\n{}\n{}\n{}\n{}\n{}\n\n'
                .format(' '.join(map(str, hp.Rc)),
                        ' '.join(map(str, hp.eta)),
                        ' '.join(map(str, hp.Rs)),
                        ' '.join(map(str, hp.lambda_)),
                        ' '.join(map(str, hp.zeta))))

        if hp.preconditioning == 'none':
            f.write('# preconditioning parameters\n0\n\n')
        elif hp.preconditioning == 'pca':
            f.write('# preconditioning parameters\n1\npca\n\n')
            for i in range(nelements):
                element = masters[i].element
                components = precond.components[element]
                mean = precond.mean[element]
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
                        .format(masters[i].element, j + 1, W.shape[1], W.shape[0], hp.layer[j].activation))
                f.write('# weight\n')
                for row in W.T:
                    f.write('{}\n'.format(' '.join(map(str, row))))
                f.write('# bias\n')
                f.write('{}\n\n'.format(' '.join(map(str, b))))
