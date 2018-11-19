# -*- coding: utf-8 -*-

# import python modules
from os import path
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.training.extensions as ext
import chainermn

# import own modules
from . import settings as stg
from .data import DataGenerator
from .model import SingleNNP, HDNNP
from .updater import HDUpdater
from .util import pprint, flatten_dict
from .chainer_extensions import Evaluator
from .chainer_extensions import set_log_scale
from .chainer_extensions import scatter_plot


def training(dataset, elements, output=True):
    trainer = None
    time = 0

    # model and optimizer
    masters = chainer.ChainList(*[SingleNNP(element) for element in elements])
    master_opt = chainer.optimizers.Adam(stg.model.init_lr)
    master_opt = chainermn.create_multi_node_optimizer(master_opt, stg.mpi.chainer_comm)
    master_opt.setup(masters)
    master_opt.add_hook(chainer.optimizer_hooks.Lasso(stg.model.l1_norm))
    master_opt.add_hook(chainer.optimizer_hooks.WeightDecay(stg.model.l2_norm))

    for train, test, composition in dataset:
        train_iter = chainer.iterators.SerialIterator(train, stg.dataset.batch_size // stg.mpi.size,
                                                      repeat=True, shuffle=True)
        test_iter = chainer.iterators.SerialIterator(test, stg.dataset.batch_size // stg.mpi.size,
                                                     repeat=False, shuffle=False)

        hdnnp = HDNNP(composition)
        hdnnp.sync_param_with(masters)
        main_opt = chainer.Optimizer()
        main_opt = chainermn.create_multi_node_optimizer(main_opt, stg.mpi.chainer_comm)
        main_opt.setup(hdnnp)

        # updater and trainer
        updater = HDUpdater(iterator=train_iter, device=stg.mpi.gpu,
                            optimizer={'main': main_opt, 'master': master_opt})
        trainer = chainer.training.Trainer(updater, (stg.dataset.epoch, 'epoch'), out=stg.file.out_dir)

        # extensions
        trainer.extend(ext.ExponentialShift('alpha', 1 - stg.model.lr_decay,
                                            target=stg.model.final_lr, optimizer=master_opt))
        evaluator = Evaluator(iterator=test_iter, target=hdnnp, device=stg.mpi.gpu)
        trainer.extend(chainermn.create_multi_node_evaluator(evaluator, stg.mpi.chainer_comm))
        if stg.mpi.rank == 0 and output:
            config = train.config
            trainer.extend(ext.LogReport(log_name='{}.log'.format(config)))
            trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/RMSE', 'main/d_RMSE', 'main/tot_RMSE',
                                            'validation/main/RMSE', 'validation/main/d_RMSE',
                                            'validation/main/tot_RMSE']))
            trainer.extend(scatter_plot(hdnnp, test, config),
                           trigger=chainer.training.triggers.MinValueTrigger(stg.model.metrics, (5, 'epoch')))
            trainer.extend(ext.snapshot_object(masters, '{}_masters_snapshot.npz'.format(config)),
                           trigger=(stg.dataset.epoch, 'epoch'))
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

        try:
            trainer.run()
        except KeyboardInterrupt:
            if stg.mpi.rank == 0:
                pprint('stop {} training by Keyboard Interrupt!'.format(config))
                chainer.serializers.save_npz(path.join(stg.file.out_dir, '{}_masters_snapshot.npz'.format(config)),
                                             masters)
        time += trainer.elapsed_time

    result = {'input': masters[0].l0.W.shape[1], 'elapsed_time': time}
    result.update(flatten_dict(trainer.observation))
    return masters, result


def predict(save=True):
    generator = DataGenerator(stg.args.poscar, 'poscar', save=save)
    dataset, elements = generator()
    masters = chainer.ChainList(*[SingleNNP(element) for element in elements])
    chainer.serializers.load_npz(stg.args.masters, masters)
    hdnnp = HDNNP(dataset.composition)
    hdnnp.sync_param_with(masters)
    energy, force = hdnnp.predict(dataset.input, dataset.dinput)
    return dataset, energy, force


def optimize():
    name = path.splitext(path.split(stg.args.masters)[1])[0]
    _, energy, force = predict(save=False)
    nsample = len(energy)
    energy = energy.data.reshape(-1)
    force = np.sqrt((force.data ** 2).mean(axis=(1, 2)))
    x = np.linspace(0.9, 1.1, nsample)
    plt.plot(x, energy, label='energy')
    plt.plot(x, force, label='force')
    plt.legend()
    plt.savefig(path.join(stg.file.out_dir, 'optimization_{}.png'.format(name)))
    plt.close()
    return x[np.argmin(energy)], x[np.argmin(force)]


def phonon():
    pprint('drawing phonon band structure ... ', end='')
    name = path.splitext(path.split(stg.args.masters)[1])[0]
    dataset, _, force = predict()
    sets_of_forces = force.data
    phonopy = dataset.phonopy
    phonopy.set_forces(sets_of_forces)
    phonopy.produce_force_constants()

    bands = [np.concatenate([np.linspace(si, ei, stg.phonopy.points).reshape(-1, 1) for si, ei in zip(s, e)], axis=1)
             for s, e in zip(stg.phonopy.point_symmetry[:-1], stg.phonopy.point_symmetry[1:])]
    phonopy.set_mesh(stg.phonopy.mesh)
    phonopy.set_total_DOS()
    phonopy.set_band_structure(bands, is_band_connection=True)
    phonopy_plt = phonopy.plot_band_structure_and_dos(labels=stg.phonopy.labels)
    if 'callback' in dir(stg.phonopy):
        stg.phonopy.callback(phonopy_plt.gcf().axes[0])
    phonopy_plt.savefig(path.join(stg.file.out_dir, 'ph_band_HDNNP_{}.png'.format(name)))
    phonopy_plt.close()
    pprint('done')
