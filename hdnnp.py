# -*- coding: utf-8 -*-

# define variables
from config import mpi

# import python modules
from os import path
import chainer
from chainer import Variable
import chainer.training.extensions as ext

# import own modules
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
            trainer.extend(ext.ExponentialShift('alpha', 1-hp.lr_decay, target=hp.final_lr, optimizer=master_opt))  # learning rate decay
            trainer.extend(Evaluator(iterator=val_iter, target=hdnnp, device=mpi.gpu))  # evaluate validation dataset
            trainer.extend(ext.LogReport(log_name='cv_{}.log'.format(i)))
            # trainer.extend(ext.PlotReport(['main/tot_RMSE', 'validation/main/tot_RMSE'], 'epoch',
            #                               file_name='learning.png', marker=None, postprocess=set_logscale))
            # trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/RMSE', 'main/d_RMSE', 'main/tot_RMSE',
            #                                 'validation/main/RMSE', 'validation/main/d_RMSE', 'validation/main/tot_RMSE']))
            # trainer.extend(ext.ProgressBar(update_interval=10))
            # trainer.extend(scatterplot(hdnnp, val, config),
            #                trigger=chainer.training.triggers.MinValueTrigger('validation/main/tot_RMSE', (10, 'epoch')))

            trainer.run()
        results.append(trainer.observation)

    # serialize
    if not hp.cross_validation:
        chainer.serializers.save_npz(path.join(out_dir, 'masters.npz'), masters)
        chainer.serializers.save_npz(path.join(out_dir, 'optimizer.npz'), master_opt)

    result = {k: sum([r[k].data.item() if isinstance(r[k], Variable) else r[k].item() for r in results]) / hp.cross_validation
              for k in results[0].keys()}
    result['id'] = hp.id
    result['sample'] = len(generator)
    return result
