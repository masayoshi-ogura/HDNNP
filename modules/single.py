# -*- coding: utf-8 -*-

# define variables
from config import mpi

# import python modules
import chainer
import chainer.training.extensions as ext

# import own modules
from .data import get_simple_function
from .model import SingleNNP
from .updater import Updater
from .util import flatten_dict
from .extensions import Evaluator
from .extensions import set_logscale
from .extensions import scatterplot


def run(hp, out_dir, log):
    results = []
    # dataset and iterator
    dataset = get_simple_function(hp.single)
    if hp.mode == 'cv':
        generator = chainer.datasets.get_cross_validation_datasets_random(dataset, n_fold=hp.kfold)
    elif hp.mode == 'training':
        generator = [chainer.datasets.split_dataset_random(dataset, len(dataset)*9/10)]
    for i, (train, val) in enumerate(generator):
        train_iter = chainer.iterators.SerialIterator(train, hp.batch_size)
        val_iter = chainer.iterators.SerialIterator(val, hp.batch_size, repeat=False, shuffle=False)

        # model and optimizer
        model = SingleNNP(hp, hp.single)
        optimizer = chainer.optimizers.Adam(hp.init_lr)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer_hooks.Lasso(hp.l1_norm))
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(hp.l2_norm))

        # updater and trainer
        updater = Updater(iterator=train_iter, optimizer=optimizer, device=mpi.gpu)
        trainer = chainer.training.Trainer(updater, (hp.epoch, 'epoch'), out=out_dir)

        # extensions
        log_name = 'cv_{}.log'.format(i) if hp.mode == 'cv' else 'log'
        trainer.extend(ext.ExponentialShift('alpha', 1-hp.lr_decay, target=hp.final_lr, optimizer=optimizer))
        trainer.extend(Evaluator(iterator=val_iter, target=model, device=mpi.gpu))
        trainer.extend(ext.LogReport(log_name=log_name))
        if log:
            trainer.extend(ext.observe_lr('main', 'learning rate'))
            trainer.extend(ext.PlotReport(['learning rate'], 'epoch',
                                          file_name='learning_rate.png', marker=None, postprocess=set_logscale))
            trainer.extend(ext.PlotReport(['main/tot_RMSE', 'validation/main/tot_RMSE'], 'epoch',
                                          file_name='RMSE.png', marker=None, postprocess=set_logscale))
            trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/RMSE', 'main/d_RMSE', 'main/tot_RMSE',
                                            'validation/main/RMSE', 'validation/main/d_RMSE', 'validation/main/tot_RMSE']))
            trainer.extend(scatterplot(model, val, hp.single),
                           trigger=chainer.training.triggers.MinValueTrigger(hp.metrics, (10, 'epoch')))

        trainer.run()
        results.append(flatten_dict(trainer.observation))

    if hp.mode == 'cv':
        result = {k: sum([r[k] for r in results]) / hp.kfold
                  for k in results[0].keys()}
        result['id'] = hp.id
    elif hp.mode == 'training':
        result, = results
    result['sample'], result['input'] = dataset._datasets[0].shape
    return result
