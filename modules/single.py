# -*- coding: utf-8 -*-

# define variables
from config import mpi

# import python modules
import chainer
from chainer import Variable
import chainer.training.extensions as ext

# import own modules
from .data import get_simple_function
from .model import SingleNNP
from .updater import Updater
from .extensions import Evaluator
from .extensions import set_logscale
from .extensions import scatterplot


def run(hp, out_dir, log):
    results = []
    # dataset and iterator
    dataset = get_simple_function(hp.single)
    for i, (train, val) in enumerate(chainer.datasets.get_cross_validation_datasets_random(dataset, n_fold=hp.kfold)):
        train_iter = chainer.iterators.SerialIterator(train, hp.batch_size)
        val_iter = chainer.iterators.SerialIterator(val, hp.batch_size, repeat=False, shuffle=False)

        # model and optimizer
        model = SingleNNP(hp, hp.single)
        optimizer = chainer.optimizers.Adam(hp.init_lr)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.Lasso(hp.l1_norm))
        optimizer.add_hook(chainer.optimizer.WeightDecay(hp.l2_norm))

        # updater and trainer
        updater = Updater(iterator=train_iter, optimizer=optimizer, device=mpi.gpu)
        trainer = chainer.training.Trainer(updater, (hp.epoch, 'epoch'), out=out_dir)

        # extensions
        log_name = 'cv_{}.log'.format(i) if hp.mode == 'cv' else 'log'
        trainer.extend(ext.ExponentialShift('alpha', 1-hp.lr_decay, target=hp.final_lr, optimizer=optimizer))
        trainer.extend(Evaluator(iterator=val_iter, target=model, device=mpi.gpu))
        trainer.extend(ext.LogReport(log_name=log_name))
        if log:
            trainer.extend(ext.PlotReport(['main/tot_RMSE', 'validation/main/tot_RMSE'], 'epoch',
                                          file_name='learning.png', marker=None, postprocess=set_logscale))
            trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/RMSE', 'main/d_RMSE', 'main/tot_RMSE',
                                            'validation/main/RMSE', 'validation/main/d_RMSE', 'validation/main/tot_RMSE']))
            trainer.extend(scatterplot(model, val, hp.single),
                           trigger=chainer.training.triggers.MinValueTrigger(hp.metrics, (10, 'epoch')))

        trainer.run()
        results.append(trainer.observation)

    if hp.mode == 'cv':
        result = {k: sum([r[k].data.item() if isinstance(r[k], Variable) else r[k].item() for r in results]) / hp.kfold
                  for k in results[0].keys()}
    elif hp.mode == 'training':
        result = {k: v.data.item() if isinstance(v, Variable) else v.item() for k, v in results[0].items()}
    result['id'] = hp.id
    result['sample'], result['input'] = dataset._datasets[0].shape
    return result
