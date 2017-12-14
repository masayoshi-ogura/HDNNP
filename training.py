# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import file_
from config import mpi

# import python modules
from os import path
from os import makedirs
from datetime import datetime
import chainer
import chainer.training.extensions as ext

# import own modules
from modules.data import DataGenerator
from modules.model import SingleNNP, HDNNP
from modules.updater import HDUpdater
from modules.visualize import scatterplot, set_logscale

datestr = datetime.now().strftime('%m%d-%H%M%S')
out_dir = path.join(file_.out_dir, datestr)
makedirs(out_dir)

for i, dataset in enumerate(DataGenerator()):
    # model and optimizer
    masters = chainer.ChainList(*[SingleNNP(element) for element in dataset.composition['index']])
    optimizer = chainer.optimizers.Adam(hp.init_lr)
    optimizer.setup(masters)
    optimizer.add_hook(chainer.optimizer.Lasso(hp.l1_norm))
    optimizer.add_hook(chainer.optimizer.WeightDecay(hp.l2_norm))
    if i != 0:
        chainer.serializers.load_npz(path.join(out_dir, 'masters.npz'), masters)
        chainer.serializers.load_npz(path.join(out_dir, 'optimizer.npz'), optimizer)
    hdnnp = HDNNP(dataset.composition)
    hdnnp.sync_param_with(masters)

    # dataset and iterator
    train_iter = chainer.iterators.SerialIterator(dataset, hp.batch_size)
    # val_iter = chainer.iterators.SerialIterator(val, len(val))

    # updater and trainer
    updater = HDUpdater(hdnnp, iterator=train_iter, optimizer=optimizer, device=mpi.gpu)
    trainer = chainer.training.Trainer(updater, (hp.nepoch, 'epoch'), out=out_dir)

    # extensions
    trainer.extend(ext.ExponentialShift('alpha', 1-hp.lr_decay, target=hp.final_lr, optimizer=optimizer))  # learning rate decay
    trainer.extend(ext.observe_lr())
    # trainer.extend(ext.Evaluator(iterator=val_iter, target=model, device=mpi.gpu, eval_func=))  # evaluate validation dataset
    trainer.extend(ext.LogReport())
    trainer.extend(ext.PlotReport(['loss', 'd_loss', 'total_loss'], file_name='learning.png', marker='o', postprocess=set_logscale))
    trainer.extend(ext.PrintReport(['epoch', 'iteration', 'loss', 'd_loss', 'total_loss', 'lr']))
    trainer.extend(ext.ProgressBar(update_interval=10))
    trainer.extend(scatterplot(hdnnp, dataset), trigger=chainer.training.triggers.MinValueTrigger('total_loss', (10, 'epoch')))

    trainer.run()

    # serialize
    chainer.serializers.save_npz(path.join(out_dir, 'masters.npz'), masters)
    chainer.serializers.save_npz(path.join(out_dir, 'optimizer.npz'), optimizer)
