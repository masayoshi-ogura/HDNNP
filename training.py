# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import file_
from config import mpi

# import python modules
from os import path
from os import makedirs
from shutil import copy2
from datetime import datetime
import chainer
from chainer.training import extensions

# import own modules
from modules.data import DataGenerator
from modules.model import SingleNNP, HDNNP
from modules.updater import HDUpdater
from modules.visualize import scatterplot, set_logscale

datestr = datetime.now().strftime('%m%d-%H%M%S')
out_dir = path.join(file_.out_dir, datestr)
makedirs(out_dir)
copy2('config.py', path.join(out_dir, 'config.py'))

for i, dataset in enumerate(DataGenerator()):
    # model and optimizer
    masters = {}
    optimizers = {}
    for element in dataset.composition['index'].keys():
        master = SingleNNP(element)
        optimizer = chainer.optimizers.SGD()
        if i != 0:
            chainer.serializers.load_npz(path.join(out_dir, '{}.model'.format(element)), master)
            chainer.serializers.load_npz(path.join(out_dir, '{}.optimizer'.format(element)), optimizer)
        optimizer.setup(master)
        masters[element] = master
        optimizers[element] = optimizer
    hdnnp = HDNNP(dataset.composition)
    hdnnp.sync_master(masters)

    # dataset and iterator
    train_iter = chainer.iterators.SerialIterator(dataset, hp.batch_size)
    # val_iter = chainer.iterators.SerialIterator(val, len(val))

    # updater and trainer
    updater = HDUpdater(hdnnp, iterator=train_iter, optimizer=optimizers, device=mpi.gpu)
    trainer = chainer.training.Trainer(updater, (hp.nepoch, 'epoch'), out=out_dir)

    trainer.extend(extensions.ExponentialShift('lr', 0.9999, target=0.0001, optimizer=optimizers['Ga']))  # learning rate decay
    trainer.extend(extensions.observe_lr('Ga'))
    # trainer.extend(extensions.Evaluator(iterator=val_iter, target=model, device=mpi.gpu, eval_func=))  # evaluate validation dataset
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['loss', 'd_loss', 'total_loss'], file_name='learning.png', marker='o', postprocess=set_logscale))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'd_loss', 'total_loss', 'lr']))
    trainer.extend(extensions.ProgressBar(update_interval=100))
    trainer.extend(scatterplot(hdnnp, dataset, out_dir), trigger=chainer.training.triggers.MinValueTrigger('total_loss', (10, 'epoch')))
    trainer.run()

    for element, optimizer in optimizers.items():
        chainer.serializers.save_npz(path.join(out_dir, '{}.model'.format(element)), optimizer.target)
        chainer.serializers.save_npz(path.join(out_dir, '{}.optimizer'.format(element)), optimizer)
