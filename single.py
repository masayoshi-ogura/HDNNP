# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import file_
from config import mpi

# import python modules
from sys import argv
from os import path
from os import makedirs
from shutil import copy2
from datetime import datetime
import chainer
import chainer.training.extensions as ext

# import own modules
from modules.data import get_simple_function
from modules.model import SingleNNP
from modules.updater import Updater
from modules.visualize import scatterplot, set_logscale

name = argv[1]
datestr = datetime.now().strftime('%m%d-%H%M%S')
out_dir = path.join(file_.out_dir, datestr)
makedirs(out_dir)
copy2('config.py', path.join(out_dir, 'config.py'))

dataset = get_simple_function(name)

# model and optimizer
model = SingleNNP(name)
optimizer = chainer.optimizers.Adam(hp.init_lr)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.Lasso(hp.l1_norm))
optimizer.add_hook(chainer.optimizer.WeightDecay(hp.l2_norm))

# dataset and iterator
train_iter = chainer.iterators.SerialIterator(dataset, hp.batch_size)
# val_iter = chainer.iterators.SerialIterator(val, len(val))

# updater and trainer
updater = Updater(iterator=train_iter, optimizer=optimizer, device=mpi.gpu)
trainer = chainer.training.Trainer(updater, (hp.nepoch, 'epoch'), out=out_dir)

# extensions
trainer.extend(ext.ExponentialShift('alpha', 1-hp.lr_decay, target=hp.final_lr, optimizer=optimizer))  # learning rate decay
trainer.extend(ext.observe_lr())
# trainer.extend(ext.Evaluator(iterator=val_iter, target=model, device=mpi.gpu, eval_func=))  # evaluate validation dataset
trainer.extend(ext.LogReport())
trainer.extend(ext.PlotReport(['loss', 'd_loss', 'total_loss'], file_name='learning.png', marker='o', postprocess=set_logscale))
trainer.extend(ext.PrintReport(['epoch', 'iteration', 'loss', 'd_loss', 'total_loss', 'lr']))
trainer.extend(ext.ProgressBar(update_interval=10))
trainer.extend(scatterplot(model, dataset), trigger=chainer.training.triggers.MinValueTrigger('total_loss', (10, 'epoch')))

trainer.run()
