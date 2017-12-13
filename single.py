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
from modules.data import get_simple_function
from modules.model import SingleNNP
from modules.updater import Updater
from modules.visualize import scatterplot, set_logscale

name = 'complex'
datestr = datetime.now().strftime('%m%d-%H%M%S')
out_dir = path.join(file_.out_dir, datestr)
makedirs(out_dir)
copy2('config.py', path.join(out_dir, 'config.py'))

dataset = get_simple_function(name)

# model and optimizer
model = SingleNNP(name)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# dataset and iterator
train_iter = chainer.iterators.SerialIterator(dataset, hp.batch_size)
# val_iter = chainer.iterators.SerialIterator(val, len(val))

# updater and trainer
updater = Updater(iterator=train_iter, optimizer=optimizer, device=mpi.gpu)
trainer = chainer.training.Trainer(updater, (hp.nepoch, 'epoch'), out=out_dir)

# trainer.extend(extensions.ExponentialShift('alpha', 0.999, target=0.0001, optimizer=optimizer))  # learning rate decay
# trainer.extend(extensions.Evaluator(iterator=val_iter, target=model, device=mpi.gpu, eval_func=))  # evaluate validation dataset
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PlotReport(['loss', 'd_loss', 'total_loss'], file_name='learning.png', marker='o', postprocess=set_logscale))
trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'd_loss', 'total_loss']))
trainer.extend(extensions.ProgressBar(update_interval=10))
trainer.extend(scatterplot(model, dataset, out_dir), trigger=chainer.training.triggers.MinValueTrigger('total_loss', (10, 'epoch')))

trainer.run()
