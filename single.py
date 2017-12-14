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
from modules.extensions import Evaluator, scatterplot, set_logscale

name = argv[1]
datestr = datetime.now().strftime('%m%d-%H%M%S')
out_dir = path.join(file_.out_dir, datestr)
makedirs(out_dir)
copy2('config.py', path.join(out_dir, 'config.py'))

# dataset and iterator
dataset = get_simple_function(name)
train, val = chainer.datasets.split_dataset_random(dataset, int(0.8*len(dataset)))
# for train, val in chainer.datasets.get_cross_validation_datasets_random(dataset, n_fold=hp.kfold_cv):
train_iter = chainer.iterators.SerialIterator(train, hp.batch_size)
val_iter = chainer.iterators.SerialIterator(val, hp.batch_size, repeat=False, shuffle=False)

# model and optimizer
model = SingleNNP(name)
optimizer = chainer.optimizers.Adam(hp.init_lr)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.Lasso(hp.l1_norm))
optimizer.add_hook(chainer.optimizer.WeightDecay(hp.l2_norm))

# updater and trainer
updater = Updater(iterator=train_iter, optimizer=optimizer, device=mpi.gpu)
trainer = chainer.training.Trainer(updater, (hp.nepoch, 'epoch'), out=out_dir)

# extensions
trainer.extend(ext.ExponentialShift('alpha', 1-hp.lr_decay, target=hp.final_lr, optimizer=optimizer))  # learning rate decay
trainer.extend(Evaluator(iterator=val_iter, target=model, device=mpi.gpu))  # evaluate validation dataset
trainer.extend(ext.LogReport())
trainer.extend(ext.PlotReport(['main/tot_loss', 'validation/main/tot_loss'], 'epoch', file_name='learning.png', marker='o', postprocess=set_logscale))
trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/loss', 'main/d_loss', 'main/tot_loss',
                                'validation/main/loss', 'validation/main/d_loss', 'validation/main/tot_loss']))
trainer.extend(ext.ProgressBar(update_interval=10))
trainer.extend(scatterplot(model, val, name), trigger=chainer.training.triggers.MinValueTrigger('validation/main/tot_loss', (10, 'epoch')))

trainer.run()
