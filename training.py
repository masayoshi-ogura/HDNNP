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
import chainer.training.extensions as ext

# import own modules
from modules.data import DataGenerator
from modules.model import SingleNNP, HDNNP
from modules.updater import HDUpdater
from modules.extensions import Evaluator, scatterplot, set_logscale

datestr = datetime.now().strftime('%m%d-%H%M%S')
out_dir = path.join(file_.out_dir, datestr)
makedirs(out_dir)
copy2('config.py', path.join(out_dir, 'config.py'))

# dataset and iterator
for i, dataset in enumerate(DataGenerator()):
    train, val = chainer.datasets.split_dataset_random(dataset, int(0.8*len(dataset)))
    # for train, val in chainer.datasets.get_cross_validation_datasets_random(dataset, n_fold=hp.kfold_cv):
    train_iter = chainer.iterators.SerialIterator(train, hp.batch_size)
    val_iter = chainer.iterators.SerialIterator(val, hp.batch_size, repeat=False, shuffle=False)

    # model and optimizer
    masters = chainer.ChainList(*[SingleNNP(element) for element in dataset.composition['index']])
    master_opt = chainer.optimizers.Adam(hp.init_lr)
    master_opt.setup(masters)
    master_opt.add_hook(chainer.optimizer.Lasso(hp.l1_norm))
    master_opt.add_hook(chainer.optimizer.WeightDecay(hp.l2_norm))
    if i != 0:
        chainer.serializers.load_npz(path.join(out_dir, 'masters.npz'), masters)
        chainer.serializers.load_npz(path.join(out_dir, 'optimizer.npz'), master_opt)
    hdnnp = HDNNP(dataset.composition)
    hdnnp.sync_param_with(masters)
    main_opt = chainer.Optimizer()
    main_opt.setup(hdnnp)

    # updater and trainer
    updater = HDUpdater(iterator=train_iter, optimizer={'main': main_opt, 'master': master_opt}, device=mpi.gpu)
    trainer = chainer.training.Trainer(updater, (hp.nepoch, 'epoch'), out=out_dir)

    # extensions
    trainer.extend(ext.ExponentialShift('alpha', 1-hp.lr_decay, target=hp.final_lr, optimizer=master_opt))  # learning rate decay
    trainer.extend(Evaluator(iterator=val_iter, target=hdnnp, device=mpi.gpu))  # evaluate validation dataset
    trainer.extend(ext.LogReport())
    trainer.extend(ext.PlotReport(['main/tot_RMSE', 'validation/main/tot_RMSE'], 'epoch',
                                  file_name='learning.png', marker=None, postprocess=set_logscale))
    trainer.extend(ext.PrintReport(['epoch', 'iteration', 'main/RMSE', 'main/d_RMSE', 'main/tot_RMSE',
                                    'validation/main/RMSE', 'validation/main/d_RMSE', 'validation/main/tot_RMSE']))
    # trainer.extend(ext.ProgressBar(update_interval=10))
    trainer.extend(scatterplot(hdnnp, val, dataset.config),
                   trigger=chainer.training.triggers.MinValueTrigger('validation/main/tot_RMSE', (10, 'epoch')))
    # trainer.extend(ext.ParameterStatistics(masters))

    trainer.run()

    # serialize
    chainer.serializers.save_npz(path.join(out_dir, 'masters.npz'), masters)
    chainer.serializers.save_npz(path.join(out_dir, 'optimizer.npz'), master_opt)
