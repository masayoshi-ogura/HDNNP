# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import numpy as np


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        optimizer = self.get_optimizer('main')
        model = optimizer.target
        batch = self.converter(self.get_iterator('main').next(), self.device)

        model.cleargrads()

        _, _, loss = model(*batch, train=True)
        loss.backward()

        optimizer.update()


class HDUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        super(HDUpdater, self).__init__(*args, **kwargs)

    def convert(self, batch):
        result = []
        # G
        result.append([Variable(np.concatenate([g[0][None, i, :] for g in batch], axis=0))
                       for i in xrange(len(batch[0][0]))])
        # dG
        # result.append([[[Variable(f[i])
        # result.append([[[f[i]
        #                  for f in g[1]]
        #                 for g in batch]
        #                for i in xrange(len(batch[0][0]))])
        result.append(np.array([[[f[i] for f in g[1]] for g in batch] for i in xrange(len(batch[0][0]))]))
        print result[1].shape
        # E
        result.append(np.concatenate([g[2][None, :] for g in batch]))
        # F
        result.append(np.concatenate([g[3][None, :, :] for g in batch]))
        # neighbour
        result.append(np.concatenate([g[4][None, :, :, :] for g in batch]))
        return tuple(result)

    def update_core(self):
        master_opt = self.get_optimizer('master')
        main_opt = self.get_optimizer('main')
        masters = master_opt.target
        hdnnp = main_opt.target
        batch = self.get_iterator('main').next()
        # batch ... sample x 各データ
        # batch[0][0] ... G (atom, feature)
        # batch[0][1] ... dG [feature, (atom, neighbour, 3)]
        # batch[0][2] ... E (1)
        # batch[0][3] ... F (atom, 3)
        # batch[0][4] ... neighbour (feature, atom, atom, [neighbour])

        # batch = self.converter(hoge, self.device)
        # batch ... 各データ x sample
        # batch[0][0] ... G (atom, feature)
        # batch[1][0] ... dG (feature, atom, neighbour, 3) !!! numpy.asarrayで変換されてしまう！ !!!
        # batch[2][0] ... E (1)
        # batch[3][0] ... F (atom, 3)
        # batch[4][0] ... neighbour (feature, atom, atom, neighbour) !!! 同上 !!!

        batch = self.convert(batch)
        # batch[0] ... G [atom, Variable(sample, feature)]
        # batch[1] ... dG [atom, [sample, [feature, Variable(neighbour, 3)]]]
        # batch[2] ... E (sample)
        # batch[3] ... F (sample, atom, 3)
        # batch[4] ... neighbour (sample, feature, atom, atom, [neighbour])

        masters.cleargrads()
        hdnnp.cleargrads()

        _, _, loss = hdnnp(*batch, train=True)
        loss.backward()

        hdnnp.reduce_grad_to(masters)
        master_opt.update()
        hdnnp.sync_param_with(masters)
