# -*- coding: utf-8 -*-

from config import hp

import chainer
import chainer.functions as F
from chainer import Variable


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        optimizer = self.get_optimizer('main')
        model = optimizer.target
        batch = self.converter(self.get_iterator('main').next(), self.device)

        _, _, loss = model(*batch, train=True)
        loss.backward()

        optimizer.update()


class HDUpdater(chainer.training.StandardUpdater):
    def __init__(self, hdnnp, *args, **kwargs):
        self.hdnnp = hdnnp
        super(HDUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        optimizer = self.get_optimizer('main')
        masters = optimizer.target
        batch = self.converter(self.get_iterator('main').next(), self.device)

        _, _, loss = self.hdnnp(*batch, train=True)
        loss.backward()

        self.hdnnp.reduce_grad_to(masters)
        optimizer.update()
        self.hdnnp.sync_param_with(masters)
