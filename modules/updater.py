# -*- coding: utf-8 -*-

from config import hp

import chainer
import chainer.functions as F
from chainer import Variable


def loss(y_pred, y_true, dy_pred, dy_true):
    y_loss = (1. - hp.mixing_beta) * F.mean_squared_error(y_pred, y_true)
    dy_loss = hp.mixing_beta * F.mean_squared_error(dy_pred, dy_true)
    loss = y_loss + dy_loss
    chainer.report({'loss': y_loss, 'd_loss': dy_loss, 'total_loss': loss})
    return loss


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        optimizer = self.get_optimizer('main')
        model = optimizer.target
        batch = self.get_iterator('main').next()
        x, dx, y_true, dy_true = self.converter(batch, self.device)
        x = Variable(x)
        dx = Variable(dx)

        y_pred = model(x)
        dy_pred = dx * chainer.grad([y_pred], [x], retain_grad=True, enable_double_backprop=True)[0]
        optimizer.update(loss, y_pred, y_true, dy_pred, dy_true)


class HDUpdater(chainer.training.StandardUpdater):
    def __init__(self, hdnnp, *args, **kwargs):
        self.hdnnp = hdnnp
        super(HDUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        optimizers = [opt for opt in self.get_all_optimizers().values()]
        masters = [opt.target for opt in optimizers]
        batch = self.get_iterator('main').next()
        G, dG, E_true, F_true = self.converter(batch, self.device)
        G = [Variable(g) for g in G.transpose(1, 0, 2)]
        dG = [[Variable(dg2) for dg2 in dg1] for dg1 in dG.transpose(2, 1, 0, 3)]

        for nnp in self.hdnnp:
            nnp.cleargrads()
        for master in masters:
            master.cleargrads()

        E = self.hdnnp(G)
        dE_dG = chainer.grad(E, G, retain_grad=True, enable_double_backprop=True)
        F_pred = - F.concat([F.sum(sum([dg * de_dg for dg, de_dg in zip(dg_dx, dE_dG)]), axis=1, keepdims=True) for dg_dx in dG], axis=1)
        E_pred = sum(E)
        loss(E_pred, E_true, F_pred, F_true).backward()

        for optimizer, master in zip(optimizers, masters):
            for nnp in self.hdnnp.get_by_element(master.element):
                master.addgrads(nnp)
            optimizer.update()
            for nnp in self.hdnnp.get_by_element(master.element):
                nnp.copyparams(master)
