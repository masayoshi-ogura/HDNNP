# -*- coding: utf-8 -*-

from config import hp

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable


def loss_func(y_pred, y_true, dy_pred, dy_true, obs):
    y_loss = (1. - hp.mixing_beta) * F.mean_squared_error(y_pred, y_true)
    dy_loss = hp.mixing_beta * F.mean_squared_error(dy_pred, dy_true)
    loss = y_loss + dy_loss
    chainer.report({'RMSE': np.sqrt(y_loss.data), 'd_RMSE': np.sqrt(dy_loss.data), 'tot_RMSE': np.sqrt(loss.data)}, obs)
    return loss


class SingleNNP(chainer.Chain):
    def __init__(self, element):
        super(SingleNNP, self).__init__()
        nodes = [None] + [h['node'] for h in hp.hidden_layers]
        self.nlink = len(hp.hidden_layers)
        with self.init_scope():
            w = chainer.initializers.HeNormal()
            for i in range(self.nlink):
                setattr(self, 'f{}'.format(i), eval('F.{}'.format(hp.hidden_layers[i]['activation'])))
                setattr(self, 'l{}'.format(i), L.Linear(nodes[i], nodes[i+1], initialW=w))
        self.add_persistent('element', element)

    def __call__(self, x, dx, y_true, dy_true, train=False):
        x = Variable(x)
        dx = Variable(dx)
        y_pred = self.predict_y(x)
        dy_pred = self.predict_dy(dx, y_pred, x, train)
        return y_pred, dy_pred, loss_func(y_pred, y_true, dy_pred, dy_true, self)

    def predict_y(self, x):
        h = x
        for i in range(self.nlink):
            h = eval('self.f{}(self.l{}(h))'.format(i, i))
        y = h
        return y

    def predict_dy(self, dx, y, x, train):
        return F.batch_matmul(dx, chainer.grad([y], [x], retain_grad=train, enable_double_backprop=train)[0])


class HDNNP(chainer.ChainList):
    def __init__(self, composition):
        super(HDNNP, self).__init__(*[SingleNNP(element) for element in composition['element']])

    def __call__(self, xs, dxs, y_true, dy_true, train=False):
        xs = [Variable(x) for x in xs.transpose(1, 0, 2)]
        dxs = [Variable(dx) for dx in dxs.transpose(1, 0, 2, 3)]
        # dxs = [[Variable(dx2) for dx2 in dx1] for dx1 in dxs.transpose(2, 1, 0, 3)]
        y_pred = self.predict_y(xs)
        dy_pred = self.predict_dy(dxs, y_pred, xs, train)
        y_pred = sum(y_pred)
        return y_pred, dy_pred, loss_func(y_pred, y_true, dy_pred, dy_true, self)

    def predict_y(self, xs):
        return [nnp.predict_y(x) for nnp, x in zip(self, xs)]

    def predict_dy(self, dxs, y, xs, train):
        dy = chainer.grad(y, xs, retain_grad=train, enable_double_backprop=train)
        return - sum([F.batch_matmul(dxi, dyi) for dxi, dyi in zip(dxs, dy)])
        # return - F.concat([F.sum(sum([dxi * dyi for dxi, dyi in zip(dx, dy)]), axis=1, keepdims=True) for dx in dxs], axis=1)

    def get_by_element(self, element):
        return [nnp for nnp in self if nnp.element == element]

    def reduce_grad_to(self, masters):
        for master in masters.children():
            for nnp in self.get_by_element(master.element):
                master.addgrads(nnp)

    def sync_param_with(self, masters):
        for master in masters.children():
            for nnp in self.get_by_element(master.element):
                nnp.copyparams(master)
