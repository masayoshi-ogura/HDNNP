# -*- coding: utf-8 -*-

__all__ = [
    'HDNNP',
    'SingleNNP',
    'loss_func',
    ]

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

from hdnnpy.settings import stg


class SingleNNP(chainer.Chain):
    def __init__(self, element):
        super(SingleNNP, self).__init__()
        nodes = [None] + [h['node'] for h in stg.model.layer]
        self._nlink = len(stg.model.layer)
        with self.init_scope():
            w = chainer.initializers.HeNormal()
            for i in range(self._nlink):
                setattr(self, f'f{i}',
                        eval(f'F.{stg.model.layer[i]["activation"]}'))
                setattr(self, f'l{i}',
                        L.Linear(nodes[i], nodes[i + 1], initialW=w))
        self.add_persistent('element', element)

    def __len__(self):
        return self._nlink

    def __call__(self, x):
        h = x
        for i in range(self._nlink):
            h = eval(f'self.f{i}(self.l{i}(h))')
        y = h
        return y


class HDNNP(chainer.ChainList):
    def __init__(self, elemental_composition, loss_func):
        super(HDNNP, self).__init__(
            *[SingleNNP(element) for element in elemental_composition])
        self.loss_func = loss_func

    def __call__(self, xs, dxs, y_true, dy_true, train=False):
        y_pred, dy_pred = self.predict(xs, dxs, train)
        y_loss, dy_loss, loss = self.loss_func(y_pred, dy_pred,
                                               y_true, dy_true)
        RMSE = F.sqrt(y_loss)
        d_RMSE = F.sqrt(dy_loss)
        tot_RMSE = ((1.0-stg.model.mixing_beta) * RMSE
                    + stg.model.mixing_beta * d_RMSE)
        chainer.report({'RMSE': RMSE, 'd_RMSE': d_RMSE, 'tot_RMSE': tot_RMSE},
                       self)
        return loss

    def predict(self, xs, dxs, train=False):
        xs, dxs = self._preprocess(xs, dxs)
        y_pred = self._predict_y(xs)
        dy_pred = self._predict_dy(dxs, y_pred, xs, train)
        y_pred = sum(y_pred)
        return y_pred, dy_pred

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

    def _predict_y(self, xs):
        return [nnp(x) for nnp, x in zip(self, xs)]

    @staticmethod
    def _predict_dy(dxs, y, xs, train):
        """
        INPUT
        dxs: list of Variable [natom, (nsample, nfeature, natom, 3)]
        y:   list of Variable [natom, (nsample, 1)]
        xs:  list of Variable [natom, (nsample, nfeature)]
        train: boolean
        OUTPUT
        forces: Variable (nsample, 1, natom, 3)

        natom, which is length of the list, is the atom energy or energy change of which will be computed
        natom, which is shape[2] of ndarray of y, is the atom forces you want to compute acting on.
        """
        n_atom = dxs[0].shape[2]
        dy_shape = y[0].shape + (n_atom, 3)
        dys = chainer.grad(y, xs, enable_double_backprop=train)
        forces = - sum([F.sum(dx * F.repeat(dy, n_atom*3).reshape(dx.shape),
                              axis=1)
                        for dx, dy in zip(dxs, dys)]).reshape(dy_shape)
        return forces

    @staticmethod
    def _preprocess(xs, dxs):
        """
        convert computed Symmetry Functions from ndarray to list of chainer.Variable
        """
        xs = [Variable(x) for x in xs.transpose(1, 0, 2)]
        dxs = [Variable(dx) for dx in dxs.transpose(1, 0, 2, 3, 4)]
        return xs, dxs


def loss_func(y_pred, dy_pred, y_true, dy_true):
    y_loss = F.mean_squared_error(y_pred, y_true)
    dy_loss = F.mean_squared_error(dy_pred, dy_true)
    loss = ((1.0-stg.model.mixing_beta) * y_loss
            + stg.model.mixing_beta * dy_loss)
    return y_loss, dy_loss, loss