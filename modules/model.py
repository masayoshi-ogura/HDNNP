# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

from . import settings as stg


def loss_func(mixing_beta, y_pred, y_true, dy_pred, dy_true, obs):
    y_loss = F.mean_squared_error(y_pred, y_true)
    dy_loss = F.mean_squared_error(dy_pred, dy_true)
    loss = (1. - mixing_beta) * y_loss + mixing_beta * dy_loss
    RMSE = F.sqrt(y_loss)
    d_RMSE = F.sqrt(dy_loss)
    tot_RMSE = (1. - mixing_beta) * RMSE + mixing_beta * d_RMSE
    chainer.report({'RMSE': RMSE, 'd_RMSE': d_RMSE, 'tot_RMSE': tot_RMSE}, obs)
    return loss


class SingleNNP(chainer.Chain):
    def __init__(self, element):
        super(SingleNNP, self).__init__()
        nodes = [None] + [h['node'] for h in stg.model.layer]
        self._mixing_beta = stg.model.mixing_beta
        self._nlink = len(stg.model.layer)
        with self.init_scope():
            w = chainer.initializers.HeNormal()
            for i in range(self._nlink):
                setattr(self, 'f{}'.format(i), eval('F.{}'.format(stg.model.layer[i]['activation'])))
                setattr(self, 'l{}'.format(i), L.Linear(nodes[i], nodes[i + 1], initialW=w))
        self.add_persistent('element', element)

    def __call__(self, x, dx, y_true, dy_true, train=False):
        x = Variable(x)
        dx = Variable(dx)
        y_pred = self.predict_y(x)
        dy_pred = self.predict_dy(dx, y_pred, x, train)
        return y_pred, dy_pred, loss_func(self._mixing_beta, y_pred, y_true, dy_pred, dy_true, self)

    def __len__(self):
        return self._nlink

    def predict_y(self, x):
        h = x
        for i in range(self._nlink):
            h = eval('self.f{}(self.l{}(h))'.format(i, i))
        y = h
        return y

    @staticmethod
    def predict_dy(dx, y, x, train):
        return F.matmul(dx, chainer.grad([y], [x], enable_double_backprop=train)[0])


class HDNNP(chainer.ChainList):
    def __init__(self, composition):
        super(HDNNP, self).__init__(*[SingleNNP(element) for element in composition['atom']])
        self._mixing_beta = stg.model.mixing_beta

    def __call__(self, xs, dxs, y_true, dy_true, train=False):
        xs, dxs = self._preprocess(xs, dxs)
        y_pred = self.predict_y(xs)
        dy_pred = self.predict_dy(dxs, y_pred, xs, train)
        y_pred = sum(y_pred)
        return y_pred, dy_pred, loss_func(self._mixing_beta, y_pred, y_true, dy_pred, dy_true, self)

    def predict(self, xs, dxs):
        """
        INPUT
        xs:  ndarray (nsample, natom, nfeature)
        dxs: ndarray (nsample, natom, nfeature, natom, 3)
        OUTPUT
        y_pred:  Variable (nsample, 1)
        dy_pred: Variable (nsample, natom, 3)
        """
        xs, dxs = self._preprocess(xs, dxs)
        y_pred = self.predict_y(xs)
        dy_pred = self.predict_dy(dxs, y_pred, xs, False)
        y_pred = sum(y_pred)
        return y_pred, dy_pred

    def predict_y(self, xs):
        return [nnp.predict_y(x) for nnp, x in zip(self, xs)]

    @staticmethod
    def predict_dy(dxs, y, xs, train):
        """
        INPUT
        dxs: list of Variable [natom, (nsample, nfeature, natom, 3)]
        y:   list of Variable [natom, (nsample, 1)]
        xs:  list of Variable [natom, (nsample, nfeature)]
        train: boolean
        OUTPUT
        forces: Variable (nsample, natom, 3)

        natom, which is length of the list, is the atom energy or energy change of which will be computed
        natom, which is shape[2] of ndarray of y, is the atom forces you want to compute acting on.
        """
        shape = dxs[0].shape
        natom = shape[2]
        dys = chainer.grad(y, xs, enable_double_backprop=train)
        forces = - sum([F.sum(dx * F.repeat(dy, 3 * natom).reshape(shape), axis=1) for dx, dy in zip(dxs, dys)])
        return forces

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

    @staticmethod
    def _preprocess(xs, dxs):
        """
        convert computed Symmetry Functions from ndarray to list of chainer.Variable
        """
        xs = [Variable(x) for x in xs.transpose(1, 0, 2)]
        dxs = [Variable(dx) for dx in dxs.transpose(1, 0, 2, 3, 4)]
        return xs, dxs
