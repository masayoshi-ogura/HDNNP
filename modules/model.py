# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import numpy as np
from itertools import product


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
    def __init__(self, hp, element):
        super(SingleNNP, self).__init__()
        nodes = [None] + [h.node for h in hp.layer]
        self._mixing_beta = hp.mixing_beta
        self._nlink = len(hp.layer)
        with self.init_scope():
            # w = chainer.initializers.HeNormal()
            for i in range(self._nlink):
                setattr(self, 'f{}'.format(i), eval('F.{}'.format(hp.layer[i].activation)))
                setattr(self, 'l{}'.format(i), L.Linear(nodes[i], nodes[i+1], initialW=None))
        self.add_persistent('element', element)

    def __call__(self, x, dx, y_true, dy_true, train=False):
        x = Variable(x)
        dx = Variable(dx)
        y_pred = self.predict_y(x)
        dy_pred = self.predict_dy(dx, y_pred, x, train)
        return y_pred, dy_pred, loss_func(self._mixing_beta, y_pred, y_true, dy_pred, dy_true, self)

    def predict_y(self, x):
        h = x
        for i in range(self._nlink):
            h = eval('self.f{}(self.l{}(h))'.format(i, i))
        y = h
        return y

    def predict_dy(self, dx, y, x, train):
        return F.batch_matmul(dx, chainer.grad([y], [x], retain_grad=train, enable_double_backprop=train)[0])


class HDNNP(chainer.ChainList):
    def __init__(self, hp, composition):
        super(HDNNP, self).__init__(*[SingleNNP(hp, element) for element in composition.element])
        self._mixing_beta = hp.mixing_beta

    def __call__(self, xs, dxs, y_true, dy_true, neighbour, train=False):
        y_pred = self.predict_y(xs)
        dy_pred = self.predict_dy(dxs, y_pred, xs, neighbour, train)
        y_pred = sum(y_pred)
        return y_pred, dy_pred, loss_func(self._mixing_beta, y_pred, y_true, dy_pred, dy_true, self)

    def predict(self, xs, dxs):
        y_pred = self.predict_y(xs)
        dy_pred = self.predict_dy(dxs, y_pred, xs, False)
        y_pred = sum(y_pred)
        return y_pred, dy_pred

    def predict_y(self, xs):
        return [nnp.predict_y(x) for nnp, x in zip(self, xs)]

    def predict_dy(self, dxs, y, xs, neighbour, train):
        dys = chainer.grad(y, xs, retain_grad=train, enable_double_backprop=train)
        print type(dxs), type(dxs[0]), type(dxs[0][0]), type(dxs[0][0][0])
        print dxs
        print '-------------------------------'
        print dxs[0]
        print '-------------------------------'
        print dxs[0][0]
        print '-------------------------------'
        print dxs[0][0][0]
        nd = np.array(dxs)
        print nd.shape

        natom = len(dxs)
        batch_size = len(dxs[0])
        nfeature = len(dxs[0][0])
        result = []
        for dx, dy in zip(dxs, dys):
            for dxx, dyy in zip(dx, dy):
                for dxxx, dyyy in zip(dxx, dyy):
                    result.append(dxxx * F.tile(dyyy, dxxx.shape))
        print natom, batch_size, nfeature
        dy = Variable(np.zeros((batch_size, natom, 3), dtype=np.float32))
        for i, j, k in product(xrange(batch_size), xrange(natom), xrange(3)):
            dy[i, j, k] += sum([sum(result[l*batch_size*nfeature + i*nfeature + m][neighbour[i, m, l, j], k])
                               for l in xrange(natom) for m in xrange(nfeature)])

        # F = F.reshape(F.concat(result), (natom, batch_size, n_neighb, 3))
        print 'OK'

        # dysはセル内の原子数natomの配列になっている。つまり、対応するindexのdG_dxを掛け合わせてあげれば一つの原子のエネルギー変化がもとまる。
        # それらをneighboursの数だけsumしてあげれば、ある原子のある方向の力がわかる。
        # dys.shape = [atom, V(batch, feature)]
        # dxs.shape = [atom, [batch, [feature, V(neighbour, 3)]]]
        # F.shape = (batch, atom, 3)
        # batchはひとまず無視して、
        # F = - dE_dG * dG_dx
        #   = - SUM[neighbour](dEi_dGi * dGi_dx)
        #   = - SUM[neighbour]SUM[feature]()
        # F[0][0][0]、つまり1個目のサンプルの原子1のx方向の力は、
        # dys[0][0, :] * dxs[:][0][0, :, 0]

        # return - sum([F.batch_matmul(dxi, dyi) for dxi, dyi in zip(dxs, dy)])

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
