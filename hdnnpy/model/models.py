# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable


class HighDimensionalNNP(chainer.ChainList):
    def __init__(self, elemental_composition, layers, order, **kwargs):
        assert 0 <= order <= 1
        super().__init__(
            *[SubNNP(element, layers) for element in elemental_composition])
        self._set_loss_function(order, **kwargs)
        self._order = order

    def __call__(self, inputs, labels, train=False):
        predictions = self.predict(inputs, train)
        loss = self._loss_function(predictions, labels)
        return loss

    def predict(self, inputs, train=False):
        if self._order == 0:
            xs, = inputs
            xs = [Variable(x) for x in xs.transpose(1, 0, 2)]
            y_pred = sum(self._predict_y(xs))
            return [y_pred]

        elif self._order == 1:
            xs, dxs = inputs
            xs = [Variable(x) for x in xs.transpose(1, 0, 2)]
            dxs = [Variable(dx) for dx in dxs.transpose(1, 0, 2, 3, 4)]
            y_pred = self._predict_y(xs)
            dy_pred = self._predict_dy(dxs, y_pred, xs, train)
            y_pred = sum(y_pred)
            return [y_pred, dy_pred]

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
        """INPUT
        dxs: list of Variable [natom, (nsample, nfeature, natom, 3)]
        y:   list of Variable [natom, (nsample, 1)]
        xs:  list of Variable [natom, (nsample, nfeature)]
        train: boolean
        OUTPUT
        forces: Variable (nsample, 1, natom, 3)

        natom, which is length of the list, is the atom energy
         or energy change of which will be computed
        natom, which is shape[2] of ndarray of y, is the atom
         that forces you want to compute acting on.
        """
        n_atom = dxs[0].shape[2]
        dy_shape = y[0].shape + (n_atom, 3)
        dys = chainer.grad(y, xs, enable_double_backprop=train)
        forces = - sum([F.sum(dx * F.repeat(dy, n_atom*3).reshape(dx.shape),
                              axis=1)
                        for dx, dy in zip(dxs, dys)]).reshape(dy_shape)
        return forces

    def _set_loss_function(self, order, **kwargs):
        if order == 0:
            def zeroth_order(predictions, labels):
                pred0, = predictions
                true0, = labels
                loss0 = F.mean_squared_error(pred0, true0)
                RMSE0 = F.sqrt(loss0) / len(self)
                chainer.report({
                    'RMSE0': RMSE0,
                    'total_RMSE': RMSE0,
                    }, observer=self)
                return loss0

            self._loss_function = zeroth_order

        elif order == 1:
            mixing_beta = kwargs['mixing_beta']

            def first_order(predictions, labels):
                pred0, pred1 = predictions
                true0, true1 = labels
                loss0 = F.mean_squared_error(pred0, true0)
                loss1 = F.mean_squared_error(pred1, true1)
                total_loss = ((1.0 - mixing_beta) * loss0
                              + mixing_beta * loss1)

                RMSE0 = F.sqrt(loss0) / len(self)
                RMSE1 = F.sqrt(loss1)
                total_RMSE = ((1.0 - mixing_beta) * RMSE0
                              + mixing_beta * RMSE1)
                chainer.report({
                    'RMSE0': RMSE0,
                    'RMSE1': RMSE1,
                    'total_RMSE': total_RMSE,
                    }, observer=self)

                return total_loss

            self._loss_function = first_order


class MasterNNP(chainer.ChainList):
    def __init__(self, elements, layers):
        super().__init__(*[SubNNP(element, layers) for element in elements])


class SubNNP(chainer.Chain):
    def __init__(self, element, layers):
        super().__init__()
        self.add_persistent('element', element)
        self._n_layer = len(layers)
        nodes = [None] + [layer[0] for layer in layers]
        activations = [layer[1] for layer in layers]
        with self.init_scope():
            w = chainer.initializers.HeNormal()
            for i, (insize, outsize, activation) in enumerate(zip(
                    nodes[:-1], nodes[1:], activations)):
                setattr(self, f'activation_function{i}',
                        eval(f'F.{activation}'))
                setattr(self, f'fc_layer{i}',
                        L.Linear(insize, outsize, initialW=w))

    def __len__(self):
        return self._n_layer

    def __call__(self, x):
        h = x
        for i in range(self._n_layer):
            h = eval(f'self.activation_function{i}(self.fc_layer{i}(h))')
        y = h
        return y
