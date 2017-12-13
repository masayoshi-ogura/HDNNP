# -*- coding: utf-8 -*-

from config import hp

import chainer
import chainer.links as L
import chainer.functions as F


class SingleNNP(chainer.Chain):
    def __init__(self, element):
        super(SingleNNP, self).__init__()
        nodes = [None] + [h['node'] for h in hp.hidden_layers]
        self.nlink = len(hp.hidden_layers)
        with self.init_scope():
            for i in range(self.nlink):
                setattr(self, 'f{}'.format(i), eval('F.{}'.format(hp.hidden_layers[i]['activation'])))
                setattr(self, 'l{}'.format(i), L.Linear(nodes[i], nodes[i+1]))
        self.add_persistent('element', element)

    def __call__(self, x):
        h = x
        for i in range(self.nlink):
            h = eval('self.f{}(self.l{}(h))'.format(i, i))
        return h


class HDNNP(chainer.ChainList):
    def __init__(self, composition):
        super(HDNNP, self).__init__(*[SingleNNP(element) for element in composition['element']])

    def __call__(self, xs):
        return [nnp(x) for nnp, x in zip(self, xs)]

    def get_by_element(self, element):
        return [nnp for nnp in self if nnp.element == element]

    def sync_master(self, masters):
        for element, master in masters.items():
            for nnp in self.get_by_element(element):
                nnp.copyparams(master)
