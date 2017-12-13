# -*- coding: utf -*-

from config import visual

from os import path
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
from chainer import Variable


def scatterplot(model, dataset, out_dir):
    G, dG, E_true, F_true = dataset._datasets
    G = [Variable(g) for g in G.transpose(1, 0, 2)]
    dG = [[Variable(dg2) for dg2 in dg1] for dg1 in dG.transpose(2, 1, 0, 3)]

    @chainer.training.make_extension()
    def make_image(trainer):
        def artist(pred, true, title, unit):
            fig = plt.figure()
            min = np.min(true)
            max = np.max(true)
            plt.scatter(pred, true, c='blue'),
            plt.xlabel('NNP ({})'.format(unit)),
            plt.ylabel('DFT ({})'.format(unit)),
            plt.xlim(min, max),
            plt.ylim(min, max),
            plt.text(0.5, 0.9,
                     '{} @epoch={}'.format(title, trainer.updater.epoch),
                     fontsize=visual.fontsize, ha='center', transform=plt.gcf().transFigure)
            fig.savefig(path.join(out_dir, '{}.png'.format(title)))

        E = model(G)
        dE_dG = chainer.grad(E, G)
        F_pred = - F.concat([F.sum(sum([dg * de_dg for dg, de_dg in zip(dg_dx, dE_dG)]), axis=1, keepdims=True) for dg_dx in dG], axis=1)
        E_pred = sum(E)

        artist(E_pred.data, E_true, 'Energy', 'eV')
        artist(F_pred.data, F_true, 'Force', 'eV/$\AA$')
        plt.close('all')
    return make_image


def set_logscale(f, a, summary):
    a.set_yscale('log')
