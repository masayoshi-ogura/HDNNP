# -*- coding: utf -*-

from config import visual

from os import path
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
from chainer import Variable

from data import AtomicStructureDataset


def scatterplot(model, dataset):
    @chainer.training.make_extension()
    def make_image1(trainer):
        def artist(pred, true, title):
            fig = plt.figure()
            min = np.min(true)
            max = np.max(true)
            plt.scatter(pred, true, c='blue'),
            plt.xlabel('prediction'),
            plt.ylabel('target'),
            plt.xlim(min, max),
            plt.ylim(min, max),
            plt.text(0.5, 0.9,
                     '{} @epoch={}'.format(title, trainer.updater.epoch),
                     fontsize=visual.fontsize, ha='center', transform=plt.gcf().transFigure)
            fig.savefig(path.join(trainer.out, '{}.png'.format(title)))

        x, dx, y_true, dy_true = dataset._datasets
        x = Variable(x)
        dx = Variable(dx)

        y_pred = model(x)
        dy_pred = chainer.grad([y_pred], [x])[0]

        artist(y_pred.data, y_true, 'original')
        artist(dy_pred.data, dy_true, 'derivative')
        plt.close('all')

    @chainer.training.make_extension()
    def make_image2(trainer):
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
            fig.savefig(path.join(trainer.out, '{}.png'.format(title)))

        G, dG, E_true, F_true = dataset._datasets
        G = [Variable(g) for g in G.transpose(1, 0, 2)]
        dG = [[Variable(dg2) for dg2 in dg1] for dg1 in dG.transpose(2, 1, 0, 3)]

        E = model(G)
        dE_dG = chainer.grad(E, G)
        F_pred = - F.concat([F.sum(sum([dg * de_dg for dg, de_dg in zip(dg_dx, dE_dG)]), axis=1, keepdims=True) for dg_dx in dG], axis=1)
        E_pred = sum(E)

        artist(E_pred.data, E_true, 'Energy', 'eV')
        artist(F_pred.data, F_true, 'Force', 'eV/$\AA$')
        plt.close('all')

    if isinstance(dataset, chainer.datasets.TupleDataset):
        return make_image1
    elif isinstance(dataset, AtomicStructureDataset):
        return make_image2


def set_logscale(f, a, summary):
    a.set_yscale('log')
