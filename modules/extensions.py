# -*- coding: utf -*-

from config import visual

from os import path
from copy import copy
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import chainer
from chainer import reporter as reporter_module
from chainer.training.extensions import evaluator


class Evaluator(evaluator.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                if isinstance(in_arrays, tuple):
                    eval_func(*in_arrays)
                elif isinstance(in_arrays, dict):
                    eval_func(**in_arrays)
                else:
                    eval_func(in_arrays)

            summary.add(observation)

        return summary.compute_mean()


def scatterplot(model, dataset, config):
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
            fig.savefig(path.join(trainer.out, '{}.png'.format(title)))

        G, dG, E_true, F_true = chainer.dataset.concat_examples(dataset)
        E_pred, F_pred, _ = model(G, dG, E_true, F_true)
        #E_pred = E_pred[:,[0]]
        #F_pred = F_pred.diagonal(0,1,2)

        artist(E_pred.data, E_true, '{}_Energy'.format(config), 'eV')
        artist(F_pred.data, F_true, '{}_Force'.format(config), 'eV/$\AA$')
        plt.close('all')
    return make_image


def set_logscale(f, a, summary):
    a.set_yscale('log')
