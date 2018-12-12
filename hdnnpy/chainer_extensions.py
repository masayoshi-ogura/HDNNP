# -*- coding: utf-8 -*-

__all__ = [
    'Evaluator',
    'scatter_plot',
    'set_log_scale',
    ]

from copy import copy

import chainer
from chainer import reporter as reporter_module
from chainer.training.extensions import evaluator
import matplotlib.pyplot as plt
import numpy as np


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
            # backprop_mode is needed for HDNNP
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


def scatter_plot(model, dataset):
    @chainer.training.make_extension()
    def make_image(trainer):
        def artist(pred, true, title, unit):
            fig = plt.figure(figsize=(10, 10))
            min_ = np.min(true)
            max_ = np.max(true)
            plt.scatter(pred, true, c='blue'),
            plt.xlabel(f'NNP ({unit})'),
            plt.ylabel(f'DFT ({unit})'),
            plt.xlim(min_, max_),
            plt.ylim(min_, max_),
            plt.text(0.5, 0.9,
                     f'{title} @ epoch={trainer.updater.epoch}',
                     ha='center', transform=plt.gcf().transFigure)
            fig.savefig(trainer.out/f'{title}.png')

        G, dG, E_true, F_true = chainer.dataset.concat_examples(dataset)
        E_pred, F_pred = model.predict(G, dG)

        artist(E_pred.data, E_true, 'Energy', 'eV')
        artist(F_pred.data, F_true, 'Force', 'eV/$\AA$')
        plt.close('all')

    return make_image


def set_log_scale(f, a, summary):
    a.set_yscale('log')
