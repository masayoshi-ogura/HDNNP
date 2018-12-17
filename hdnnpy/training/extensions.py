# coding: utf-8

from copy import copy

import chainer
from chainer import (DictSummary, report_scope)
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

        summary = DictSummary()

        for batch in it:
            observation = {}
            # backprop_mode is needed for HDNNP
            with report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                half = len(in_arrays) // 2
                inputs, labels = in_arrays[:half], in_arrays[half:]
                eval_func(inputs, labels, train=False)

            summary.add(observation)

        return summary.compute_mean()


def scatter_plot(model, dataset, order):
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

        batch = chainer.dataset.concat_examples(dataset)
        half = len(batch) // 2
        inputs, labels = batch[:half], batch[half:]
        predictions = model.predict(inputs)

        if order >= 0:
            E_pred = predictions[0].data * 1000 / len(model)
            E_true = labels[0] * 1000 / len(model)
            artist(E_pred, E_true, 'Energy', 'meV/atom')
        if order >= 1:
            F_pred = predictions[1].data * 1000
            F_true = labels[1] * 1000
            artist(F_pred, F_true, 'Force', 'meV/$\AA$')
        plt.close('all')

    return make_image


def set_log_scale(_, a, __):
    a.set_yscale('log')