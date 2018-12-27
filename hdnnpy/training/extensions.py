# coding: utf-8

from copy import copy

import chainer
from chainer import (DictSummary, report_scope)
from chainer.training import Extension
from chainer.training.extensions import evaluator
import matplotlib.pyplot as plt
import numpy as np


class Evaluator(evaluator.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func

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
            with report_scope(observation):
                with chainer.no_backprop_mode():
                    in_arrays = self.converter(batch, self.device)
                    half = len(in_arrays) // 2
                    inputs = in_arrays[:half]
                    labels = in_arrays[half:]
                    eval_func(inputs, labels)

            summary.add(observation)

        return summary.compute_mean()


class ScatterPlot(Extension):
    def __init__(self, dataset, model, order, comm):
        self._model = model
        self._order = order
        self._comm = comm.mpi_comm

        self._properties = []
        self._units = []
        self._inputs = []
        self._labels = []
        self._predictions = []
        self._init_labels(dataset)

    def __call__(self, trainer):
        with chainer.using_config('train', False), \
             chainer.using_config('enable_backprop', False):
            predictions = self._model.predict(self._inputs)

        for i in range(self._order + 1):
            pred_send = predictions[i].data
            if self._comm.Get_rank() == 0:
                self._comm.Gatherv(pred_send, self._predictions[i], root=0)
                self._plot(trainer, self._predictions[i], self._labels[i],
                           self._properties[i], self._units[i])
            else:
                self._comm.Gatherv(pred_send, None, root=0)

        plt.close('all')

    def _init_labels(self, dataset):
        self._properties = dataset.property.properties
        self._units = dataset.property.units
        batch = chainer.dataset.concat_examples(dataset)
        half = len(batch) // 2
        self._inputs = batch[:half]
        labels = batch[half:]
        self._count = np.array(self._comm.gather(len(labels[0]), root=0))

        for i in range(self._order + 1):
            label_send = labels[i]
            if self._comm.Get_rank() == 0:
                total_size = sum(self._count)
                prediction = np.empty((total_size,) + label_send[0].shape,
                                      dtype=np.float32)
                self._predictions.append(prediction)

                label = np.empty((total_size,) + label_send[0].shape,
                                 dtype=np.float32)
                label_recv = (label, self._count * label_send[0].size)
                self._comm.Gatherv(label_send, label_recv, root=0)
                self._labels.append(label)
            else:
                self._comm.Gatherv(label_send, None, root=0)

    @staticmethod
    def _plot(trainer, prediction, label, property_, unit):
        fig = plt.figure(figsize=(10, 10))
        min_ = np.min(label)
        max_ = np.max(label)
        plt.scatter(prediction, label, c='blue'),
        plt.xlabel(f'Prediction ({unit})'),
        plt.ylabel(f'Label ({unit})'),
        plt.xlim(min_, max_),
        plt.ylim(min_, max_),
        plt.text(0.5, 0.9,
                 f'{property_} @ epoch={trainer.updater.epoch}',
                 ha='center', transform=plt.gcf().transFigure)
        fig.savefig(trainer.out/f'{property_}.png')


def set_log_scale(_, a, __):
    a.set_yscale('log')
