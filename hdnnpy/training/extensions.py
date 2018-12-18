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
            # backprop_mode is needed for HDNNP
            with report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                half = len(in_arrays) // 2
                inputs = in_arrays[:half]
                labels = in_arrays[half:]
                eval_func(inputs, labels, train=False)

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
        with chainer.using_config('train', False):
            predictions = self._model.predict(self._inputs, train=False)

        if self._order >= 0:
            pred_send = predictions[0].data * 1000 / len(self._model)
            if self._comm.Get_rank() == 0:
                self._comm.Gatherv(pred_send, self._predictions[0], root=0)
                self._plot(trainer, self._predictions[0], self._labels[0],
                           self._properties[0], self._units[0])
            else:
                self._comm.Gatherv(pred_send, None, root=0)

        if self._order >= 1:
            pred_send = predictions[1].data * 1000
            if self._comm.Get_rank() == 0:
                self._comm.Gatherv(pred_send, self._predictions[1], root=0)
                self._plot(trainer, self._predictions[1], self._labels[1],
                           self._properties[1], self._units[1])
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
        size = sum(self._comm.gather(len(labels[0]), root=0))

        if self._order >= 0:
            label_send = labels[0] * 1000 / len(self._model)
            if self._comm.Get_rank() == 0:
                prediction = np.empty((size,) + label_send.shape[1:],
                                      dtype=np.float32)
                label = np.empty((size,) + label_send.shape[1:],
                                 dtype=np.float32)
                self._comm.Gatherv(label_send, label, root=0)
                self._predictions.append(prediction)
                self._labels.append(label)
            else:
                self._comm.Gatherv(label_send, None, root=0)

        if self._order >= 1:
            label_send = labels[1] * 1000
            if self._comm.Get_rank() == 0:
                prediction = np.empty((size,) + label_send.shape[1:],
                                      dtype=np.float32)
                label = np.empty((size,) + label_send.shape[1:],
                                 dtype=np.float32)
                self._comm.Gatherv(label_send, label, root=0)
                self._predictions.append(prediction)
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
