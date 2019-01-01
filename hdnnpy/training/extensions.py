# coding: utf-8

"""Custom chainer training extensions."""

import chainer
from chainer.training import Extension
import matplotlib.pyplot as plt
import numpy as np


class ScatterPlot(Extension):
    """Trainer extension to output predictions/labels scatter plots."""
    def __init__(self, dataset, model, comm):
        """
        Args:
            dataset (HDNNPDataset):
                Test dataset to plot a scatter plot. It has to have both
                input dataset and label dataset.
            model (HighDimensionalNNP): HDNNP model to evaluate.
            comm (~chainermn.CommunicatorBase):
                ChainerMN communicator instance.
        """
        self._model = model
        self._comm = comm.mpi_comm

        self._properties = []
        self._units = []
        self._inputs = []
        self._labels = []
        self._predictions = []
        self._init_labels(dataset)

    def __call__(self, trainer):
        """Execute scatter plot extension.

        | Perform prediction with the parameters of the model when this
          extension was executed, using the data set at initialization.
        | Horizontal axis shows the predicted values and vertical axis
          shows the true values.
        | Plot configurations are written in :meth:`_plot`.

        Args:
            trainer (~chainer.training.Trainer):
                Trainer object that invokes this extension.
        """
        with chainer.using_config('train', False), \
             chainer.using_config('enable_backprop', False):
            predictions = self._model.predict(self._inputs)

        for i in range(self._model.order + 1):
            pred_send = predictions[i].data
            if self._comm.Get_rank() == 0:
                self._comm.Gatherv(pred_send, self._predictions[i], root=0)
                self._plot(trainer, self._predictions[i], self._labels[i],
                           self._properties[i], self._units[i])
            else:
                self._comm.Gatherv(pred_send, None, root=0)

        plt.close('all')

    def _init_labels(self, dataset):
        """Gather label dataset to root process and initialize other
        instance variables."""
        self._properties = dataset.property.properties
        self._units = dataset.property.units
        batch = chainer.dataset.concat_examples(dataset)
        half = len(batch) // 2
        self._inputs = batch[:half]
        labels = batch[half:]
        self._count = np.array(self._comm.gather(len(labels[0]), root=0))

        for i in range(self._model.order + 1):
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
        """Plot and save a scatter plot."""
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
    """Change y axis scale as log scale."""
    a.set_yscale('log')
