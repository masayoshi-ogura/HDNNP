# -*- coding: utf-8 -*-

__all__ = [
    'DatasetGenerator',
    ]

from sklearn.model_selection import KFold

from hdnnpy.dataset.hdnnp_dataset import HDNNPDataset


class DatasetGenerator(object):
    def __init__(self, *datasets):
        if not datasets:
            raise ValueError('No dataset are given')
        for dataset in datasets:
            assert isinstance(dataset, HDNNPDataset)
        self._datasets = list(datasets)

    def all(self):
        return self._datasets

    def foreach(self):
        for dataset in self._datasets:
            yield dataset

    def holdout(self, ratio):
        split = []
        while self._datasets:
            dataset = self._datasets.pop(0)
            s = int(dataset.partial_size * ratio)
            train = dataset.take(slice(None, s, None))
            test = dataset.take(slice(s, None, None))
            split.append((train, test))
        return split

    def kfold(self, kfold):
        kf = KFold(n_splits=kfold)
        kfold_indices = [kf.split(range(dataset.partial_size))
                         for dataset in self._datasets]

        for indices in zip(*kfold_indices):
            split = []
            for dataset, (train_idx, test_idx) in zip(self._datasets, indices):
                train = dataset.take(train_idx)
                test = dataset.take(test_idx)
                split.append((train, test))
            yield split
