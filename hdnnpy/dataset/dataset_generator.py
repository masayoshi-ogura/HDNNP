# -*- coding: utf-8 -*-

from sklearn.model_selection import KFold


class DatasetGenerator(object):
    def __init__(self, datasets):
        self._datasets = datasets

    def all(self):
        return self._datasets

    def foreach(self):
        for dataset in self._datasets:
            yield dataset

    def holdout(self, ratio):
        split = []
        while self._datasets:
            dataset = self._datasets.pop(0)
            s = int(len(dataset) * ratio)
            train = dataset.take(slice(None, s, None))
            test = dataset.take(slice(s, None, None))
            split.append((train, test))
        return split

    def kfold(self, kfold):
        kf = KFold(n_splits=kfold)
        kfold_indices = [kf.split(range(len(dataset))) for dataset in self._datasets]

        for indices in zip(*kfold_indices):
            split = []
            for dataset, (train_idx, test_idx) in zip(self._datasets, indices):
                train = dataset.take(train_idx)
                test = dataset.take(test_idx)
                split.append((train, test))
            yield split
