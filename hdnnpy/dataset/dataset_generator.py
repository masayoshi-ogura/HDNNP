# coding: utf-8

"""Deal out datasets as needed."""

from sklearn.model_selection import KFold

from hdnnpy.dataset.hdnnp_dataset import HDNNPDataset


class DatasetGenerator(object):
    """Deal out datasets as needed."""
    def __init__(self, *datasets):
        """
        Args:
            *datasets (HDNNPDataset): What you want to unite.
        """
        if not datasets:
            raise ValueError('No dataset are given')
        for dataset in datasets:
            assert isinstance(dataset, HDNNPDataset)
        self._datasets = list(datasets)

    def all(self):
        """Pass all datasets an instance have.

        Returns:
            list [HDNNPDataset]: All stored datasets.
        """
        return self._datasets

    def foreach(self):
        """Pass all datasets an instance have one by one.

        Returns:
            Iterator [HDNNPDataset]: a stored dataset object.
        """
        for dataset in self._datasets:
            yield dataset

    def holdout(self, ratio):
        """Split each dataset at a certain rate and pass it

        Args:
            ratio (float):
                Specify the rate you want to use as training data.
                Remains are test data.

        Returns:
            list [tuple [HDNNPDataset, HDNNPDataset]]:
            All stored dataset split by specified ratio into training
            and test data.
        """
        split = []
        for dataset in self._datasets:
            s = int(dataset.partial_size * ratio)
            train = dataset.take(slice(None, s, None))
            test = dataset.take(slice(s, None, None))
            split.append((train, test))
        return split

    def kfold(self, kfold):
        """Split each dataset almost equally and pass it for cross
        validation.

        Args:
            kfold (int): Number of folds to split dataset.

        Returns:
            Iterator [list [tuple [HDNNPDataset, HDNNPDataset]]]:
            All stored dataset split into training and test data.
            It iterates k times while changing parts used for test data.
        """
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
