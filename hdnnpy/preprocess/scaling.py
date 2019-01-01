# coding: utf-8

"""Scale all feature values into the certain range."""

import numpy as np

from hdnnpy.preprocess.preprocess_base import PreprocessBase
from hdnnpy.utils import (MPI, pprint)


class Scaling(PreprocessBase):
    """Scale all feature values into the certain range."""
    name = 'scaling'
    """str: Name of this class."""

    def __init__(self, min_=0.0, max_=1.0):
        """
        Args:
            min\_ (float): Target minimum value of scaling.
            max\_ (float): Target maximum value of scaling.
        """
        assert isinstance(min_, float)
        assert isinstance(max_, float)
        assert min_ < max_
        super().__init__()
        self._max = {}
        self._min = {}
        self._target_max = max_
        self._target_min = min_

    @property
    def max(self):
        """dict [~numpy.ndarray]: Initialized maximum values in each
        feature dimension and each element."""
        return self._max

    @property
    def min(self):
        """dict [~numpy.ndarray]: Initialized minimum values in each
        feature dimension and each element."""
        return self._min

    @property
    def target(self):
        """tuple [float, float]: Target min & max values of scaling."""
        return self._target_min, self._target_max

    def apply(self, dataset, elemental_composition, verbose=True):
        """Apply the same pre-processing for each element to dataset.

        It accepts 1 or 2 for length of ``dataset``, each element of
        which is regarded as ``0th-order``, ``1st-order``, ...

        Args:
            dataset (list [~numpy.ndarray]): Input dataset to be scaled.
            elemental_composition (list [str]):
                Element symbols corresponding to 1st dimension of
                ``dataset``.
            verbose (bool, optional): Print log to stdout.

        Returns:
            list [~numpy.ndarray]:
                Processed dataset into the same min-max range.
        """
        order = len(dataset) - 1
        assert 0 <= order <= 1

        self._initialize_params(dataset[0], elemental_composition, verbose)

        max_ = np.array(
            [self._max[element] for element in elemental_composition])
        min_ = np.array(
            [self._min[element] for element in elemental_composition])

        if order >= 0:
            dataset[0] = ((dataset[0] - min_)
                          / (max_ - min_)
                          * (self._target_max - self._target_min)
                          + self._target_min)
        if order >= 1:
            dataset[1] = (dataset[1]
                          / (max_ - min_)[..., None, None]
                          * (self._target_max - self._target_min))

        return dataset

    def load(self, file_path, verbose=True):
        """Load internal parameters for each element.

        Only root MPI process loads parameters.

        Args:
            file_path (~pathlib.Path): File path to load parameters.
            verbose (bool, optional): Print log to stdout.
        """
        if MPI.rank != 0:
            return

        ndarray = np.load(file_path)
        self._elements = ndarray['elements'].item()
        self._max = {element: ndarray[f'max:{element}']
                     for element in self._elements}
        self._min = {element: ndarray[f'min:{element}']
                     for element in self._elements}
        if verbose:
            pprint(f'Loaded Scaling parameters from {file_path}.')

    def save(self, file_path, verbose=True):
        """Save internal parameters for each element.

        Only root MPI process saves parameters.

        Args:
            file_path (~pathlib.Path): File path to save parameters.
            verbose (bool, optional): Print log to stdout.
        """
        if MPI.rank != 0:
            return

        info = {'elements': self._elements}
        max_ = {f'max:{k}': v for k, v in self._max.items()}
        min_ = {f'min:{k}': v for k, v in self._min.items()}
        np.savez(file_path, **info, **max_, **min_)
        if verbose:
            pprint(f'Saved Scaling parameters to {file_path}.')

    def _initialize_params(self, data, elemental_composition, verbose):
        """Initialize parameters only once for new elements."""
        for element in set(elemental_composition) - self._elements:
            n_feature = data.shape[2]
            mask = np.array(elemental_composition) == element
            X = data[:, mask].reshape(-1, n_feature)
            self._elements.add(element)
            self._max[element] = X.max(axis=0, dtype=np.float32)
            self._min[element] = X.min(axis=0, dtype=np.float32)
            if verbose:
                pprint(f'Initialized Scaling parameters for {element}')
