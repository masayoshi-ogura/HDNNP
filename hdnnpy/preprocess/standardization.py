# coding: utf-8

"""Scale all feature values to be zero-mean and unit-variance."""

import numpy as np

from hdnnpy.preprocess.preprocess_base import PreprocessBase
from hdnnpy.utils import (MPI, pprint)


class Standardization(PreprocessBase):
    """Scale all feature values to be zero-mean and unit-variance."""
    name = 'standardization'
    """str: Name of this class."""

    def __init__(self):
        super().__init__()
        self._mean = {}
        self._std = {}

    @property
    def mean(self):
        """dict [~numpy.ndarray]: Initialized mean values in each
        feature dimension and each element."""
        return self._mean

    @property
    def std(self):
        """dict [~numpy.ndarray]: Initialized standard deviation values
        in each feature dimension and each element."""
        return self._std

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
                Processed dataset to be zero-mean and unit-variance.
        """
        order = len(dataset) - 1
        assert 0 <= order <= 1

        self._initialize_params(dataset[0], elemental_composition, verbose)

        mean = np.array(
            [self._mean[element] for element in elemental_composition])
        std = np.array(
            [self._std[element] for element in elemental_composition])

        if order >= 0:
            dataset[0] -= mean
            dataset[0] /= std
        if order >= 1:
            dataset[1] /= std[..., None, None]

        return dataset

    def dump_params(self):
        """Dump its own parameters as :obj:`str`.

        Returns:
            str: Formed parameters.
        """
        params_str = ''
        for element in self._elements:
            mean = self._mean[element]
            std = self._std[element]
            mean_str = ' '.join(map(str, mean))
            std_str = ' '.join(map(str, std))

            params_str += f'''
            {element} {mean.shape[0]}
            # mean
            {mean_str}
            # standard deviation
            {std_str}
            '''

        return params_str

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
        self._mean = {element: ndarray[f'mean:{element}']
                      for element in self._elements}
        self._std = {element: ndarray[f'std:{element}']
                     for element in self._elements}
        if verbose:
            pprint(f'Loaded Standardization parameters from {file_path}.')

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
        mean = {f'mean:{k}': v for k, v in self._mean.items()}
        std = {f'std:{k}': v for k, v in self._std.items()}
        np.savez(file_path, **info, **mean, **std)
        if verbose:
            pprint(f'Saved Standardization parameters to {file_path}.')

    def _initialize_params(self, data, elemental_composition, verbose):
        """Initialize parameters only once for new elements."""
        for element in set(elemental_composition) - self._elements:
            n_feature = data.shape[2]
            mask = np.array(elemental_composition) == element
            X = data[:, mask].reshape(-1, n_feature)
            self._elements.add(element)
            self._mean[element] = X.mean(axis=0)
            self._std[element] = X.std(axis=0, ddof=1)
            if verbose:
                pprint(f'Initialized Standardization parameters for {element}')
