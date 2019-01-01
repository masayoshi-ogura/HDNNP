# coding: utf-8

"""Principal component analysis (PCA)."""

import numpy as np
from sklearn import decomposition

from hdnnpy.preprocess.preprocess_base import PreprocessBase
from hdnnpy.utils import (MPI, pprint)


class PCA(PreprocessBase):
    """Principal component analysis (PCA).

    The core part of this class uses `sklearn.decomposition.PCA`
    implementation.
    """
    name = 'pca'
    """str: Name of this class."""

    def __init__(self, n_components=None):
        """
        Args:
            n_components (int, optional):
                Number of features to keep in decomposition. If
                ``None``, decomposition is not performed.
        """
        super().__init__()
        self._n_components = n_components
        self._mean = {}
        self._transform = {}

    @property
    def n_components(self):
        """int or None: Number of features to keep in decomposition."""
        return self._n_components

    @property
    def mean(self):
        """dict [~numpy.ndarray]: Initialized mean values in each
        feature dimension and each element."""
        return self._mean

    @property
    def transform(self):
        """dict [~numpy.ndarray]: Initialized transformation matrix in
        each feature dimension and each element."""
        return self._transform

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
        transform = np.array(
            [self._transform[element] for element in elemental_composition])

        if order >= 0:
            dataset[0] = np.einsum('ijk,jkl->ijl', dataset[0]-mean, transform)
        if order >= 1:
            dataset[1] = np.einsum('ijkmn,jkl->ijlmn', dataset[1], transform)

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
        self._n_components = ndarray['n_components'].item()
        self._mean = {element: ndarray[f'mean:{element}']
                      for element in self._elements}
        self._transform = {element: ndarray[f'transform:{element}']
                           for element in self._elements}
        if verbose:
            pprint(f'Loaded PCA parameters from {file_path}.')

    def save(self, file_path, verbose=True):
        """Save internal parameters for each element.

        Only root MPI process saves parameters.

        Args:
            file_path (~pathlib.Path): File path to save parameters.
            verbose (bool, optional): Print log to stdout.
        """
        if MPI.rank != 0:
            return

        info = {
            'elements': self._elements,
            'n_components': self._n_components,
            }
        mean = {f'mean:{k}': v for k, v in self._mean.items()}
        transform = {f'transform:{k}': v
                     for k, v in self._transform.items()}
        np.savez(file_path, **info, **mean, **transform)
        if verbose:
            pprint(f'Saved PCA parameters to {file_path}.')

    def _initialize_params(self, data, elemental_composition, verbose):
        """Initialize parameters only once for new elements."""
        for element in set(elemental_composition) - self._elements:
            n_feature = data.shape[2]
            mask = np.array(elemental_composition) == element
            X = data[:, mask].reshape(-1, n_feature)
            pca = decomposition.PCA(n_components=self._n_components)
            pca.fit(X)
            if self._n_components is None:
                self._n_components = n_feature
            self._elements.add(element)
            self._mean[element] = pca.mean_.astype(np.float32)
            self._transform[element] = pca.components_.T.astype(np.float32)
            if verbose:
                pprint(f'''
Initialized PCA parameters for {element}
    Feature dimension: {n_feature} => {self._n_components}
    Cumulative contribution rate = {np.sum(pca.explained_variance_ratio_)}
''')
