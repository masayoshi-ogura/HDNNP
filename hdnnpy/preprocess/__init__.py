# coding: utf-8

"""Pre-processing of input dataset subpackage."""

__all__ = [
    'PREPROCESS',
    ]

from hdnnpy.preprocess.normalization import Normalization
from hdnnpy.preprocess.pca import PCA
from hdnnpy.preprocess.standardization import Standardization

PREPROCESS = {
    Normalization.name: Normalization,
    PCA.name: PCA,
    Standardization.name: Standardization,
    }
