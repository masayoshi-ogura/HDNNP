# coding: utf-8

"""Pre-processing of input dataset subpackage."""

__all__ = [
    'PREPROCESS',
    ]

from hdnnpy.preprocess.pca import PCA
from hdnnpy.preprocess.scaling import Scaling
from hdnnpy.preprocess.standardization import Standardization

PREPROCESS = {
    PCA.name: PCA,
    Scaling.name: Scaling,
    Standardization.name: Standardization,
    }
