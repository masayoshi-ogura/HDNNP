# coding: utf-8

"""Base class of pre-processing.

If you want to add new pre-processing method to extend HDNNP, inherits
this base class.
"""

from abc import (ABC, abstractmethod)


class PreprocessBase(ABC):
    """Base class of pre-processing."""
    name = None
    """str: Name of this class."""

    def __init__(self):
        """
        Initialize private variable :attr:`_elements` as a empty `set`.
        """
        self._elements = set()

    @property
    def elements(self):
        """list [str]: List of elements whose parameters have already
        been initialized."""
        return sorted(self._elements)

    @abstractmethod
    def apply(self, *args, **kwargs):
        """Apply the same pre-processing for each element to dataset.

        This is abstract method.
        Subclass of this base class have to override.
        """
        pass

    @abstractmethod
    def dump_params(self):
        """Dump its own parameters as :obj:`str`.

        This is abstract method.
        Subclass of this base class have to override.
        """
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        """Load internal parameters for each element.

        This is abstract method.
        Subclass of this base class have to override.
        """
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        """Save internal parameters for each element.

        This is abstract method.
        Subclass of this base class have to override.
        """
        pass
