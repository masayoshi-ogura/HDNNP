# -*- coding: utf-8 -*-

__all__ = [
    'PreprocessBase',
    ]

from abc import (ABC,
                 abstractmethod,
                 )


class PreprocessBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

    @abstractmethod
    def dump_params(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass
