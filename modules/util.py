# -*- coding: utf-8 -*-

from config import mpi

from os import makedirs
from itertools import product


def mpiprint(str):
    if mpi.rank == 0:
        print str


def mpisave(obj, *args):
    if mpi.rank == 0:
        obj.save(*args)


def mpimkdir(path):
    if mpi.rank == 0:
        makedirs(path)


def mpiwrite(f, str):
    if mpi.rank == 0:
        with open(f, 'a') as f:
            f.write(str)


class DictAsAttributes(dict):
    def __init__(self, dic):
        super(DictAsAttributes, self).__init__(self, **dic)

    def __dir__(self):
        return dir(super(DictAsAttributes, self))

    def __getattr__(self, name):
        if name in dir(self):
            return self.name

        value = self[name]
        if isinstance(value, list):
            value = [DictAsAttributes(v) if isinstance(v, dict) else v for v in value]
        elif isinstance(value, dict):
            value = DictAsAttributes(value)
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __add__(self, other):
        new = self.copy()
        new.update(other)
        return DictAsAttributes(new)


class HyperParameter(DictAsAttributes):
    def __iter__(self):
        keys = self.keys()
        value_list = self.values()
        for values in product(*value_list):
            yield DictAsAttributes({k: v for k, v in zip(keys, values)})

    def __len__(self):
        return reduce(lambda x, y: x*y, [len(v) for v in self.values()])
