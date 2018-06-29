# -*- coding: utf-8 -*-
from __future__ import print_function

from config import mpi

from os import makedirs
from sys import stdout
from itertools import product
from collections import defaultdict
import random
import numpy as np
from chainer import Variable


def pprint(string, root_only=True, flush=False, **options):
    if mpi.rank == 0 or not root_only:
        print(string, **options)
        if flush:
            stdout.flush()


def mkdir(path):
    if mpi.rank == 0:
        makedirs(path)


def write(f, string):
    with open(f, 'a') as f:
        f.write(string)


def flatten_dict(dic):
    return {k: v.data.item() if isinstance(v, Variable)
            else v.item() if isinstance(v, np.float64)
            else v for k, v in dic.iteritems()}


class DictAsAttributes(dict):
    def __init__(self, dic):
        super(DictAsAttributes, self).__init__(self, **dic)

    def __dir__(self):
        return dir(super(DictAsAttributes, self))

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, name):
        if name in dir(DictAsAttributes):
            return self.name

        value = self[name]
        if isinstance(value, defaultdict):
            pass
        elif isinstance(value, list):
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


class HyperParameter(object):
    def __init__(self, dic, random_search):
        self.hyperparameters = dic
        self.random_search = random_search
        self.indices = list(product(*[range(len(v)) for v in dic.values()]))

    def __iter__(self):
        if self.random_search:
            for i in range(self.random_search):
                yield DictAsAttributes({k: random.uniform(min(v), max(v)) if k != 'layer'
                                        else random.choice(v) for k, v in self.hyperparameters.iteritems()})
        else:
            for index in self.indices:
                yield DictAsAttributes({k: v[i] for (k, v), i in zip(self.hyperparameters.iteritems(), index)})

    def __len__(self):
        return self.random_search if self.random_search else len(self.indices)

    def __getitem__(self, n):
        return DictAsAttributes({k: v[i] for (k, v), i in zip(self.hyperparameters.iteritems(), self.indices[n])})
