# -*- coding: utf-8 -*-
from __future__ import print_function

import settings as stg

from pprint import pprint as pretty_print
from os import makedirs
from sys import stdout
from collections import defaultdict
import numpy as np
from chainer import Variable


def pprint(data, root_only=True, flush=False, **options):
    if stg.mpi.rank == 0 or not root_only:
        if isinstance(data, list) or isinstance(data, dict):
            pretty_print(data, **options)
        else:
            print(data, **options)
        if flush:
            stdout.flush()


def mkdir(path):
    if stg.mpi.rank == 0:
        makedirs(path)


def write(f, string):
    with open(f, 'a') as f:
        f.write(string)


def flatten_dict(dic):
    return {k: v.data.item() if isinstance(v, Variable)
            else v.item() if isinstance(v, np.float64)
            else v for k, v in dic.iteritems()}


class DictAsAttributes(dict):
    def __init__(self, **dic):
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
            value = [DictAsAttributes(**v) if isinstance(v, dict) else v for v in value]
        elif isinstance(value, dict):
            value = DictAsAttributes(**value)
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __add__(self, other):
        new = self.copy()
        new.update(other)
        return DictAsAttributes(**new)
