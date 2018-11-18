# -*- coding: utf-8 -*-
import settings as stg

from pprint import pprint as pretty_print
from os import makedirs
from sys import stdout
from collections import defaultdict
import numpy as np
from chainer import Variable


def pprint(data, root_only=True, flush=True, **options):
    if stg.mpi['rank'] == 0 or not root_only:
        if isinstance(data, list) or isinstance(data, dict):
            pretty_print(data, **options)
        else:
            print(data, **options)
        if flush:
            stdout.flush()


def mkdir(path):
    if stg.mpi['rank'] == 0:
        makedirs(path, exist_ok=True)


def write(f, string):
    with open(f, 'a') as f:
        f.write(string)


def flatten_dict(dic):
    return {k: v.data.item() if isinstance(v, Variable)
            else v.item() if isinstance(v, np.float64)
            else v for k, v in dic.items()}
