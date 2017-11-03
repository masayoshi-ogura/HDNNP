# -*- coding: utf-8 -*-

from config import mpi

from os import makedirs
import numpy as np
from itertools import combinations


def mpiprint(str):
    if mpi.rank == 0:
        print str


def mpisave(obj, *args):
    if mpi.rank == 0:
        obj.save(*args)


def mpimkdir(path):
    if mpi.rank == 0:
        makedirs(path)
    mpi.comm.Barrier()


def mpiwrite(f, str):
    if f.__class__ is file:
        f.write(str)
    else:
        f.Write(str)


def rmse(pred, true):
    return np.sqrt(((pred - true)**2).mean())


def comb(n, r):
    for c in combinations(xrange(1, n), r-1):
        ret = []
        low = 0
        for p in c:
            ret.append(p - low)
            low = p
        ret.append(n - low)
        yield ret
