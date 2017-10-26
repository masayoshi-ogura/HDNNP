# -*- coding: utf-8 -*-

from config import mpi

from os import makedirs


def mpiprint(str):
    if mpi.rank == 0:
        print str


def mpisave(obj, *args):
    if mpi.rank == 0:
        obj.save(*args)


def mpimkdir(path):
    if mpi.rank == 0:
        makedirs(path)
