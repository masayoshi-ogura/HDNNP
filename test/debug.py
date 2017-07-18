# -*- coding: utf-8 -*-

import time
from mpi4py import MPI
import numpy as np
import random

# set MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a = np.random.random((3,4))
b = np.zeros_like(a)

comm.Allreduce(a, b, op=MPI.SUM)

file = open('rank'+str(rank), 'w')
for line in a:
    for val in line:
        file.write(str(val)+',')
    file.write('\n')
for line in b:
    for val in line:
        file.write(str(val)+',')
    file.write('\n')
file.close()
