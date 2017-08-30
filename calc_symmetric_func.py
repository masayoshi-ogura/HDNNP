# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import other

# import python modules
from os import path
from os import mkdir
from mpi4py import MPI
from quippy import AtomsReader

# import own modules
from modules.input import Generator

# set MPI variables
allcomm = MPI.COMM_WORLD
allrank = allcomm.Get_rank()
allsize = allcomm.Get_size()

# set variables to all procs
train_dir = 'training_data'
train_xyz_file = path.join(train_dir, 'xyz', other.xyzfile)
train_npy_dir = path.join(train_dir, 'npy', other.name)
if not path.exists(train_npy_dir):
    mkdir(train_npy_dir)

generator = Generator(train_npy_dir, other.name, hp.Rcs, hp.etas, hp.Rss, hp.lams, hp.zetas)

alldataset = AtomsReader(train_xyz_file)
coordinates = []
for data in alldataset:
    if data.config_type == other.name and data.cohesive_energy < 0.0:
        coordinates.append(data)
hp.nsample = len(coordinates)
Es, Fs = generator.calc_EF(coordinates, hp.natom, hp.nsample)
hp.ninput = len(hp.Rcs) + \
    len(hp.Rcs)*len(hp.etas)*len(hp.Rss) + \
    len(hp.Rcs)*len(hp.etas)*len(hp.lams)*len(hp.zetas)
Gs, dGs = generator.calc_G(allcomm, allsize, allrank, coordinates, hp.natom, hp.nsample, hp.ninput)
