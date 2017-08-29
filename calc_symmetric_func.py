# -*- coding: utf-8 -*-

# define variables
from config import hp,other

# import python modules
from mpi4py import MPI
from quippy import AtomsReader

# import own modules
import my_func

# set MPI variables
allcomm = MPI.COMM_WORLD
allrank = allcomm.Get_rank()
allsize = allcomm.Get_size()

# set variables to all procs
weight_dir = 'weight_params/'
train_dir = 'training_data/'
train_xyz_dir = train_dir+'xyz/'
train_npy_dir = train_dir+'npy/'

alldataset = AtomsReader(train_xyz_dir+other.xyzfile)
coordinates = [data for data in alldataset if data.config_type == other.name and data.cohesive_energy < 0.0]
hp.nsample = len(coordinates)
Es,Fs = my_func.calc_EF(coordinates, train_npy_dir, other.name, hp.natom, hp.nsample)
hp.ninput = len(hp.Rcs) + len(hp.Rcs)*len(hp.etas)*len(hp.Rss) + len(hp.Rcs)*len(hp.etas)*len(hp.lams)*len(hp.zetas)
Gs,dGs = my_func.load_or_calc_G(allcomm, allsize, allrank, coordinates, train_npy_dir, other.name, hp.Rcs, hp.etas, hp.Rss, hp.lams, hp.zetas, hp.natom, hp.nsample, hp.ninput)
