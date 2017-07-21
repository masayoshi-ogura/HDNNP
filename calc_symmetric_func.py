# -*- coding: utf-8 -*-

import numpy as np
from quippy import AtomsReader
import my_func

name = 'Ge'

xyz_dir = 'training_data/xyz/'
npy_dir = 'training_data/npy/'
alldataset = AtomsReader(xyz_dir+'AllSiGe.xyz')
rawdataset = [data for data in alldataset if data.config_type == 'CrystalSi0Ge8' and data.cohesive_energy < -10.0]
cordinates = [data for data in rawdataset]
nsample = len(rawdataset)
natom = 8
Es = np.array([data.cohesive_energy for data in rawdataset])
Fs = np.array([np.array(data.force).T for data in rawdataset]).reshape((nsample,3*natom))
a = cordinates[0].lattice[1][1]
Rcs = [a]
Rss = [1.0]
etas = [0.0]
gnum = len(Rcs)*len(Rss)*len(etas)
Gs,dGs = my_func.symmetric_func(cordinates, natom, nsample, gnum, Rcs, Rss, etas)
np.save(npy_dir+name+'-Es.npy', Es)
np.save(npy_dir+name+'-Fs.npy', Fs)
np.save(npy_dir+name+'-Gs.npy', Gs)
np.save(npy_dir+name+'-dGs.npy', dGs)
