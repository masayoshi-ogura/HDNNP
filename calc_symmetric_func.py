# -*- coding: utf-8 -*-

import numpy as np
from quippy import AtomsReader
import my_func

name = 'Ge'

train_dir = 'training_data/xyz/'
alldataset = AtomsReader(train_dir+'AllSiGe.xyz')
rawdataset = [data for data in alldataset if data.config_type == 'CrystalSi0Ge8' and data.cohesive_energy < 0.0]
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
np.save(train_dir+name+'-Es.npy', Es)
np.save(train_dir+name+'-Fs.npy', Fs)
np.save(train_dir+name+'-Gs.npy', Gs)
np.save(train_dir+name+'-dGs.npy', dGs)
