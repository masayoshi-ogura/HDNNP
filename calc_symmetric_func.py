# -*- coding: utf-8 -*-

import numpy as np
from quippy import AtomsReader
import my_func

weight_dir = 'weight_params/'
name = 'Ge'

train_dir = 'training_data/xyz/'
alldataset = AtomsReader(train_dir+'AllSiGe.xyz')
rawdataset = [data for data in alldataset if data.config_type == 'CrystalSi0Ge8' and data.cohesive_energy < 0.0]
cordinates = [data for data in rawdataset]
nsample = len(rawdataset)
natom = 8
Es = np.array([data.cohesive_energy for data in rawdataset])
Fs = np.array([np.array(data.force).T for data in rawdataset]).reshape((nsample,3*natom))
Rcs = [cordinates[0].lattice[1][1]]
Rss = [1.0,2.0,3.0,4.0,5.0]
etas = [0.0,0.1,0.5,1.0,1.5]
gnum = len(Rcs)*len(Rss)*len(etas)
Gs,dGs = my_func.symmetric_func(cordinates, natom, nsample, gnum, Rcs, Rss, etas)
np.save(weight_dir+name+'-Es.npy', Es)
np.save(weight_dir+name+'-Fs.npy', Fs)
np.save(weight_dir+name+'-Gs.npy', Gs)
np.save(weight_dir+name+'-dGs.npy', dGs)
