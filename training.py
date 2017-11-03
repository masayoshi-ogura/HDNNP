# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import file_
from config import mpi

# import python modules
from os import path
from os import copy2
from datetime import datetime
from mpi4py import MPI

# import own modules
from modules.data import DataGenerator
from modules.model import HDNNP
from modules.util import mpimkdir
from modules.util import mpisave

datestr = datetime.now().strftime('%m%d-%H%M%S')
save_dir = path.join(file_.save_dir, datestr)
out_dir = path.join(file_.out_dir, datestr)
mpimkdir(save_dir)
mpimkdir(out_dir)
copy2('config.py', path.join(out_dir, 'config.py'))
progress = MPI.File.Open(mpi.comm, path.join(out_dir, 'progress.dat'), MPI.MODE_CREATE | MPI.MODE_WRONLY)

generator = DataGenerator('training', preconditioning=hp.preconditioning)
for config, dataset in generator:
    output_file = path.join(out_dir, '{}.npz'.format(config))
    hdnnp = HDNNP(dataset.ninput)
    hdnnp.load(save_dir)
    hdnnp.fit(dataset, progress=progress)

    hdnnp.save(save_dir, output_file)
    mpi.comm.Barrier()
mpisave(generator, save_dir)
progress.Close()
