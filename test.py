# -*- coding: utf-8 -*-

from config import file_

# import python modules
from datetime import datetime
from mpi4py import MPI

# import own modules
from modules.generator import make_dataset
from modules.model import HDNNP

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

datestr = datetime.now().strftime('%m%d-%H%M%S')
if rank == 0:
    file = open('test-'+datestr+'.out', 'w')

for ret in make_dataset(comm, rank, size):
    config, Es, Fs, Gs, dGs, natom, nsample, ninput, composition = ret

    # initialize HDNNP
    hdnnp = HDNNP(comm, rank, size, natom, nsample, ninput, composition)
    hdnnp.load_w(file_.test_weight)

    # test
    E_RMSE, F_RMSE, RMSE = hdnnp.calc_RMSE(0, Es, Fs, Gs, dGs)
    if rank == 0:
        file.write('E_RMSE: {}\nF_RMSE: {}\nRMSE: {}\n'.format(E_RMSE, F_RMSE, RMSE))
        file.flush()

    # save
    hdnnp.save_fig(datestr, config, 'png')

if rank == 0:
    file.close()