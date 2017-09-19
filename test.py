# -*- coding: utf-8 -*-

# import python modules
from datetime import datetime
from mpi4py import MPI

# import own modules
from modules.generator import make_dataset
from modules.model import HDNNP

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Es, Fs, Gs, dGs, natom, nsample, ninput, composition = make_dataset(comm, rank, size)

# initialize HDNNP
hdnnp = HDNNP(comm, rank, size, natom, nsample, ninput, composition)
hdnnp.load_w()

# test
E_RMSE, F_RMSE, RMSE = hdnnp.calc_RMSE(0, Es, Fs, Gs, dGs)
if rank == 0:
    datestr = datetime.now().strftime('%m%d-%H%M%S')
    with open('progress-'+datestr+'.out', 'w') as file:
        file.write('E_RMSE: {}\nF_RMSE: {}\nRMSE: {}\n'.format(E_RMSE, F_RMSE, RMSE))
hdnnp.save_fig('png')
