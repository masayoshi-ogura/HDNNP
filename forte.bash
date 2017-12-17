#!/bin/sh
#PBS -N hogehoge
#PBS -j oe
#PBS -q default
#PBS -l nodes=1:ppn=12:B

NPROCS=`wc -l <$PBS_NODEFILE`
cd $PBS_O_WORKDIR
source /var/mpi-selector/data/openmpi-1.5.1-intel64-v12.0.0u1.sh

mpirun -machinefile $PBS_NODEFILE -np $NPROCS ./hdnnpy -e ${epoch} -c ${cv}
