#!/bin/sh
#PBS -N hogehoge
#PBS -j oe
#PBS -q default
#PBS -l nodes=1:ppn=8:B

NPROCS=`wc -l <$PBS_NODEFILE`
cd $PBS_O_WORKDIR
source /var/mpi-selector/data/openmpi-1.5.1-intel64-v12.0.0u1.sh

if [ "${mode}" = 'train' -o -z "${mode}" ]; then
  mpirun -machinefile $PBS_NODEFILE -np $NPROCS python train.py
elif [ "${mode}" = 'test' ]; then
  mpirun -machinefile $PBS_NODEFILE -np $NPROCS python test.py
else
  echo 'Usage: qsub forte.bash -v mode=train|test'
  echo "       if you don't define \$mode, 'train' will be chosen."
fi
