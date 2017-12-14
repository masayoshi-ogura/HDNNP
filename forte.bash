#!/bin/sh
#PBS -N hogehoge
#PBS -j oe
#PBS -q default
#PBS -l nodes=1:ppn=12:B

NPROCS=`wc -l <$PBS_NODEFILE`
cd $PBS_O_WORKDIR
source /var/mpi-selector/data/openmpi-1.5.1-intel64-v12.0.0u1.sh

if [ "${mode}" = 'training' -o -z "${mode}" ]; then
  # mpirun -machinefile $PBS_NODEFILE -np $NPROCS python training.py
  python training.py
elif [ "${mode}" = 'test' ]; then
  # mpirun -machinefile $PBS_NODEFILE -np $NPROCS python test.py
  echo 'Not implemented.'
elif [ "${mode}" = 'preproc' ]; then
  mpirun -machinefile $PBS_NODEFILE -np $NPROCS python modules/data.py
else
  echo 'Usage: qsub -v mode=training|test|preproc forte.bash'
  echo "       if you don't define \$mode, 'training' will be chosen."
fi
