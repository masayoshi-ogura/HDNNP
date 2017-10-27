#!/bin/bash
#$ -cwd
#$ -V -S /bin/bash
#$ -N hogehoge
#$ -pe smp 24

export PATH=/home/ogura/python/Python-2.7.9/bin:$PATH
export PYTHONPATH=/home/ogura/python/Python-2.7.9/lib/python-2.7/site-packages:$PYTHONPATH
export PYTHONHOME=/home/ogura/python/Python-2.7.9

if [ "${mode}" = 'training' -o -z "${mode}" ]; then
  mpirun -np 24 python training.py
  python visualize.py
elif [ "${mode}" = 'test' ]; then
  mpirun -np 24 python test.py
elif [ "${mode}" = 'preproc' ]; then
  mpirun -np 24 python modules/data.py
else
  echo 'Usage: qsub -v mode=training|test|preproc iris.bash'
  echo "       if you don't define \$mode, 'training' will be chosen."
fi
