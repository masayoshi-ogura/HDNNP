#!/bin/bash
#$ -cwd
#$ -V -S /bin/bash
#$ -N hogehoge
#$ -pe smp 24

export PATH=/home/ogura/python/Python-2.7.9/bin:$PATH
export PYTHONPATH=/home/ogura/python/Python-2.7.9/lib/python-2.7/site-packages:$PYTHONPATH
export PYTHONHOME=/home/ogura/python/Python-2.7.9

mpirun -np 24 ./hdnnpy -e ${epoch} -c ${cv}
