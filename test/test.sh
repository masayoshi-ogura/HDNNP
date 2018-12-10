#!/bin/sh
set -eux

clean="rm -rf data/CrystalGa16N16 data/CrystalGa2N2 data/GaN.xyz.tag output/ __pycache__/"
timeout="gtimeout 5"
mpirun="mpirun -np 2"
hdnnpy="hdnnpy"

${clean}; ${hdnnpy} train --verbose

${clean}; ${mpirun} ${hdnnpy} train --verbose

${clean}; ${hdnnpy} param-search

${clean}; ${mpirun} ${hdnnpy} param-search

${clean}; ${hdnnpy} sym-func

${clean}; ${mpirun} ${hdnnpy} sym-func


set +e
(${timeout} ${hdnnpy} train --verbose)
set -e
${hdnnpy} train --verbose --resume output/CrystalGa16N16

set +e
(${timeout} ${mpirun} ${hdnnpy} train --verbose)
set -e
${mpirun} ${hdnnpy} train --verbose --resume output/CrystalGa16N16


${hdnnpy} predict energy force --poscar data/POSCAR --masters output/masters.npz --write output/prediction.dat
