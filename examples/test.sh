#!/bin/sh
set -eux

clean="rm -rf data/CrystalGa16N16 data/CrystalGa2N2 data/config_type.pickle data/Symmetry_Function.npz output/ __pycache__/"
timeout="gtimeout 5"
mpirun="mpirun -np 2"
hdnnpy="hdnnpy"

${clean}; ${hdnnpy} training --verbose

${clean}; ${mpirun} ${hdnnpy} training --verbose

${clean}; ${hdnnpy} param_search

${clean}; ${mpirun} ${hdnnpy} param_search

${clean}; ${hdnnpy} sym_func

${clean}; ${mpirun} ${hdnnpy} sym_func


set +e
(${timeout} ${hdnnpy} training --verbose)
set -e
${hdnnpy} training --verbose --resume output/CrystalGa16N16

set +e
(${timeout} ${mpirun} ${hdnnpy} training --verbose)
set -e
${mpirun} ${hdnnpy} training --verbose --resume output/CrystalGa16N16


${hdnnpy} prediction --poscar data/POSCAR --masters output/masters.npz

${hdnnpy} phonon --poscar data/POSCAR --masters output/masters.npz
