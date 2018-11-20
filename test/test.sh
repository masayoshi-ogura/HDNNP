#!/bin/sh
set -ux

timeout="gtimeout 5"
mpirun="mpirun -np 2"
hdnnpy="python -W ignore ../hdnnpy"

rm -rf data/{CrystalGa16N16,CrystalGa2N2,config_type.pickle,Symmetry_Function.npz} output/ __pycache__/

rm -rf data/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${hdnnpy} sym_func

rm -rf data/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${mpirun} ${hdnnpy} sym_func

rm -rf data/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${hdnnpy} training --verbose

rm -rf data/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${mpirun} ${hdnnpy} training --verbose

rm -rf data/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${hdnnpy} param_search --verbose

rm -rf data/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${mpirun} ${hdnnpy} param_search --verbose

${hdnnpy} training --verbose

${mpirun} ${hdnnpy} training --verbose

(${timeout} ${hdnnpy} training --verbose)
${hdnnpy} training --verbose --resume output/CrystalGa16N16

(${timeout} ${mpirun} ${hdnnpy} training --verbose)
${mpirun} ${hdnnpy} training --verbose --resume output/CrystalGa16N16

${hdnnpy} prediction --poscar data/POSCAR --masters output/masters.npz

${hdnnpy} phonon --poscar data/POSCAR --masters output/masters.npz
