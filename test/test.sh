#!/bin/sh
set -euxp

mpirun="mpirun -np 2"
hdnnpy="hdnnpy"

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

${hdnnpy} test --poscar data/POSCAR --masters output/masters.npz

${hdnnpy} phonon --poscar data/POSCAR --masters output/masters.npz

${hdnnpy} optimize --poscar data/POSCAR --masters output/masters.npz
