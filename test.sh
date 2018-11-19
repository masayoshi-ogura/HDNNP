#!/bin/sh
set -euxp

trap finally EXIT

function finally {
  cp ${tmp1} settings.py
  cp ${tmp2} phonopy_settings.py
  rm ${tmp1} ${tmp2}
}

debug_path=./test/data
tmp1=$(mktemp)
tmp2=$(mktemp)

mpirun="mpirun -np 2"
hdnnpy="python -W ignore hdnnpy"

cp settings.py ${tmp1}
cp phonopy_settings.py ${tmp2}
cp test/settings.py settings.py
cp test/phonopy_settings.py phonopy_settings.py


rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${hdnnpy} sym_func > /dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${mpirun} ${hdnnpy} sym_func > /dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${hdnnpy} training --verbose > /dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${mpirun} ${hdnnpy} training --verbose > /dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${hdnnpy} param_search --verbose > /dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${mpirun} ${hdnnpy} param_search --verbose > /dev/null

${hdnnpy} training --verbose > /dev/null

${mpirun} ${hdnnpy} training --verbose > /dev/null

${hdnnpy} test --poscar test/POSCAR --masters test/output/masters.npz > /dev/null

${hdnnpy} phonon --poscar test/POSCAR --masters test/output/masters.npz > /dev/null

${hdnnpy} optimize --poscar test/POSCAR --masters test/output/masters.npz > /dev/null
