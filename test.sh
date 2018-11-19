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
${hdnnpy} sym_func

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${mpirun} ${hdnnpy} sym_func

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${hdnnpy} training --verbose

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${mpirun} ${hdnnpy} training --verbose

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${hdnnpy} param_search --verbose

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.pickle}
${mpirun} ${hdnnpy} param_search --verbose

${hdnnpy} training --verbose

${mpirun} ${hdnnpy} training --verbose

${hdnnpy} test --poscar test/POSCAR --masters test/output/masters.npz

${hdnnpy} phonon --poscar test/POSCAR --masters test/output/masters.npz

${hdnnpy} optimize --poscar test/POSCAR --masters test/output/masters.npz
