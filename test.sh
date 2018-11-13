#!/bin/sh
set -euxp

trap finally EXIT

function finally {
  cp ${tmp1} settings.py
  cp ${tmp2} phonopy_settings.py
  rm ${tmp1} ${tmp2}
}

debug_path=./GaN/debug_test
tmp1=$(mktemp)
tmp2=$(mktemp)

cp settings.py ${tmp1}
cp phonopy_settings.py ${tmp2}
cp settings.py.sample settings.py
cp phonopy_settings.py.sample phonopy_settings.py


rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
./hdnnpy sym_func >/dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
mpirun -np 2 ./hdnnpy sym_func >/dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
./hdnnpy training --verbose >/dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
mpirun -np 2 ./hdnnpy training --verbose >/dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
./hdnnpy param_search --verbose >/dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
mpirun -np 2 ./hdnnpy param_search --verbose >/dev/null

./hdnnpy training --verbose >/dev/null

mpirun -np 2 ./hdnnpy training --verbose >/dev/null

./hdnnpy test >/dev/null

./hdnnpy phonon >/dev/null

./hdnnpy optimize >/dev/null


