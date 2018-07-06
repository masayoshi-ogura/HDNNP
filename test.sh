#!/bin/sh
set -euxp

debug_path=./GaN/debug_test
tmp1=$(mktemp)
tmp2=$(mktemp)

cp settings.py ${tmp1}
cp phonopy_settings.py ${tmp2}
cp settings.py.sample settings.py
cp phonopy_settings.py.sample phonopy_settings.py


rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
./hdnnpy sym_func > /dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
mpirun -np 4 ./hdnnpy sym_func > /dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
./hdnnpy training --verbose > /dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
mpirun -np 4 ./hdnnpy training --verbose > /dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
./hdnnpy param_search --kfold 2 --init 1 --max-iter 1 > /dev/null

rm -rf ${debug_path}/{CrystalGa16N16,CrystalGa2N2,config_type.dill}
mpirun -np 4 ./hdnnpy param_search --kfold 2 --init 1 --max-iter 1 > /dev/null

./hdnnpy training --verbose > /dev/null

mpirun -np 4 ./hdnnpy training --verbose > /dev/null

./hdnnpy test > /dev/null

./hdnnpy phonon > /dev/null

./hdnnpy optimize > /dev/null


cp ${tmp1} settings.py
cp ${tmp2} phonopy_settings.py
rm ${tmp1} ${tmp2}