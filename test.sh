#!/bin/sh

home_path=/Users/jas/Desktop/HDNNP
debug_path=${home_path}/GaN/debug_test

cd $debug_path; rm -rf CrystalGa16N16 CrystalGa2N2 config_type.dill; cd $home_path
echo "test './hdnnpy sf'"
./hdnnpy sf > /dev/null

cd $debug_path; rm -rf CrystalGa16N16 CrystalGa2N2 config_type.dill; cd $home_path
echo "test 'mpirun -np 4 ./hdnnpy sf'"
mpirun -np 4 ./hdnnpy sf > /dev/null

cd $debug_path; rm -rf CrystalGa16N16 CrystalGa2N2 config_type.dill; cd $home_path
echo "test './hdnnpy training -e 10'"
./hdnnpy training -e 10 > /dev/null

cd $debug_path; rm -rf CrystalGa16N16 CrystalGa2N2 config_type.dill; cd $home_path
echo "test 'mpirun -np 4 ./hdnnpy training -e 10'"
mpirun -np 4 ./hdnnpy training -e 10 > /dev/null

cd $debug_path; rm -rf CrystalGa16N16 CrystalGa2N2 config_type.dill; cd $home_path
echo "test './hdnnpy cv -k 2 -e 10 --random-search 5'"
./hdnnpy cv -k 2 -e 10 --random-search 5 > /dev/null

cd $debug_path; rm -rf CrystalGa16N16 CrystalGa2N2 config_type.dill; cd $home_path
echo "test 'mpirun -np 4 ./hdnnpy cv -k 2 -e 10 --random-search 5'"
mpirun -np 4 ./hdnnpy cv -k 2 -e 10 --random-search 5 > /dev/null

echo "test './hdnnpy training -e 10'"
./hdnnpy training -e 10 > /dev/null

echo "test 'mpirun -np 4 ./hdnnpy training -e 10'"
mpirun -np 4 ./hdnnpy training -e 10 > /dev/null

echo "test './hdnnpy phonon'"
./hdnnpy phonon > /dev/null

echo "test './hdnnpy optimize'"
./hdnnpy optimize > /dev/null
