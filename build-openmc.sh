#! ~/bin/bash
set -e
# Paths
HDF5_PATH=$CONDA_PREFIX

# Build Openmc from source
mkdir -p build app && cd build
CC=gcc \
CXX=g++ \
HDF5_ROOT=$HDF5_PATH \
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_FIND_ROOT_PATH=$CONDA_PREFIX \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DCMAKE_INSTALL_PREFIX=~/projects/openmc/app ..
make -j 12
make install


