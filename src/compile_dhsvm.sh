#!/bin/bash
# Compile DHSVM v3.2 on Mahti.
# Run this from the login node; output binary goes to $DHSVM_INSTALL/DHSVM.
# Usage: bash compile_dhsvm.sh <project>
#   e.g. bash compile_dhsvm.sh project_2012345

set -euo pipefail

PROJECT=${1:?Usage: $0 <project>}
DHSVM_INSTALL="/projappl/${PROJECT}/dhsvm"

# --- modules ---
# Check available names first if these fail:
#   module spider netcdf-c
#   module spider hdf5
module load gcc
module load cmake
module load netcdf-c
module load hdf5

# --- download source ---
WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

cd "$WORKDIR"
curl --fail -L -o dhsvm3.2.tar.gz \
    https://codeload.github.com/pnnl/DHSVM-PNNL/tar.gz/refs/tags/v3.2
tar -xzf dhsvm3.2.tar.gz
cd DHSVM-PNNL-3.2

# --- build ---
mkdir build && cd build
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D DHSVM_USE_X11:BOOL=OFF \
    -D DHSVM_USE_NETCDF:BOOL=ON \
    ..
cmake --build . -- -j4

# --- install ---
mkdir -p "$DHSVM_INSTALL"
cp DHSVM/sourcecode/DHSVM "$DHSVM_INSTALL/DHSVM"
echo "Installed: $DHSVM_INSTALL/DHSVM"
