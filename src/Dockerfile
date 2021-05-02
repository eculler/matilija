# Initialize the build with compiler
FROM ubuntu:bionic-20200311
RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y libnetcdf-dev
RUN apt-get install -y libx11-dev
RUN apt-get install -y cmake

# Copy over DHSVM source code
WORKDIR /matilija/src/dhsvm
COPY dhsvm3.2 /matilija/src/dhsvm

# Compile DHSVM
WORKDIR /matilija/src/dhsvm/build
RUN cmake -D CMAKE_BUILD_TYPE:STRING=Release \
  -D DHSVM_USE_X11:BOOL=ON \
  -D DHSVM_USE_NETCDF:BOOL=ON \
  ..
RUN cmake --build .

# Move to main directory
WORKDIR /matilija
