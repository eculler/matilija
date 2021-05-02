#/bin/bash
PREFIX=$PWD
RUNID=$1

docker build --tag=dhsvm-matilija src
docker run \
  -v $PREFIX/cfg:/matilija/cfg \
  -v $PREFIX/input:/matilija/input \
  -v $PREFIX/state:/matilija/state \
  -v $PREFIX/output/${RUNID}:/matilija/output \
  -t dhsvm-matilija:latest \
  /matilija/src/dhsvm3.2/build/DHSVM/sourcecode/DHSVM \
  /matilija/cfg/${RUNID}.cfg
