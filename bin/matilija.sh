#/bin/bash
RUNID=$1
DATAPREFIX=$2
CFGPREFIX=$3

docker build --tag=dhsvm-matilija $CFGPREFIX/src
docker run \
  -v $CFGPREFIX/cfg:/matilija/cfg \
  -v $DATAPREFIX/input:/matilija/input \
  -v $DATAPREFIX/state:/matilija/state \
  -v $DATAPREFIX/output/${RUNID}:/matilija/output \
  -t dhsvm-matilija:latest \
  /matilija/src/dhsvm/build/DHSVM/sourcecode/DHSVM \
  /matilija/cfg/${RUNID}.cfg
