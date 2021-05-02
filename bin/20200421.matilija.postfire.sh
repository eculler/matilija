#/bin/bash
RUNID=20200421.matilija.postfire

cd ~/GoogleDrive/research/matilija/dhsvm/docker && \
  docker build --tag=dhsvm-matilija .
docker run --rm -it \
  -v ~/GoogleDrive/research/matilija/dhsvm/cfg:/matilija/cfg \
  -v ~/GoogleDrive/research/matilija/dhsvm/input:/matilija/input \
  -v ~/GoogleDrive/research/matilija/dhsvm/state:/matilija/state \
  -v ~/GoogleDrive/research/matilija/dhsvm/output/${RUNID}:/matilija/output \
  -t dhsvm-matilija:latest \
  /matilija/DHSVM3.2/build/DHSVM/sourcecode/DHSVM /matilija/cfg/${RUNID}.cfg
