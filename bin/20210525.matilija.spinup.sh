#/bin/bash
RUNID=20210525.matilija.spinup

docker run --rm -it \
  -v ~/Documents/research/landslide.matilija.git/cfg:/matilija/cfg \
  -v ~/GoogleDrive/research/landslide.matilija.sync/input:/matilija/input \
  -v ~/GoogleDrive/research/landslide.matilija.sync/state:/matilija/state \
  -v ~/GoogleDrive/research/landslide.matilija.sync/output/${RUNID}:/matilija/output \
  -t dhsvm-matilija:latest \
  /matilija/src/dhsvm/build/DHSVM/sourcecode/DHSVM \
  /matilija/cfg/${RUNID}.cfg
