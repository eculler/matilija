#/bin/bash
CFGFILE=$1

docker build ~/Documents/lsmutils -t eculler/land-surface-modeling-utilities
docker run --rm -it\
  --security-opt seccomp=unconfined \
  -v /Volumes/WD-Data/data/complete/SRTMv3:/lsmutils/data/SRTMv3 \
  -v /Volumes/WD-Data/data/by.project/matilija:/lsmutils/data \
  -v /Volumes/WD-Data/models/DHSVM/DHSVM3.2-ubuntu/DHSVM/program:/lsmutils/scripts \
  -v ~/Documents/lsmutils:/lsmutils/src/lsmutils \
  -v ~/Documents/research/landslide.matilija.git/setup/cfg:/lsmutils/cfg \
  -v ~/GoogleDrive/research/landslide.matilija.sync/setup/tmp:/lsmutils/dhsvm \
  -t eculler/land-surface-modeling-utilities \
  /bin/bash -c \
  "conda run -n lsmutils \
   pip install --ignore-installed -e /lsmutils/src/lsmutils && \
   conda run -n lsmutils \
   python -m lsmutils /lsmutils/cfg/'$CFGFILE'.cfg.yaml"
