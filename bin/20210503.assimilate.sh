#/bin/bash
python3 ../src/pbsmoother.py 2 1D \
  ~/GoogleDrive/research/landslide.matilija.sync \
  matilija.discharge.20200408.tsv \
  Station.20190514.*.tsv \
  state.2002.10.01.00.00.00 \
  ~/GoogleDrive/research/landslide.matilija.sync/pbs/20210520
