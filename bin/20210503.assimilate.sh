#/bin/bash
python3 ../src/pbsmoother.py 2 1MS \
  ~/GoogleDrive/research/landslide.matilija.sync \
  matilija.discharge.20200408.tsv \
  Station.20190514.*.tsv \
  ~/GoogleDrive/research/landslide.matilija.sync/pbs/20210520
