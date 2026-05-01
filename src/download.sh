#!/bin/bash
RUNID=20260501-soils-2wk
LOCAL=~/01-research/landslide-matilija/05-model-results/pbs/$RUNID
REMOTE=eculler@mahti.csc.fi:/scratch/project_2019087/matilija-soils-2wk

mkdir -p $LOCAL

# particles.csv
rsync -av $REMOTE/particles.csv $LOCAL/

# Streamflow.Only files only, preserving window/particle directory structure
rsync -av \
    --include='window*/' \
    --include='particle*/' \
    --include='Streamflow.Only' \
    --exclude='*' \
    $REMOTE/ $LOCAL/
