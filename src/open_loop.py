#!/usr/bin/env python3
"""Single open-loop DHSVM run with fixed parameter values.

Fills the config template, trims station files to the run period, and runs
DHSVM once.  Assumes the working directory contains a template/ directory
(same layout expected by calibrate.py).

Usage:
    python open_loop.py <start> <end> <dhsvm> <station_slug> [param=value ...]

Example:
    python open_loop.py 2006-10-01 2021-06-25 /path/to/DHSVM adj.1-87 \\
        sandy_loam_inf=2e-5 loam_inf=1e-5 forest_lai=3.0 shrub_lai=1.5 \\
        forest_minres=500 shrub_minres=300
"""
import os
import shutil
import subprocess
import sys
import pandas as pd

COLUMNS = ['datetime', 'air_temp', 'wind_speed', 'relative_humidity',
           'incoming_shortwave', 'incoming_longwave', 'precipitation']
WORKDIR = 'open_loop_work'


def _find_state_dir(start):
    """Return the state subdirectory that matches start, or the base state dir."""
    base = os.path.join(WORKDIR, 'state')
    state_file = 'Channel.State.{:%m.%d.%Y.%H.%M.%S}'.format(start)
    if os.path.isfile(os.path.join(base, state_file)):
        return base
    dated = os.path.join(base, 'state.{:%Y.%m.%d.%H.%M.%S}'.format(start))
    if os.path.isfile(os.path.join(dated, state_file)):
        return dated
    return base


def _trim_station_files(slug, start, end):
    station_dir = os.path.join(WORKDIR, 'input', 'station')
    prefix = 'Station.{}.'.format(slug)
    for fname in sorted(f for f in os.listdir(station_dir) if f.startswith(prefix)):
        df = pd.read_csv(
            os.path.join(station_dir, fname),
            sep='\t', names=COLUMNS, index_col='datetime')
        df.index = pd.to_datetime(df.index, format='%m/%d/%Y-%H:%M')
        df = df[(df.index >= start) & (df.index <= end)]
        df.to_csv(
            os.path.join(station_dir, fname),
            sep='\t', float_format='%.5f', header=False,
            index_label='datetime', date_format='%m/%d/%Y-%H:%M')


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    start = pd.Timestamp(sys.argv[1])
    end = pd.Timestamp(sys.argv[2])
    dhsvm = sys.argv[3]
    station_slug = sys.argv[4]
    params = {k: float(v) for k, v in (kv.split('=', 1) for kv in sys.argv[5:])}

    if os.path.isdir(WORKDIR):
        shutil.rmtree(WORKDIR)
    shutil.copytree('template', WORKDIR, symlinks=True)

    _trim_station_files(station_slug, start, end)

    output_dir = os.path.join(WORKDIR, 'output')
    os.makedirs(output_dir, exist_ok=True)

    state_dir = _find_state_dir(start)
    output_rel = os.path.relpath(output_dir, WORKDIR)
    state_rel = os.path.relpath(state_dir, WORKDIR)

    with open(os.path.join(WORKDIR, 'dhsvm.cfg.template')) as f:
        cfg = f.read()

    cfg_text = cfg.format(
        start=start, end=end,
        output=output_rel + '/',
        state=state_rel + '/',
        station_slug=station_slug,
        **params)

    cfg_path = os.path.join(output_dir, 'dhsvm.cfg')
    with open(cfg_path, 'w') as f:
        f.write(cfg_text)

    cfg_rel = os.path.relpath(cfg_path, WORKDIR)
    sys.exit(subprocess.call([dhsvm, cfg_rel], cwd=WORKDIR))
