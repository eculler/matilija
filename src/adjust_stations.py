#!/usr/bin/env python3
"""
Pre-multiply station precipitation by a fixed factor and write adjusted copies.

Reads files named Station.<input_slug>.<N>.tsv from station_dir and writes
Station.adj.<multiplier>.<N>.tsv to the same directory.

Usage:
    python adjust_stations.py <station_dir> <input_slug> [multiplier]

Example:
    python adjust_stations.py input/station 20210625 1.87

Default multiplier is 1.87 (calibrated P_adj from prefire run).
"""
import os
import sys
import pandas as pd

COLUMNS = ['datetime', 'air_temp', 'wind_speed', 'relative_humidity',
           'incoming_shortwave', 'incoming_longwave', 'precipitation']

def adjust(station_dir, input_slug, multiplier):
    prefix = 'Station.{}.'.format(input_slug)
    station_files = sorted(
        f for f in os.listdir(station_dir)
        if f.startswith(prefix) and f.endswith('.tsv'))
    if not station_files:
        print('No files matching Station.{}.<N>.tsv in {}'.format(
            input_slug, station_dir))
        sys.exit(1)

    multiplier_str = '{:.2f}'.format(multiplier).replace('.', '-')

    for fname in station_files:
        station_num = fname[len(prefix):].removesuffix('.tsv')
        df = pd.read_csv(
            os.path.join(station_dir, fname),
            sep='\t',
            names=COLUMNS,
            index_col='datetime',
        )
        orig_precip = df.precipitation.copy()
        df.precipitation *= multiplier
        out_name = 'Station.adj.{}.{}.tsv'.format(multiplier_str, station_num)
        out_path = os.path.join(station_dir, out_name)
        df.to_csv(out_path, sep='\t', float_format='%.5f', header=False,
                  index_label='datetime')
        nonzero = orig_precip[orig_precip > 0]
        if not nonzero.empty:
            idx = nonzero.index[0]
            print('  {}: {:.5f} -> {:.5f} (ratio {:.4f})'.format(
                idx, nonzero[idx], df.precipitation[idx],
                df.precipitation[idx] / nonzero[idx]))
        print('Wrote {}'.format(out_path))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    station_dir = sys.argv[1]
    input_slug = sys.argv[2]
    multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 1.87
    adjust(station_dir, input_slug, multiplier)
