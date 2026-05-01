#!/usr/bin/env python3
"""
Pre-multiply station precipitation by a fixed factor and write adjusted copies.

Output filenames embed the multiplier: Station.adj.1-87.1.tsv for 1.87x.
The station number is taken from the last dot-separated field before .tsv
(e.g. Station.20210625.1.tsv -> station 1).

Usage:
    python adjust_stations.py <station_dir> <output_dir> [multiplier]

Default multiplier is 1.87 (calibrated P_adj from prefire run).
"""
import os
import sys
import pandas as pd

COLUMNS = ['datetime', 'air_temp', 'wind_speed', 'relative_humidity',
           'incoming_shortwave', 'incoming_longwave', 'precipitation']

def _output_name(fname, multiplier):
    stem = fname.rsplit('.tsv', 1)[0]
    station_num = stem.rsplit('.', 1)[-1]
    multiplier_str = '{:.2f}'.format(multiplier).replace('.', '-')
    return 'Station.adj.{}.{}.tsv'.format(multiplier_str, station_num)

def adjust(station_dir, output_dir, multiplier):
    os.makedirs(output_dir, exist_ok=True)
    station_files = sorted(
        f for f in os.listdir(station_dir) if f.startswith('Station'))
    for fname in station_files:
        df = pd.read_csv(
            os.path.join(station_dir, fname),
            sep='\t',
            names=COLUMNS,
            index_col='datetime',
        )
        orig_precip = df.precipitation.copy()
        df.precipitation *= multiplier
        out_name = _output_name(fname, multiplier)
        out_path = os.path.join(output_dir, out_name)
        df.to_csv(out_path, sep='\t', float_format='%.5f', header=False,
                  index_label='datetime')
        # Verification: show first non-zero precipitation pair
        nonzero = orig_precip[orig_precip > 0]
        if not nonzero.empty:
            idx = nonzero.index[0]
            print(f'  {idx}: {nonzero[idx]:.5f} -> {df.precipitation[idx]:.5f} '
                  f'(ratio {df.precipitation[idx]/nonzero[idx]:.4f})')
        print(f'Wrote {out_path}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    station_dir = sys.argv[1]
    output_dir = sys.argv[2]
    multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 1.87
    adjust(station_dir, output_dir, multiplier)
