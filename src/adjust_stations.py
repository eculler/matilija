#!/usr/bin/env python3
"""
Pre-multiply station precipitation by a fixed factor and write adjusted copies.

Usage:
    python adjust_stations.py <station_dir> <output_dir> [multiplier]

Default multiplier is 1.87 (calibrated P_adj from prefire run).
"""
import os
import sys
import pandas as pd

COLUMNS = ['datetime', 'air_temp', 'wind_speed', 'relative_humidity',
           'incoming_shortwave', 'incoming_longwave', 'precipitation']

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
        df.precipitation *= multiplier
        out_path = os.path.join(output_dir, fname)
        df.to_csv(out_path, sep='\t', float_format='%.5f', header=False,
                  index_label='datetime')
        print(f'Wrote {out_path}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    station_dir = sys.argv[1]
    output_dir = sys.argv[2]
    multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 1.87
    adjust(station_dir, output_dir, multiplier)
