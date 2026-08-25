#!/usr/bin/env python3
"""
average_seeds.py

For each database folder, average recall values across all seed CSVs.
Writes one averaged CSV per database into the `averages/` folder.
"""
import os
import glob
import pandas as pd

def main():
    db_dirs = ["mnist", "emnist", "svhn", "harus" , "fashion_mnist", "cifar10", "cifar100"]

    if not db_dirs:
        print("No database folders found in current directory.")
        return

    os.makedirs('averages', exist_ok=True)

    for db in sorted(db_dirs):
        pattern = os.path.join(db, f"{db}_seed*.csv")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"Skipping {db}: no CSV files matching {pattern}")
            continue

        # Load every seed CSV and stack them
        dfs = [pd.read_csv(f) for f in files]
        combined = pd.concat(dfs, ignore_index=True)

        # Mean across seeds for each batch size
        avg = combined.groupby('B', sort=True).mean().reset_index()

        # Round to 3 decimals
        numeric_cols = avg.select_dtypes(include='number').columns
        avg[numeric_cols] = avg[numeric_cols].round(3)

        out_path = os.path.join('averages', f"{db}_averaged.csv")
        avg.to_csv(out_path, index=False)
        print(f"[{db}] Averaged {len(files)} seeds -> {out_path}")

if __name__ == '__main__':
    main()