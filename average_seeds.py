#!/usr/bin/env python3
"""
average_seeds.py

For each database folder, average recall values across all seed CSVs.
Writes one averaged CSV per database into the `averages/` folder.
"""
import os
import glob
import pandas as pd

def take_averate(filename,dataset,input_dir,output_dir = 'averages'):
    pattern = os.path.join(input_dir, f"{dataset}_seed*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Skipping {input_dir}: no CSV files matching {pattern}")
        return

    # Load every seed CSV and stack them
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    # Mean across seeds for each batch size
    avg = combined.groupby('B', sort=True).mean().reset_index()

    # Round to 3 decimals
    numeric_cols = avg.select_dtypes(include='number').columns
    avg[numeric_cols] = avg[numeric_cols].round(3)

    out_path = os.path.join(output_dir, f"{filename}_averaged.csv")
    avg.to_csv(out_path, index=False)
    print(f"[{dataset}] Averaged {len(files)} seeds -> {out_path}")

def main():
    db_dirs = ["mnist", "emnist", "svhn", "harus" , "fashion_mnist", "cifar10", "cifar100","imagenet"]
    if not db_dirs:
        print("No database folders found in current directory.")
        return

    out_dir = 'averages'
    os.makedirs(out_dir, exist_ok=True)
    for db in sorted(db_dirs):
        input_dir = os.path.join('results_final',db)
        outdir = os.path.join(out_dir, db)
        os.makedirs(outdir, exist_ok=True)

        for r_dir in os.listdir(input_dir):
            in_dir = os.path.join(input_dir,r_dir)
            take_averate(r_dir,db,in_dir,outdir)

if __name__ == '__main__':
    main()