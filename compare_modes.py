#!/usr/bin/env python3
"""
Compare the four first-layer initialization methods on the iterative-subtraction
attack across multiple databases and seeds.
"""
from peeling_attack import (
    L2_DIST, DATABASES,
    load_data, build_model, build_problem,
    IterativeSubtractionAttack, score_attack,
    activation_stats,
)
import tensorflow as tf
import numpy as np
import itertools
import csv
import os

# --------------------------------------------------------------------- config
MODES = ('soliton_free', 'soliton_data', 'trap_weights')
CHECK_PASSIVE = True
MIRRORED = (False, True)
BATCHES = (64, 128, 256, 300, 400, 512, 600, 700, 800, 900, 1024)
NUM_NEURONS = 1000

# tuple of experiments to run
DATABASES_LIST = ("mnist", "emnist", "svhn", "harus" , "fashion_mnist", "cifar10", "cifar100",
 #"imagenet",
 )
SEEDS = (55726, 42523, 93687, 32443, 56581)

S = 0.99
SOLITON = (0.05, 0.4)

# --------------------------------------------------------------------- helpers
def make_filename(db, seed, soliton, s):
  c, delta = soliton
  return (
    f"{db}_"
    f"seed{seed}_"
    f"soliton{c}_{delta}_"
    f"S{s}.csv"
  )


def run_mode(mode, mirrored, xt, yt, B, seed,db,x_calib=None):
  """Run a single (mode, mirrored) configuration for one batch size."""
  if mode == 'trap_weights':
    model = build_model(
      *DATABASES[db], n_neurons=NUM_NEURONS,
      mirrored=mirrored, mode=mode, s=S, seed=seed
    )
  elif mode == 'soliton_free':
    model = build_model(
      *DATABASES[db], n_neurons=NUM_NEURONS,
      mirrored=mirrored, mode=mode, soliton=SOLITON, B=B, seed=seed
    )
  elif mode == 'passive':
    model = build_model(
      *DATABASES[db], n_neurons=NUM_NEURONS,
      mode=mode, seed=seed
    )
  else:  # soliton_data
    model = build_model(
      *DATABASES[db], n_neurons=NUM_NEURONS,
      mirrored=mirrored, mode=mode, soliton=SOLITON,
      calib_x=x_calib, seed=seed
    )
  cert_tol = 1e-8
  dedup_tol = 1e-6
  if(db == "imagenet"):
    cert_tol = 1e-9
    dedup_tol = 1e-5
  prob = build_problem(model, xt, yt, B)
  peel = IterativeSubtractionAttack(model, B,dedup_tol=dedup_tol,cert_tol=cert_tol).run(prob['gw'], prob['gb'])
  sc = score_attack(peel, prob)
  A = activation_stats(prob)[0]

  print(
    f"    B={B:>4}  "
    f"peel R={sc['recall']:.3f}  iters={peel['iters']:>2}  A={A}  "
    f"G1={peel['G1']:<3}  exact={sc['B0']:<3}  lab_acc={sc['lab_acc']:.3f}",
    flush=True
  )
  return {
      'sc': {
          'recall': sc['recall'],
          'lab_acc': sc['lab_acc'],
          'B0': sc['B0'],
      },
      'peel': {
          'iters': peel['iters'],
          'G1': peel['G1'],
      },
      'A': A,
  }


def run_all_batches(db, seed, xt, yt):
  """Run every mode/mirrored combination across all batch sizes for one db/seed."""
  rows = {}
  # Pre-load calibration data once if needed
  x_calib, _ = load_data(db, B=max(BATCHES), train=False,seed=seed) \
    if 'soliton_data' in MODES else (None, None)

  for mode in MODES:
    for mirrored in MIRRORED:
      print(f"  [{db}] seed={seed}  mode={mode}  mirrored={mirrored}")
      rows[mode, mirrored] = {}
      for B in BATCHES:
        x_b = x_calib[:B] if mode == 'soliton_data' else None
        rows[mode, mirrored][B] = run_mode(
          mode, mirrored, xt, yt, B, seed, x_calib=x_b,db=db
        )

  if CHECK_PASSIVE:
    print(f"  [{db}] seed={seed}  mode=passive  mirrored=False")
    rows['passive', False] = {}
    for B in BATCHES:
      rows['passive', False][B] = run_mode(
        'passive', False, xt, yt, B, seed,db=db
      )

  return rows


def write_csv(db, seed, results):
  """Write the recall table for one (db, seed) into the database folder."""
  os.makedirs(db, exist_ok=True)

  filepath = os.path.join(db, make_filename(db, seed, SOLITON, S))

  # header
  header = ["B"]
  for mode, mirrored in itertools.product(MODES, MIRRORED):
    label = f"{mode}_{'mirrored' if mirrored else 'independent'}"
    header.append(label)
  if CHECK_PASSIVE:
    header.append("passive")

  # rows
  csv_rows = []
  for B in BATCHES:
    row = [B]
    for m, mirrored in itertools.product(MODES, MIRRORED):
      row.append(results[m, mirrored][B]['sc']['recall'])
    if CHECK_PASSIVE:
      row.append(results['passive', False][B]['sc']['recall'])
    csv_rows.append(row)

  with open(filepath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_rows)

  print(f"\n[Saved CSV: {filepath}]")
  return filepath


def print_console_table(db, seed, results):
  """Pretty-print the recall table to stdout."""
  print("\n\n" + "=" * 84)
  print(f"[{db}]  seed={seed}  — extraction recall, iterative attack")
  print("=" * 84)

  header = f"{'B':>5} |" + "|".join(
    f"{(mode + '_' + ('mirrored' if mirrored else 'independent')):<25}"
    for mode, mirrored in itertools.product(MODES, MIRRORED)
  )
  if CHECK_PASSIVE:
    header += f"{'|passive':<25}"
  print(header)
  print("-" * 84)

  for B in BATCHES:
    cells = [
      f"{results[m, mirrored][B]['sc']['recall']:>25.3f}"
      for m, mirrored in itertools.product(MODES, MIRRORED)
    ]
    if CHECK_PASSIVE:
      cells.append(
        f"{results['passive', False][B]['sc']['recall']:>25.3f}"
      )
    print(f"{B:>5} |" + "|".join(cells))
  print("-" * 84)


# --------------------------------------------------------------------- main
def main():
  for db in DATABASES_LIST:
    print(f"\n{'='*60}")
    print(f"DATABASE: {db}")
    print(f"{'='*60}")

    # load training data once per database (max batch size needed)
    print(f"Loading {db} ...")

    for seed in SEEDS:
      xt, yt = load_data(db, B=max(BATCHES), train=True,seed=seed)
      print(f"\n--- seed = {seed} ---")
      results = run_all_batches(db, seed, xt, yt)
      write_csv(db, seed, results)
      print_console_table(db, seed, results)


if __name__ == "__main__":
  main()