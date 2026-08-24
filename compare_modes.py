#!/usr/bin/env python3
"""
Compare the four first-layer initialization methods on the iterative-subtraction
attack:
  Trap-weights: (mode = trap-weights)
  mirrored       Boenisch-style trap weights, scale s
  independent    corrected trap weights, scale s

  Trap-Biases: (mode = soliton_free or soliton_data)
  soliton_free   LT-code, data-free: degrees ~ RobustSoliton(B), uses Trap-Biases, s=1 mirrored or independent
  soliton_data   LT-code same degrees, biases set using the server's own batch, s=1 mirrored or independent

  passive: non-mirrored, S=1
"""
from trapweights import (
    L2_DIST, SEED, DATABASES,
    load_data, build_model, build_problem,
    IterativeSubtractionAttack, score_attack,
    activation_stats, metric_row,
)
import tensorflow as tf
import numpy as np
import itertools

MODES = ('soliton_free', 'soliton_data','trap_weights',)
CHECK_PASSIVE = True
#trap-weights, zero bias (can set s)
#soliton_free, is data free (s=1) - random weights
#soliton_data users server's own batches to calibrate the bias (s=1) - random weights
MIRRORED = (False ,True)
BATCHES = (64, 128, 256,300,350,400,512, 1024,)
NUM_NEURONS = 1000              # width of the attacked layer
S = 0.95                 # only used by 'trap-weights' mode
SOLITON = (0.07, 0.4)     # Robust Soliton (c, delta)
DEFAULT_DATABASE = "cifar100"

def run_mode(mode,mirrored, xt, yt):
  if mode == 'soliton_data':
    x_b, y_b = load_data(DEFAULT_DATABASE, B=max(BATCHES),train=False)
  rows = {}
  for B in BATCHES:
    x_b = x_b[:B] if mode == 'soliton_data' else None
    
    if mode == 'trap_weights':
      model = build_model(*DATABASES[DEFAULT_DATABASE],n_neurons=NUM_NEURONS , mirrored=mirrored, mode=mode, s=S)
    elif mode == 'soliton_free':
      model = build_model(*DATABASES[DEFAULT_DATABASE],n_neurons=NUM_NEURONS , mirrored=mirrored, mode=mode, soliton=SOLITON, B=B)
    elif mode == 'passive':
      model = build_model(*DATABASES[DEFAULT_DATABASE],n_neurons=NUM_NEURONS , mode=mode)
    else:
      model = build_model(*DATABASES[DEFAULT_DATABASE],n_neurons=NUM_NEURONS , mirrored=mirrored, mode=mode, soliton=SOLITON,
                          calib_x=x_b)
    prob = build_problem(model, xt, yt, B)
    peel = IterativeSubtractionAttack(model, B).run(prob['gw'], prob['gb'])
    sc = score_attack(peel, prob)
    A = activation_stats(prob)[0]
    rows[B] = dict(peel=peel, sc=sc, A=A)
    print(f"    B={B:>4}  "
          f"peel R={sc['recall']:.3f}  iters={peel['iters']:>2}  A={A}  "
          f"G1={peel['G1']:<3}  exact={sc['B0']:<3}  lab_acc={sc['lab_acc']:.3f}",
          flush=True)
  return rows


def main():
  print("Loading " + DEFAULT_DATABASE + " ...")
  xt, yt = load_data(DEFAULT_DATABASE, B=max(BATCHES),train=True)
  print(f"N={NUM_NEURONS}  s={S}  soliton(c,delta)={SOLITON}  "
        f"L2<{L2_DIST}  seed={SEED}\n")

  results = {}
  for mode in MODES:
    for mirrored in MIRRORED:
      print(f"  mode={mode},Mirrored={mirrored}  BATCHES={BATCHES}")
      results[mode,mirrored] = run_mode(mode, mirrored, xt, yt)

  # ------------------------------------------------------------- summary
  if(CHECK_PASSIVE):
    print(f"  mode=passive,Mirrored={False}  BATCHES={BATCHES}")
    results['passive',False] = run_mode('passive',False,xt, yt)
  print("\n\n" + "=" * 84)
  print("extraction recall, iterative attack (certificate-admitted)")
  print("=" * 84)
  header = f"{'B':>5} |" + "|".join(f"{(mode+'_'+('mirrored' if mirrored else 'independent')):<25}" for mode, mirrored in itertools.product(MODES, MIRRORED))
  header = header + (f"{'|passive':<25}")
  print(header)
  print("-" * 84)
  for B in BATCHES:
    cells = [f"{results[m,mirrored][B]['sc']['recall']:>25.3f}" for m,mirrored in itertools.product(MODES, MIRRORED)]
    cells.append(f"{results['passive',False][B]['sc']['recall']:>25.3f}")
    print(f"{B:>5} |" + "|".join(cells))
  print("-" * 84)
if __name__ == "__main__":
  main()
