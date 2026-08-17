#!/usr/bin/env python3
"""
Compare the four first-layer initialization methods on the iterative-subtraction
attack:
  Trap-weights: (mode = None)
  mirrored       Boenisch-style trap weights, scale s
  independent    corrected trap weights, scale s

  Trap-Biases: (mode = soliton_free or soliton_data)
  soliton_free   LT-code, data-free: degrees ~ RobustSoliton(B), uses Trap-Biases, s=1 mirrored or independent
  soliton_data   LT-code same degrees, biases set using the server's own batch, s=1 mirrored or independent

"""
from trapweights import (
    NUM_NEURONS, L2_DIST, SEED,
    load_cifar10, build_model, build_problem,
    IterativeSubtractionAttack, attack_baseline, score_attack,
    activation_stats, metric_row,
)
import tensorflow as tf
import numpy as np
import itertools

MODES = ('soliton_free', 'soliton_data','None') 
#none, zero bias (can set s)
#soliton_free, is data free (s=1) - random weights
#soliton_data users server's own batches to calibrate the bias (s=1) - random weights
MIRRORED = (False ,True)
BATCHES = (64, 128, 256,300,350,400,512, 1024)
S = 0.95                  # only used by 'Node' mode
SOLITON = (0.05, 0.1)     # Robust Soliton (c, delta)
FEATURES = 32 * 32 * 3


def run_mode(mode,mirrored, xt, yt):
  if mode == 'soliton_data':
    _, (x_b,y_b) = tf.keras.datasets.cifar10.load_data() #used in soliton_data mode only
    x_b, y_b = x_b.astype(np.float64) / 255.0, y_b.flatten().astype(int)
  rows = {}
  for B in BATCHES:
    x_b = x_b[:B] if mode == 'soliton_data' else None
    kw = dict(num_classes=10, n_neurons=NUM_NEURONS, seed=SEED)
    if mode == 'None':
      model = build_model(FEATURES, mirrored=mirrored, mode=mode, s=S, **kw)
    elif mode == 'soliton_free':
      model = build_model(FEATURES, mirrored=mirrored, mode=mode, soliton=SOLITON, B=B, **kw)
    else:
      model = build_model(FEATURES, mirrored=mirrored, mode=mode, soliton=SOLITON,
                          calib_x=x_b, **kw)
    prob = build_problem(model, xt, yt, B)
    base = attack_baseline(prob)
    peel = IterativeSubtractionAttack(model, B).run(prob['gw'], prob['gb'])
    sc = score_attack(peel, prob)
    A = activation_stats(prob)[0]
    rows[B] = dict(base=base, peel=peel, sc=sc, A=A)
    print(f"    B={B:>4}  base R={base['recall']:.3f}  "
          f"peel R={sc['recall']:.3f}  iters={peel['iters']:>2}  A={A}  "
          f"G1={peel['G1']}  exact={sc['B0']}  lab_acc={sc['lab_acc']:.3f}",
          flush=True)
  return rows


def main():
  print("Loading CIFAR-10 ...")
  xt, yt = load_cifar10()
  print(f"N={NUM_NEURONS}  s={S}  soliton(c,delta)={SOLITON}  "
        f"L2<{L2_DIST}  seed={SEED}\n")

  results = {}
  for mode in MODES:
    for mirrored in MIRRORED:
      print(f"  mode={mode},Mirrored={mirrored}  BATCHES={BATCHES}")
      results[mode,mirrored] = run_mode(mode, mirrored, xt, yt)

  # ------------------------------------------------------------- summary
  for mode in MODES:
    for mirrored in MIRRORED:
      print(f"\n\nmode={mode} mirrored={mirrored}")
      print("-" * 84)
      print(f"{'B':>5} | {'base R':>12} {'peel R':>12} "
            f"{'gain':>13} {'iters':>6} {'A':>5} {'G1':>5} "
            f"{'B0':>5} {'lab_acc':>8}")
      print("-" * 84)
      for B in BATCHES:
        r = results[mode,mirrored][B]
        fac = r['sc']['recall'] / r['base']['recall'] if r['base']['recall'] else float('inf')
        print(f"{B:>5} | {r['base']['recall']:>12.3f} {r['sc']['recall']:>12.3f} "
              f"{r['sc']['recall']-r['base']['recall']:>+13.3f} {r['peel']['iters']:>6} "
              f"{r['A']:>5} {r['peel']['G1']:>5} {r['sc']['B0']:>5} "
              f"{r['sc']['lab_acc']:>8.3f}")
      print("-" * 84)
  print("\n\n" + "=" * 84)
  print("extraction recall, iterative attack (certificate-admitted)")
  print("=" * 84)
  print(f"{'B':>5} |" + "|".join(f"{(mode+'_'+('mirrored' if mirrored else 'independent')):<25}" for mode, mirrored in itertools.product(MODES, MIRRORED)))
  print("-" * 84)
  for B in BATCHES:
    cells = [f"{results[m,mirrored][B]['base']['recall']:>25.3f}" for m,mirrored in itertools.product(MODES, MIRRORED)]
    print(f"{B:>5} |" + "|".join(cells))
  print("-" * 84)

  print("\n" + "=" * 84)
  print("baseline (single-pass) recall")
  print("=" * 84)
  print(f"{'B':>5} |" + "|".join(f"{(mode+'_'+('mirrored' if mirrored else 'independent')):<25}" for mode, mirrored in itertools.product(MODES, MIRRORED)))
  print("-" * 84)
  for B in BATCHES:
    cells = [f"{results[m,mirrored][B]['base']['recall']:>25.3f}" for m,mirrored in itertools.product(MODES, MIRRORED)]
    print(f"{B:>5} |" + "|".join(cells))
  print("-" * 84)

  print("\n" + "=" * 84)
  print("precision metrics of the iterative attack, P = Y * rho")
  print("=" * 84)
  for mode in MODES:
    for mirrored in MIRRORED:
      print(f"  {mode} (mirrored={mirrored})")
      print(f"  {'B':>5} {'A':>5} {'G1':>5} {'B0':>5} {'P':>6} {'Y':>6} "
            f"{'rho':>5} {'iters':>6}")
      for B in BATCHES:
        r = results[mode,mirrored][B]
        m = metric_row(r['A'], r['peel']['G1'], r['sc']['B0'], B)
        print(f"  {B:>5} {r['A']:>5} {r['peel']['G1']:>5} {r['sc']['B0']:>5} "
              f"{m['P']:>6.3f} {m['Y']:>6.3f} {m['rho']:>5.2f} "
              f"{r['peel']['iters']:>6}")
      print("-" * 84)

  print("\n" + "=" * 84)
  print("per-iteration traces (iterative attack)")
  print("=" * 84)
  for mode in MODES:
    for mirrored in MIRRORED:
      for B in BATCHES:
        t = results[mode,mirrored][B]['peel']['trace']
        print(f"  {mode}_{'mirrored' if mirrored else 'independent':<13} B={B:>4}: "
              + ", ".join(f"it{e['iter']}:+{e['new']}" for e in t))
  print("=" * 84)


if __name__ == "__main__":
  main()
