#!/usr/bin/env python3
"""
Main results: extraction-recall, label accuracy, certification margins.
"""
from trapweights import (
    L2_DIST, SEED, DATABASES,
    load_data, build_model, build_problem,
    IterativeSubtractionAttack, attack_baseline, score_attack,
    activation_stats, metric_row,
)

BATCHES = (20, 64, 128, 256, 512, 1024)
S = 0.95
NUM_NEURONS = 4000              # width of the attacked layer
DEFAULT_DATABASE = "mnist"

def main():
  print("Loading " + DEFAULT_DATABASE + " ...")
  xt, yt = load_data(DEFAULT_DATABASE, B=max(BATCHES),train=True)
  print(f"N={NUM_NEURONS}  s={S}  threshold L2<{L2_DIST}  seed={SEED}\n")

  model = build_model(*DATABASES[DEFAULT_DATABASE], n_neurons=NUM_NEURONS, soliton=(0.05, 0.1), s=S,seed=SEED)

  res = {}
  for B in BATCHES:
    prob = build_problem(model, xt, yt, B)
    base = attack_baseline(prob)
    peel = IterativeSubtractionAttack(model, B).run(prob['gw'], prob['gb'])
    sc = score_attack(peel, prob)
    A = activation_stats(prob)[0]
    res[B] = (base, peel, sc, A)
    print(f"  B={B:>5}  base R={base['recall']:.3f}  peel R={sc['recall']:.3f}"
          f"  iters={peel['iters']}  A={A}  G1={base['G1']}/{peel['G1']}"
          f"  lab_acc={sc['lab_acc']:.3f}"
          f"  accepted={len(peel['samples'])} exact={sc['B0']}")

  # ------------------------------------------------------------- Table 1
  print("\n\n" + "=" * 72)
  print("TABLE 1  extraction-recall (identification by certificate, not oracle)")
  print("=" * 72)
  print(f"{'B':>5} {'baseline':>9} {'iterative':>10} {'gain':>9} "
        f"{'factor':>8} {'iters':>6}")
  print("-" * 72)
  for B in BATCHES:
    b, p, sc, A = res[B]
    fac = sc['recall'] / b['recall'] if b['recall'] else float('inf')
    print(f"{B:>5} {b['recall']:>9.3f} {sc['recall']:>10.3f} "
          f"{sc['recall']-b['recall']:>+9.3f} {fac:>7.1f}x {p['iters']:>6}")
  print("=" * 72)

  # ------------------------------------------------------------- Table 2
  print("\n" + "=" * 72)
  print("TABLE 2  label accuracy of the autodiff residual-bias fit (Eq. 10)")
  print("=" * 72)
  print(f"{'B':>5} {'recovered':>10} {'label acc':>10}")
  print("-" * 72)
  for B in BATCHES:
    sc = res[B][2]
    print(f"{B:>5} {sc['B0']:>10} {sc['lab_acc']:>10.3f}")
  print("=" * 72)

  # ------------------------------------------------------------- Table 3
  print("\n" + "=" * 72)
  print("TABLE 3  extraction-precision, P = Y * rho")
  print("=" * 72)
  print(f"{'B':>5} {'A':>5} | {'G1':>5} {'B0':>5} {'P':>6} {'Y':>6} {'rho':>5}"
        f" | {'G1':>5} {'B0':>5} {'P':>6} {'Y':>6} {'rho':>5}")
  print(f"{'':>5} {'':>5} | {'baseline':^30} | {'iterative':^30}")
  print("-" * 72)
  for B in BATCHES:
    b, p, sc, A = res[B]
    mb = metric_row(A, b['G1'], b['B0'], B)
    mp = metric_row(A, p['G1'], sc['B0'], B)
    print(f"{B:>5} {A:>5} | {b['G1']:>5} {b['B0']:>5} {mb['P']:>6.3f} "
          f"{mb['Y']:>6.3f} {mb['rho']:>5.2f} | {p['G1']:>5} {sc['B0']:>5} "
          f"{mp['P']:>6.3f} {mp['Y']:>6.3f} {mp['rho']:>5.2f}")
  print("=" * 72)

  # -------------------------------------------------- certification margins
  print("\n" + "=" * 72)
  print("certificate residuals (Eq. 11) of accepted samples")
  print("=" * 72)
  for B in BATCHES:
    sc = res[B][2]
    if sc['genuine_eps']:
      import numpy as np
      e = np.asarray(sc['genuine_eps'])
      print(f"  B={B:>5}  median eps={np.median(e):.2e}  max={e.max():.2e}")
  print("=" * 72)

  # ------------------------------------------------------ iteration traces
  print("\n" + "=" * 72)
  print("per-iteration traces")
  print("=" * 72)
  for B in BATCHES:
    t = res[B][1]['trace']
    print(f"  B={B:>5}: " + ", ".join(f"it{e['iter']}:+{e['new']}" for e in t))
  print("=" * 72)


if __name__ == "__main__":
  main()
