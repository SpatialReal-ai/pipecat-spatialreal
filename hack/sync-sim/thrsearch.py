"""Offline threshold search on stage-A tick features (fixed first prior, every 50 ms tick).

usage: python3 thrsearch.py RESULTS.pkl

``RESULTS.pkl`` is the output of ``sim.py`` (each run carries its per-tick features under
``feat``). For each fingerprint this evaluates the 500-combination grid of the four thresholds and
prints the combinations with no false lock (an accepted tick more than 20 ms off), best accept
rate first, and what each criterion does alone.
"""

import itertools
import pickle
import sys

import numpy as np

with open(sys.argv[1], "rb") as fh:
    res = pickle.load(fh)
GRID = {
    "c": (0.7, 0.8, 0.85, 0.9, 0.95),
    "r": (1.0, 1.1, 1.2, 1.5),
    "s": (0.0, 0.01, 0.02, 0.03, 0.05),
    "tol": (np.inf, 0.02, 0.01, 0.005, 0.003),
}
for fp in ("F1", "F2", "F3"):
    F = np.concatenate([r["feat"][fp] for r in res])
    F = F[~np.isnan(F[:, 0])]
    pk, ratio, sharp, agree, est, est_i, edge, err, strad = F.T
    n = len(F)
    rows = []
    for c, r, s, tol in itertools.product(*GRID.values()):
        acc = (pk >= c) & (ratio >= r) & (sharp >= s) & (agree <= tol) & (edge == 0)
        fl = acc & (err > 0.020)
        rows.append(
            (
                c,
                r,
                s,
                tol,
                int(fl.sum()),
                acc.mean(),
                np.median(err[acc & (strad == 0)]) if acc.any() else np.nan,
            )
        )
    rows.sort(key=lambda x: (x[4], -x[5]))
    print(f"### {fp}: {n} speech ticks\n")
    print("| c_min | ratio | sharp | agree tol (ms) | false locks | accept rate | med err (ms) |")
    print("|---|---|---|---|---|---|---|")
    shown = 0
    for row in rows:
        if row[4] == 0 and shown < 8:
            print(
                f"| {row[0]} | {row[1]} | {row[2]} | {row[3] * 1000:.0f} | 0 | {row[5]:.3f} | {row[6] * 1000:.2f} |"
            )
            shown += 1
    # best per false-lock count when none is zero, and the effect of each criterion alone
    print("\nsingle-criterion view (others off):")
    for name, mask in [
        ("c>=0.9", pk >= 0.9),
        ("ratio>=1.5", ratio >= 1.5),
        ("sharp>=0.02", sharp >= 0.02),
        ("agree<=5ms", agree <= 0.005),
        ("c>=0.9 & agree<=5ms", (pk >= 0.9) & (agree <= 0.005)),
    ]:
        m = mask & (edge == 0)
        print(f"  {name:22s} accept {m.mean():.3f}  false {(m & (err > 0.02)).sum()}")
    best0 = [r for r in rows if r[4] == 0]
    print("\nleast-false rows:" if not best0 else "")
    if not best0:
        for row in rows[:5]:
            print("  ", row)
    print()
