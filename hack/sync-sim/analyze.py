"""Summarise a ``sim.py`` result pickle into the markdown tables of RESULTS.md.

usage: python3 analyze.py RESULTS.pkl THR_F1 THR_F2 THR_F3

Each THR is ``c,r,s,tol`` (peak, ratio, sharpness, half-window tolerance in seconds) exactly as it
was passed to ``sim.py --thr``; they pick which replay of the pickle is tabulated per fingerprint.
"""

import pickle
import sys
from collections import defaultdict

import numpy as np

with open(sys.argv[1], "rb") as fh:
    res = pickle.load(fh)
FPN = ["F1", "F2", "F3"]
CONDS = list(dict.fromkeys(r["cond"] for r in res))
best = {fp: tuple(float(v) for v in spec.split(",")) for fp, spec in zip(FPN, sys.argv[2:5])}
print(
    "thresholds (c_min, peak/second ratio, sharpness, half-window agreement tol in s):", best, "\n"
)


def ms(v):
    return f"{v * 1000:.1f}"


def pct(a, q):
    return np.percentile(a, q) if len(a) else np.nan


def table(kind_filter, title):
    print(f"## {title}\n")
    print(
        "| fp | kbps | condition | runs | err med | err p95 | err max | no-interp med / p95 | "
        "false-lock | hold | applied err p95 | applied >20ms |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fp in FPN:
        for kb in (32, 24):
            for c in CONDS:
                rs = [r for r in res if r["kbps"] == kb and r["cond"] == c and kind_filter(r)]
                if not rs:
                    continue
                v = [r["res"][(fp, best[fp], "200ms")] for r in rs]
                ei = np.concatenate([x["acc_err_i"] for x in v])
                e0 = np.concatenate([x["acc_err"] for x in v])
                ap = np.concatenate([x["app_err"] for x in v])
                nacc = sum(x["n_acc"] for x in v)
                nf = sum(x["n_false"] for x in v)
                ng = sum(x["n_gate"] for x in v)
                nh = sum(x["n_hold"] for x in v)
                print(
                    f"| {fp} | {kb} | {c} | {len(rs)} | {ms(pct(ei, 50))} | {ms(pct(ei, 95))} | "
                    f"{ms(ei.max() if len(ei) else np.nan)} | {ms(pct(e0, 50))} / {ms(pct(e0, 95))} | "
                    f"{nf}/{nacc} | {nh / max(ng, 1):.2f} | {ms(pct(ap, 95))} | {np.mean(ap > 0.02) * 100:.1f}% |"
                )
    print()


def summary():
    print("## Summary per fingerprint (all conditions, both bitrates)\n")
    print(
        "| fp | ticks | source | accepted | false locks (>20 ms) | hold rate | err med | p95 | p99 | max | "
        "no-interp med / p95 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for fp in FPN:
        for step in ("200ms", "50ms"):
            for kind in ("tts", "real"):
                v = [r["res"][(fp, best[fp], step)] for r in res if r["kind"] == kind]
                ei = np.concatenate([x["acc_err_i"] for x in v])
                e0 = np.concatenate([x["acc_err"] for x in v])
                print(
                    f"| {fp} | {step} | {kind} | {sum(x['n_acc'] for x in v)} | {sum(x['n_false'] for x in v)} | "
                    f"{sum(x['n_hold'] for x in v) / sum(x['n_gate'] for x in v):.2f} | {ms(pct(ei, 50))} | "
                    f"{ms(pct(ei, 95))} | {ms(pct(ei, 99))} | {ms(ei.max())} | {ms(pct(e0, 50))} / {ms(pct(e0, 95))} |"
                )
    print()


summary()
table(
    lambda r: r["kind"] == "tts",
    "Alignment error, TTS references (ms; 200 ms rule, parabolic interpolation)",
)
table(
    lambda r: r["kind"] == "real", "Alignment error, real captures pushed through Opus again (ms)"
)

# ------------------------------------------------ first lock
print("## Time to first lock (ms of turn audio after the true first onset)\n")
print("| fp | ticks | source | median | p90 | max | never locked |")
print("|---|---|---|---|---|---|---|")
for fp in FPN:
    for step in ("50ms", "200ms"):
        for kind in ("tts", "real"):
            fl = [r["res"][(fp, best[fp], step)]["first_lock"] for r in res if r["kind"] == kind]
            got = np.array([f for f in fl if f is not None])
            print(
                f"| {fp} | {step} | {kind} | {ms(pct(got, 50))} | {ms(pct(got, 90))} | {ms(got.max())} | "
                f"{sum(f is None for f in fl)}/{len(fl)} |"
            )
print()
print("### first lock by condition (50 ms ticks, all sources)\n")
print("| fp | condition | median | p90 | max | never |")
print("|---|---|---|---|---|---|")
for fp in FPN:
    for c in CONDS:
        fl = [r["res"][(fp, best[fp], "50ms")]["first_lock"] for r in res if r["cond"] == c]
        got = np.array([f for f in fl if f is not None])
        print(
            f"| {fp} | {c} | {ms(pct(got, 50))} | {ms(pct(got, 90))} | {ms(got.max())} | {sum(f is None for f in fl)}/{len(fl)} |"
        )
print()

# ------------------------------------------------ relock
print(
    "## Re-lock after a gap (time from the gap until the applied offset is within 5 ms of the new offset)\n"
)
print(
    "| fp | ticks | gap | runs | relocked | median | p90 | max | stale error while held (= gap length) |"
)
print("|---|---|---|---|---|---|---|---|---|")
for fp in FPN:
    for step in ("200ms", "50ms"):
        for g in ("gap-del", "gap-ins", "all"):
            rs = [r for r in res if r["cond"] == g and r["ev"].get("gap_t") is not None]
            rl = [r["res"][(fp, best[fp], step)]["relock"] for r in rs]
            got = np.array([x for x in rl if x is not None])
            gl = np.array([r["ev"]["gap_len"] for r in rs])
            print(
                f"| {fp} | {step} | {g} | {len(rs)} | {len(got)} | {ms(pct(got, 50))} | {ms(pct(got, 90))} | "
                f"{ms(got.max() if len(got) else np.nan)} | {ms(gl.min())}-{ms(gl.max())} |"
            )
print()

# ------------------------------------------------ F4
print("## F4 - today's onset rule replayed on the same X\n")
print(
    "First pair = first received onset vs the server's first onset; every later server onset is matched to the "
    "first received onset in [expected-60, expected+200] ms and re-aligns the turn.\n"
)
print(
    "| source | kbps | condition | turns | no onset at all | onsets | unmatched | first-onset med / p95 / max | "
    "all-onset med / p95 / max | onsets >20 ms | turns with any onset >20 ms | >40 ms |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
tot = defaultdict(int)
for kind in ("tts", "real"):
    for kb in (32, 24):
        for c in CONDS:
            rs = [r for r in res if r["kbps"] == kb and r["cond"] == c and r["kind"] == kind]
            none = sum(r["f4_all"] is None for r in rs)
            first = np.abs(np.array([r["f4"] for r in rs if r["f4"] is not None]))
            allv = [e for r in rs if r["f4_all"] for e in r["f4_all"]]
            unm = sum(e is None for e in allv)
            ea = np.abs(np.array([e for e in allv if e is not None]))
            worst = [max(abs(e) for e in r["f4_all"] if e is not None) for r in rs if r["f4_all"]]
            t20 = sum(w > 0.02 for w in worst)
            t40 = sum(w > 0.04 for w in worst)
            if c in ("clean", "drift", "gap-del", "gap-ins", "gap-plc"):
                tot[(kind, "turns")] += len(rs) - none
                tot[(kind, "t20")] += t20
                tot[(kind, "t40")] += t40
            if not len(ea):
                print(f"| {kind} | {kb} | {c} | {len(rs)} | {none} | 0 | - | - | - | - | - | - |")
                continue
            print(
                f"| {kind} | {kb} | {c} | {len(rs)} | {none} | {len(allv)} | {unm} | "
                f"{ms(np.median(first))} / {ms(pct(first, 95))} / {ms(first.max())} | "
                f"{ms(np.median(ea))} / {ms(pct(ea, 95))} / {ms(ea.max())} | {np.sum(ea > 0.02)} | {t20} | {t40} |"
            )
print()
for kind in ("tts", "real"):
    print(
        f"- {kind}, noise-free conditions (clean/drift/gaps): {tot[(kind, 't20')]}/{tot[(kind, 'turns')]} "
        f"turns with an onset >20 ms off, {tot[(kind, 't40')]} >40 ms"
    )
print()
