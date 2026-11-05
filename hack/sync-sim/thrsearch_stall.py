"""Threshold search for the stall rule's constants (``dsp.StallParams``).

usage: python thrsearch_stall.py [--captures DIR ...] [--cache FILE] [--jobs N] > thrsearch_stall.out

Data: the stall simulation (``sim.py --stall``: real bot turns of the committed fixtures under
clean, stall-80/150/250/500/1200, zeros-150/250/500, loss-200/400, ins, del; exact truth) and real
captures: the committed fixtures (``tests/fixtures/sync/*``) plus every ``--captures`` directory
(the stalled Chrome captures are not committed), scored on the dense local truth of
``replay_capture.py --gaps`` (300 ms windows every 25 ms, +/- 2 s; the truth's centre is today's
first lock, so every candidate is scored on the same truth).

Per run (simulation) and per turn (captures) the cost is the received time after the turn's first
lock during which the face's position is more than 20 ms off the truth, plus 2000 ms when the turn
reached sync-lost (the face idles for at least that long). The objective ``J`` is the mean cost
per simulated run plus the mean cost per real turn. A candidate is feasible only with zero false
accepts: no accept (full window, newer half or confirm) more than 20 ms off the truth of the frames
its window covered, outside a gap, and none across a gap that lands on no real position of its
window (simulation and captures).

The search: today's rule, the spike's constants (with the confirm's ratio check, the carried bracket and the digital-silence guard, added here), the
ablations, then coordinate descent from the spike's point over each constant's grid (two rounds; a
change is kept when it lowers ``J`` by at least 1 % and stays feasible). ``worst`` is the largest
error of any accept outside a gap: the margin left to the 20 ms false-lock line. The burst-loss risk is
the loss conditions' share of the cost (``loss`` column): there the timeline does not move, so
every freeze is wrong until the confirm or a lock undoes it.
"""

import argparse
import os
import pickle
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsp
import replay_capture as rc
import sim

FIXTURES = os.path.join(HERE, "..", "..", "tests", "fixtures", "sync")
FIXTURE_RUNS = ["daily-clean", "livekit-clean", "daily-splice", "daily-splice-failures"]
LOST_COST_MS = 2000.0
MIN_GAIN = 0.01  # a change is kept when it lowers J by at least 1 %
SIM_CHUNKS = 12

# The start: the spike's constants (with the confirm's ratio check, the carried bracket and the
# digital-silence guard added since). dsp.STALL holds the chosen values.
BASE = replace(
    dsp.STALL,
    freeze_after_ms=40.0,
    silent_db=-70.0,
    ref_speech_db=-40.0,
    confirm_from_ms=300.0,
    confirm_until_ms=800.0,
)
# Swept and reported but never adopted: the confirm's acceptance thresholds stay align's own
# (peak 0.95, ratio 1.2, sharpness 0.03). A ratio of 1.11 already passed a false confirm on these
# runs (the reason the ratio check exists), so a lower J bought by relaxing them is margin spent.
REPORT_ONLY = {"confirm_min_peak", "confirm_min_ratio", "confirm_min_sharpness"}
GRID = {
    "freeze_after_ms": [20, 30, 40, 60, 80, 120],
    "silent_db": [-60.0, -65.0, -70.0, -80.0, -90.0],
    "ref_speech_db": [-30.0, -35.0, -40.0, -45.0, -50.0],
    "freeze_max_ms": [500, 1000, 2000, 3000],
    "confirm_from_ms": [150, 200, 250, 300, 400, 500],
    "confirm_until_ms": [600, 800, 1000, 1200],
    "confirm_back_ms": [50, 100, 150, 200, 300],
    "confirm_fwd_ms": [10, 20, 40, 60, 100],
    "confirm_min_peak": [0.9, 0.93, 0.95, 0.97],
    "confirm_min_ratio": [1.0, 1.1, 1.2, 1.3, 1.5],
    "confirm_min_sharpness": [0.01, 0.03, 0.05],
    "half_search_ms": [150, 300],
    "half_agree_ms": [5, 10, 15, 20],
    "half_silence_ms": [20, 40, 80, 100000],
    "match_tol_db": [3.0, 4.0, 6.0],
    "confirming_silent_db": [-1000.0, -95.0, -80.0, 0.0],
}
ABLATIONS = {
    "no freeze": {"freeze": False},
    "no confirm": {"confirm": False},
    "no newer half": {"newer_half": False},
    "loss clock counts frozen time": {"hold_clock": False},
    "no chained prior": {"chain_prior": False},
    "no ratio in confirm": {"confirm_min_ratio": 0.0},
    "freeze during a confirm drops its bracket": {"carry_bracket": False},
    "freeze during a confirm on any silence": {"confirming_silent_db": 0.0},
    "no freeze during a confirm": {"confirming_silent_db": -1000.0},
    "as the spike (no carry, freeze any time)": {
        "carry_bracket": False,
        "confirming_silent_db": 0.0,
    },
}


# ---------------------------------------------------------------- data


def capture_data(cap):
    """A capture's envelope (causal high-pass, as the browser) and per turn the dense truth."""
    x, fs, sync, _ = rc.load(cap)
    env, hop = rc.received_envelope(x, fs, True)
    today = rc.replay_all_tracked(env, hop, fs, len(x), sync, dsp.DEFAULTS, dsp.TODAY)
    truth = {}
    for t in sync["turns"]:
        own = rc.b64(t["reference"])
        if not len(own):
            continue
        own_db = dsp.dequantise(own)
        base = today[t["turn_id"]]["res"]
        acc = [k["l"] for k in (base["ticks"] if base else []) if k["accepted"]]
        centre = acc[0] if acc else today[t["turn_id"]]["prior_l_at"](t["start_tap"])
        tr = rc.dense_track(env, own_db, centre)
        lo = max(0, int(centre) - 400)
        hi = min(len(env), int(centre) + len(own_db) + 2 * rc.DENSE_SPAN)
        relaxed = rc.dense_truth_frames(tr, len(env))[lo:hi]
        strict = rc.dense_truth_frames(rc.dense_track(env, own_db, centre, strict=True), len(env))[
            lo:hi
        ]
        truth[t["turn_id"]] = (lo, relaxed, strict, tr)
    return {
        "name": os.path.basename(
            os.path.normpath(cap if not cap.endswith("chrome") else os.path.dirname(cap))
        ),
        "env": env,
        "hop": hop,
        "fs": fs,
        "n": len(x),
        "sync": sync,
        "truth": truth,
    }


def build(captures, jobs):
    t0 = time.time()
    cases = sim.build_stall_cases(jobs)
    print(f"# {len(cases)} simulated runs built in {time.time() - t0:.0f}s", file=sys.stderr)
    with Pool(min(jobs, len(captures))) as pool:
        caps = pool.map(capture_data, captures)
    print(f"# {len(caps)} captures prepared in {time.time() - t0:.0f}s", file=sys.stderr)
    return {"cases": cases, "captures": caps}


# ---------------------------------------------------------------- evaluation

_DATA = None


def _init(path):
    global _DATA
    with open(path, "rb") as fh:
        _DATA = pickle.load(fh)


def _full(truth_slice, lo, n):
    out = np.full(n, np.nan)
    out[lo : lo + len(truth_slice)] = truth_slice
    return out


def eval_capture(cap, stall):
    """Per turn of one capture: cost, sync-lost, false accepts, gaps."""
    env, hop, fs, sync = cap["env"], cap["hop"], cap["fs"], cap["sync"]
    out = rc.replay_all_tracked(env, hop, fs, cap["n"], sync, dsp.DEFAULTS, stall)
    rows = []
    for t in sync["turns"]:
        if t["turn_id"] not in cap["truth"]:
            continue
        res = out[t["turn_id"]]["res"]
        if res is None:
            continue
        lo, relaxed, strict, tr = cap["truth"][t["turn_id"]]
        strict_f = _full(strict, lo, len(env))
        relaxed_f = _full(relaxed, lo, len(env))
        false, straddle, scored, worst = rc.score_accepts(res, strict_f)
        first = next((k["count"] for k in res["ticks"] if k["accepted"]), None)
        off20 = 0.0
        if first is not None:
            js = np.array(sorted(j for j in res["disp"] if j >= first))
            if len(js):
                dl = np.array([res["disp"][j] for j in js])
                tt = relaxed_f[js]
                ok = ~np.isnan(tt)
                off20 = float((np.abs(dl[ok] - tt[ok]) * dsp.HOP_MS > 20).sum() * dsp.HOP_MS)
        own_db = dsp.dequantise(rc.b64(t["reference"]))
        rows.append(
            {
                "capture": cap["name"],
                "turn_id": t["turn_id"],
                "lost": len(res["lost_at"]),
                "cost": off20 + LOST_COST_MS * (len(res["lost_at"]) > 0),
                "false": false,
                "false_straddle": straddle,
                "scored": scored,
                "worst_ms": worst,
                "freezes": len(res["freezes"]),
                "confirmed": sum(f["how"] == "confirm" for f in res["freezes"]),
                "half": sum(k["via"] == "half" for k in res["ticks"]),
                "gaps": rc.gap_events(res, tr, env, own_db),
            }
        )
    return rows


def _task(args):
    kind, idx, stall = args
    if kind == "sim":
        cases = _DATA["cases"][idx::SIM_CHUNKS]
        rows = []
        for c in cases:
            r = sim.evaluate_stall_case(c, stall)
            r.update(cond=c["cond"], name=c["name"], seed=c["seed"])
            rows.append(r)
        return rows
    return eval_capture(_DATA["captures"][idx], stall)


def evaluate_many(pool, n_caps, stalls):
    """``[(sim rows, capture rows)]`` for every candidate in ``stalls``."""
    tasks = []
    for ci, st in enumerate(stalls):
        tasks += [(ci, ("sim", k, st)) for k in range(SIM_CHUNKS)]
        tasks += [(ci, ("cap", k, st)) for k in range(n_caps)]
    res = pool.map(_task, [t for _, t in tasks], chunksize=1)
    out = [([], []) for _ in stalls]
    for (ci, (kind, _, _)), rows in zip(tasks, res):
        out[ci][0 if kind == "sim" else 1].extend(rows)
    return out


def score(sim_rows, cap_rows):
    """The objective and the columns of the table."""
    sim_cost = np.array(
        [r["off20_post_ms"] + LOST_COST_MS * (r["lost"] > 0) for r in sim_rows], dtype=float
    )
    cond = np.array([r["cond"] for r in sim_rows])
    real_cost = np.array([r["cost"] for r in cap_rows], dtype=float)
    false = sum(len(r["false"]) for r in sim_rows) + sum(len(r["false"]) for r in cap_rows)
    straddle = sum(len(r["false_straddle"]) for r in sim_rows) + sum(
        len(r["false_straddle"]) for r in cap_rows
    )
    loss = np.isin(cond, ["loss-200", "loss-400"])
    shift = ~loss & (cond != "clean")
    gaps = [g for r in cap_rows for g in r["gaps"]]
    re_real = [g["realign_ms"] for g in gaps if g["realign_ms"] is not None]
    s = sim.summarise_stall(sim_rows)
    stall_re = [
        s[c]["realign_p50"]
        for c in s
        if c.startswith(("stall", "zeros")) and s[c]["realign_p50"] is not None
    ]
    return {
        "J": float(sim_cost.mean() + real_cost.mean()),
        "sim": float(sim_cost.mean()),
        "shift": float(sim_cost[shift].mean()),
        "loss": float(sim_cost[loss].mean()),
        "clean": float(sim_cost[cond == "clean"].mean()),
        "real": float(real_cost.mean()),
        "feasible": false == 0 and straddle == 0,
        "false": false,
        "straddle": straddle,
        "sim_lost": int(sum(r["lost"] > 0 for r in sim_rows)),
        "real_lost": int(sum(r["lost"] > 0 for r in cap_rows)),
        "stall_realign_p50": float(np.mean(stall_re)) if stall_re else None,
        "real_realign_p50": float(np.percentile(re_real, 50)) if re_real else None,
        "real_realign_p90": float(np.percentile(re_real, 90)) if re_real else None,
        "real_never": sum(g["realign_ms"] is None for g in gaps),
        "worst": float(max([r["worst_ms"] for r in sim_rows] + [r["worst_ms"] for r in cap_rows])),
        "summary": s,
    }


def row(label, sc):
    f = lambda v: "-" if v is None else f"{v:.0f}"
    return (
        f"| {label} | {sc['J']:.0f} | {sc['shift']:.0f} | {sc['loss']:.0f} | {sc['clean']:.0f} "
        f"| {sc['real']:.0f} | {sc['sim_lost']} | {sc['real_lost']} | {f(sc['stall_realign_p50'])} "
        f"| {f(sc['real_realign_p50'])}/{f(sc['real_realign_p90'])} ({sc['real_never']}) "
        f"| {sc['false']}/{sc['straddle']} | {sc['worst']:.1f} | {'yes' if sc['feasible'] else 'NO'} |"
    )


HEADER = (
    "| candidate | J ms | sim shift ms | sim loss ms | sim clean ms | real ms | sim runs lost "
    "| real turns lost | stall/zeros re-align p50 ms | real gaps re-align p50/p90 ms (never) "
    "| false outside/across gaps | worst accept error outside gaps ms | feasible |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--captures", nargs="*", default=[])
    ap.add_argument("--cache", default=None, help="data pickle (built when missing)")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    caps = [os.path.join(FIXTURES, r) for r in FIXTURE_RUNS] + a.captures
    cache = a.cache or os.path.join(HERE, ".thrsearch_stall.pkl")
    if not os.path.exists(cache):
        with open(cache, "wb") as fh:
            pickle.dump(build(caps, a.jobs), fh)
    with open(cache, "rb") as fh:
        n_caps = len(pickle.load(fh)["captures"])
    t0 = time.time()
    with Pool(a.jobs, initializer=_init, initargs=(cache,)) as pool:
        print(
            f"captures: {', '.join(os.path.relpath(c) if c.startswith(HERE) else c for c in caps)}"
        )
        print(
            f"cost per run / turn: ms > 20 ms off after the first lock + {LOST_COST_MS:.0f} ms if sync-lost\n"
        )
        print("## Reference points and ablations\n")
        print(HEADER)
        named = {"today": dsp.TODAY, "spike constants": BASE}
        named.update({k: replace(BASE, **v) for k, v in ABLATIONS.items()})
        scores = dict(
            zip(named, (score(*r) for r in evaluate_many(pool, n_caps, list(named.values()))))
        )
        for k, sc in scores.items():
            print(row(k, sc))
        best, best_sc = BASE, scores["spike constants"]
        for rnd in range(a.rounds):
            print(f"\n## Coordinate descent, round {rnd + 1} (others at the current best)\n")
            changed = False
            for name, values in GRID.items():
                cands = [replace(best, **{name: type(getattr(best, name))(v)}) for v in values]
                res = [score(*r) for r in evaluate_many(pool, n_caps, cands)]
                print(f"\n### {name} (current {getattr(best, name)})\n")
                print(HEADER)
                for v, sc in zip(values, res):
                    print(row(f"{name}={v}", sc))
                ok = [(sc["J"], i) for i, sc in enumerate(res) if sc["feasible"]]
                if name in REPORT_ONLY:
                    print("\n(reported only: the confirm keeps align's thresholds)")
                elif ok:
                    j, i = min(ok)
                    if j < best_sc["J"] * (1 - MIN_GAIN) and getattr(cands[i], name) != getattr(
                        best, name
                    ):
                        best, best_sc, changed = cands[i], res[i], True
                        print(f"\n-> {name} = {getattr(best, name)} (J {j:.0f})")
                sys.stdout.flush()
            if not changed:
                break
        print("\n## Chosen\n")
        print(HEADER)
        print(row("today", scores["today"]))
        print(row("spike constants", scores["spike constants"]))
        print(row("chosen", best_sc))
        diff = {
            k: (getattr(BASE, k), getattr(best, k))
            for k in GRID
            if getattr(BASE, k) != getattr(best, k)
        }
        print(f"\nchanged from the spike: {diff or 'nothing'}")
        print(f"dsp.STALL (the defaults) is the chosen point: {best == dsp.STALL}")
        print("\n### Per condition, chosen (simulation)\n")
        sim.print_stall_table(
            {
                "chosen": {"summary": best_sc["summary"]},
                "today": {"summary": scores["today"]["summary"]},
            }
        )
    print(f"\n# search took {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
