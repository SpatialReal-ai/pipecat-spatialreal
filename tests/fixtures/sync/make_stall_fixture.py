"""Write the oracle's ticks for the probe page's port of the stall rule.

``examples/probe/sync.js`` ports ``dsp.TurnTracker`` (the stall rule) and the tick loop of
``hack/sync-sim/replay_capture.py`` (``replay_all_tracked``: cadence, priors, rule 5 and R76);
``tests/probe_check.mjs`` drives the page's ``SyncTracker`` over the same inputs and compares it
with what this file stores, tick for tick: positions, accept/hold, how a tick accepted, the newer
half, the loss clock, the tracked position, and the ``frozen`` / ``unfrozen`` events and closed
freezes.

Two outputs:

* ``stall.json`` (next to this file): synthetic cases built from speech-like noise at 24 kHz -- a
  stall, a burst loss, a stall inside a periodic stretch (one of exactly one period, where only
  the no-shift check's second-peak guard refuses "nothing moved"), a stall past the 2 s cap, a
  chained turn after a stall in its predecessor's trailing silence (the chain far off and the
  predecessor's speech out of the window: R76; near: the onset prior re-anchors, R74), and a
  chained turn announced while its predecessor is frozen (its prior counts the frozen frames so
  far). Each case holds the received envelope (dB, as the browser
  measures it: causal 30 Hz high-pass), the avatar messages with their tap positions, the onset
  prior as the page's analysis would set it (per 128-sample block), and the oracle's result with
  the stall rule (``dsp.STALL``) and without it (``dsp.TODAY``).
* ``daily-stall/oracle-ticks.json``: the same for the committed stalled capture
  (``daily-stall/``): the received envelope the replay measures with the browser's causal
  high-pass, the onset prior the page recorded (``replay_capture.page_prior``) and the oracle's
  result. The messages are the capture's own ``capture.jsonl``.

The envelopes are given to the page's tracker as they are here, so the comparison is of the
tracker alone; ``probe_check.mjs`` also runs the capture with the page's own envelope from
``capture.wav``.

usage: ``uv run python tests/fixtures/sync/make_stall_fixture.py`` (rewrites both;
``tests/test_probe_page.py`` fails when a committed file differs from a fresh run).
"""

import base64
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parent.parent.parent / "hack" / "sync-sim"
OUT = HERE / "stall.json"
CAPTURE = HERE / "daily-stall"
CAPTURE_OUT = CAPTURE / "oracle-ticks.json"

RATE = 24000
BLOCK = 128  # the AudioWorklet's block: taps are multiples of it
LEAD_S = 0.5  # every synthetic track starts with this much silence; the turn's true place


def load_replay():
    """``hack/sync-sim/replay_capture.py`` by path (its ``import dsp`` needs its directory)."""
    spec = importlib.util.spec_from_file_location(
        "stall_fixture_replay_capture", SIM / "replay_capture.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SIM))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SIM))
    return module


rc = load_replay()
dsp = rc.dsp


def speechlike(seconds, seed):
    """Noise in syllable-length bursts with short pauses of exact zeros (``tests/test_stall_rule.py``'s
    ``_speechlike``), int16-scale floats."""
    rng = np.random.default_rng(seed)
    n = round(seconds * RATE)
    env = np.zeros(n)
    pos = 0
    while pos < n:
        syllable = round(rng.uniform(0.08, 0.25) * RATE)
        pause = round(rng.uniform(0.02, 0.06) * RATE)
        t = np.linspace(0, np.pi, syllable)
        env[pos : pos + syllable] = rng.uniform(0.1, 0.6) * np.sin(t)[: n - pos]
        pos += syllable + pause
    return np.round(rng.standard_normal(n) * env * 32767 * 0.5)


def syllable_at(pcm, from_s):
    """50 ms into the first loud stretch at or after ``from_s`` (seconds)."""
    loud = np.flatnonzero(np.abs(pcm[round(from_s * RATE) :]) > 3000)
    return from_s + loud[0] / RATE + 0.05


def tap(seconds):
    """A tap position: a whole block at or after ``seconds``."""
    return -(-round(seconds * RATE) // BLOCK) * BLOCK


def b64(values):
    return base64.b64encode(bytes(np.asarray(values, np.uint8))).decode()


def turn(turn_id, pcm, start_s, onset_prior_pos=None, prev=None):
    """A turn whose whole envelope arrives with its ``turn_start`` and whose ``turn_end`` follows
    100 ms later. ``onset_prior_pos``: what the page's analysis sets from the first tick on (as a
    recorded onset prior). ``prev``: ``(previous turn id, its pcm ms)`` for a chained turn."""
    ref = dsp.envelope_bytes(pcm, RATE)
    start = tap(start_s)
    pcm_ms = len(pcm) / RATE * 1000
    t = {
        "turn_id": turn_id,
        "start_tap": start,
        "sample_rate": RATE,
        "onsets_ms": [],
        "prev_turn_id": prev[0] if prev else "",
        "prev_turn_pcm_ms": prev[1] if prev else 0,
        "gap_ms": 0,
        "end_pcm_ms": pcm_ms,
        "end_tap": start + tap(0.1),
        "abort_tap": None,
        "reference": b64(ref),
        "splice": None,
        "onset_prior_pos": onset_prior_pos,
        "ticks": [],
    }
    if onset_prior_pos is not None:
        t["ticks"] = [
            {"pos": 0, "prior_pos": onset_prior_pos, "prior_kind": "onset", "accepted": False}
        ]
    if prev:
        t["splice"] = {"kind": "chained", "from": prev[0], "join_ms": prev[1]}
    messages = [
        {
            "tap": start,
            "data": {
                "tag": "spatialreal-avatar",
                "turn_start": {
                    "turn_id": turn_id,
                    "sample_rate": RATE,
                    "onsets_ms": [],
                    "prev_turn_id": t["prev_turn_id"],
                    "prev_turn_pcm_ms": t["prev_turn_pcm_ms"],
                    "gap_ms": 0,
                },
            },
        },
        {
            "tap": start,
            "data": {
                "tag": "spatialreal-avatar",
                "turn_envelope": {
                    "turn_id": turn_id,
                    "first_index": 0,
                    "values": t["reference"],
                    "hop_ms": 5,
                    "window_ms": 10,
                },
            },
        },
        {
            "tap": t["end_tap"],
            "data": {
                "tag": "spatialreal-avatar",
                "turn_end": {"turn_id": turn_id, "pcm_ms": pcm_ms},
            },
        },
    ]
    return t, (start, 0, ref), messages


def track(*parts):
    """The received track: silence for ``LEAD_S``, the parts, 1 s of silence, padded to a block."""
    x = np.concatenate([np.zeros(round(LEAD_S * RATE)), *parts, np.zeros(RATE)])
    return np.concatenate([x, np.zeros(-len(x) % BLOCK)])


def stall_case(seconds, seed, at_from_s, stall_s):
    pcm = speechlike(seconds, seed)
    a = round(syllable_at(pcm, at_from_s) * RATE)
    return [turn("t", pcm, LEAD_S - 0.2)], track(pcm[:a], np.zeros(round(stall_s * RATE)), pcm[a:])


def burst_case():
    # 300 ms of content replaced by digital silence, the timeline unchanged: a wrong freeze that
    # the no-shift check undoes
    pcm = speechlike(4.0, 8)
    x = track(pcm)
    a = round((LEAD_S + syllable_at(pcm, 1.8)) * RATE)
    x[a : a + round(0.3 * RATE)] = 0.0
    return [turn("t", pcm, LEAD_S - 0.2)], x


def periodic_case(period, seed, extra_s):
    # speech, a stretch whose envelope repeats every period, speech; a stall inside the periodic
    # stretch (as tests/test_stall_rule.py's periodic case)
    rng = np.random.default_rng(seed)
    u = round(period * RATE)
    unit = np.round(rng.standard_normal(u) * np.hanning(u) * 8000)
    pcm = np.concatenate(
        [speechlike(2.0, seed), np.tile(unit, round(1.6 / period)), speechlike(1.0, seed + 9)]
    )
    a = round((2.0 + 3.5 * period) * RATE)
    stall = np.zeros(round((period + extra_s) * RATE))
    return [turn("t", pcm, LEAD_S - 0.2)], track(pcm[:a], stall, pcm[a:])


def chain_case(z_s, b_start_s):
    # turn A, then B chained from it; z_s of stall in A's trailing silence (A cannot freeze there),
    # so the chain puts B early by z_s; B's onset prior is its true place. B's turn_start (and its
    # envelope) arrive at b_start_s: early, B locks on A's audio through the splice; once A's speech
    # has left the window, only the wide search can find B (R76)
    pcm_a = np.concatenate([speechlike(2.6, 21), np.zeros(round(0.4 * RATE))])
    pcm_b = speechlike(3.0, 22)
    a = round(2.75 * RATE)
    x = track(pcm_a[:a], np.zeros(round(z_s * RATE)), pcm_a[a:], pcm_b)
    lb = round(LEAD_S * RATE) + len(pcm_a) + round(z_s * RATE)  # B's true first sample
    ta = turn("A", pcm_a, LEAD_S - 0.2)
    tb = turn("B", pcm_b, b_start_s, onset_prior_pos=lb, prev=("A", len(pcm_a) / RATE * 1000))
    return [ta, tb], x


def chain_frozen_case(stall_s):
    # B chained from A, announced while A is frozen in a stall inside its speech: B's prior before
    # its first lock is A's position *now* (frozen frames included) + A's pcm ms
    pcm_a = speechlike(3.0, 21)
    pcm_b = speechlike(2.0, 22)
    at = syllable_at(pcm_a, 2.0)
    a = round(at * RATE)
    x = track(pcm_a[:a], np.zeros(round(stall_s * RATE)), pcm_a[a:], pcm_b)
    ta = turn("A", pcm_a, LEAD_S - 0.2)
    tb = turn("B", pcm_b, LEAD_S + at + 0.25, prev=("A", len(pcm_a) / RATE * 1000))
    return [ta, tb], x


CASES = {
    "stall-300": lambda: stall_case(4.0, 8, 1.8, 0.3),
    "burst-loss-300": burst_case,
    "periodic-150": lambda: periodic_case(0.15, 5, 0.06),
    "periodic-200-one-period": lambda: periodic_case(0.2, 4, 0.0),
    "stall-cap-2500": lambda: stall_case(5.0, 8, 2.0, 2.5),
    "chain-r76-800": lambda: chain_case(0.8, LEAD_S + 3.6),
    "chain-near-150": lambda: chain_case(0.15, LEAD_S + 0.8),
    "chain-while-frozen-500": lambda: chain_frozen_case(0.5),
}


def steps_of(prior_at, n_samples):
    """The onset prior the replay uses (``replay_capture.page_prior``, ``prior_at``) at every block
    position, as change points ``[[from tap, samples or None], ...]``: what the page's analysis
    would have set (``SyncTracker.setOnsetPrior``) before the block that ends at that tap."""
    out, last = [], object()
    for pos in range(BLOCK, n_samples + 1, BLOCK):
        v = prior_at(pos)
        if v != last:
            out.append([pos, None if v is None else float(v)])
            last = v
    return out


def plain(v):
    """JSON-safe: numpy scalars to Python numbers."""
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def expected(tracked):
    """The oracle's per-turn result, as the page's port is compared on it."""
    out = {}
    for tid, entry in tracked.items():
        res = entry["res"]
        if res is None:
            continue
        ticks = []
        for k in res["ticks"]:
            ticks.append(
                {
                    "pos": k["pos"],
                    "count": k["count"],
                    "speech": k["speech"],
                    "accepted": k["accepted"],
                    "via": k["via"],
                    "l": k["l"],
                    "candidate_l": k["candidate_l"],
                    "peak": None if k["checks"][0] is None else float(k["checks"][0]),
                    "half_l": k["half_l"],
                    "half_refused": k["half_refused"],
                    "prior_l": k["prior_l"],
                    "wide": k["wide"],
                    "unlocked_ms": k["unlocked_ms"],
                    "lost": k["lost"],
                    "tracked_l": rc.tracked_place_at(res, k["pos"]),
                }
            )
        out[tid] = plain(
            {
                "ticks": ticks,
                "events": [[j, e] for j, e in res["events"]],
                "freezes": res["freezes"],
                "lost_at": res["lost_at"],
            }
        )
    return out


def stall_params():
    """``dsp.STALL`` with the port's names (camelCase)."""
    words = lambda k: k.split("_")[0] + "".join(w.capitalize() for w in k.split("_")[1:])
    return {words(k): v for k, v in dsp.STALL.__dict__.items()}


def build_case(name):
    turns, x = CASES[name]()
    hop = round(dsp.HOP_MS / 1000 * RATE)
    env = dsp._window_db(rc.causal_high_pass(x, RATE), hop, 2 * hop)
    sync = {
        "turns": [t for t, _, _ in turns],
        "arrivals": {t["turn_id"]: [arr] for t, arr, _ in turns},
    }
    messages = sorted((m for _, _, ms in turns for m in ms), key=lambda m: m["tap"])
    run = {
        rule: expected(
            rc.replay_all_tracked(env, hop, RATE, len(x), sync, dsp.DEFAULTS, stall, r72_all=True)
        )
        for rule, stall in (("stall", dsp.STALL), ("today", dsp.TODAY))
    }
    return {
        "name": name,
        "rate": RATE,
        "hop": hop,
        "n_samples": len(x),
        "env": [float(v) for v in env],
        "messages": messages,
        "onset_steps": {
            t["turn_id"]: steps_of(rc.page_prior(t, RATE, hop), len(x)) for t, _, _ in turns
        },
        "expected": run,
    }


def build():
    """The synthetic cases (``stall.json``)."""
    return {
        "about": "the oracle's stall-rule ticks on synthetic cases for examples/probe/sync.js "
        "(tests/fixtures/sync/make_stall_fixture.py)",
        "stall": stall_params(),
        "cases": [build_case(name) for name in CASES],
    }


def build_capture(capture=CAPTURE):
    """The committed stalled capture's oracle ticks (``daily-stall/oracle-ticks.json``)."""
    x, fs, sync, _ = rc.load(str(capture))
    env, hop = rc.received_envelope(x, fs, True)
    tracked = rc.replay_all_tracked(env, hop, fs, len(x), sync, dsp.DEFAULTS, dsp.STALL)
    return {
        "about": "the oracle's stall-rule ticks on this capture (causal high-pass) for "
        "examples/probe/sync.js (tests/fixtures/sync/make_stall_fixture.py)",
        "rate": fs,
        "hop": hop,
        "n_samples": len(x),
        "env": [float(v) for v in env],
        "onset_steps": {
            t["turn_id"]: steps_of(rc.page_prior(t, fs, hop), len(x)) for t in sync["turns"]
        },
        "expected": {"stall": expected(tracked)},
    }


def main():
    OUT.write_text(json.dumps(build(), separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    if CAPTURE.is_dir():
        CAPTURE_OUT.write_text(json.dumps(build_capture(), separators=(",", ":")) + "\n")
        print(f"wrote {CAPTURE_OUT} ({CAPTURE_OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
