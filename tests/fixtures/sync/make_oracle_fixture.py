"""Write ``oracle.json``: the normative oracle's own numbers for the probe page's port of the rule.

``examples/probe/sync.js`` ports ``envelope_bytes`` and ``align`` from ``hack/sync-sim/dsp.py``;
``tests/probe_check.mjs`` checks the port against this file. It holds one stretch of speech-like
int16 audio and, for each case, a recipe that builds the turn's source and the received track from
it with integer operations only (sample delays, tiling, sample-and-hold, a 32-bit LCG for room
noise), so the browser code can rebuild every track exactly from the same samples instead of the
file carrying them. For each case the oracle's ``align`` result is stored twice: on the envelope
without a high-pass (the port must match it to 1e-6) and with the oracle's zero-phase 30 Hz
high-pass (the port's causal one must land within 0.5 ms of its offset). A few cases give the
received envelope in dB directly, to put the speech gate exactly on its boundary.

usage: ``uv run python tests/fixtures/sync/make_oracle_fixture.py`` (rewrites ``oracle.json`` next
to this file; ``tests/test_probe_page.py`` fails when the committed file differs from a fresh run).
"""

import base64
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent.parent.parent / "hack" / "sync-sim" / "dsp.py"
OUT = HERE / "oracle.json"

RATE = 24000
SECONDS = 2.0
SEED = 7
ENVELOPE_RATES = (16000, 22050, 24000, 44100, 48000)


def load_oracle():
    """Load ``hack/sync-sim/dsp.py`` by path, as ``tests/test_fingerprint_oracle.py`` does."""
    spec = importlib.util.spec_from_file_location("sync_sim_dsp", ORACLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def speechlike(seconds: float, seed: int, rate: int = RATE) -> np.ndarray:
    """Noise shaped by syllable-length bursts and short pauses, as int16 (the oracle test's
    ``_speechlike``): an envelope with structure on the 100-300 ms scale, like speech."""
    rng = np.random.default_rng(seed)
    n = round(seconds * rate)
    env = np.zeros(n)
    pos = 0
    while pos < n:
        syllable = round(rng.uniform(0.08, 0.25) * rate)
        pause = round(rng.uniform(0.02, 0.12) * rate)
        level = rng.uniform(0.05, 0.6)
        t = np.linspace(0, np.pi, syllable)
        env[pos : pos + syllable] = level * np.sin(t)[: n - pos]
        pos += syllable + pause
    return np.round(rng.standard_normal(n) * env * 32767 * 0.5).astype(np.int16)


def build_source(pcm: np.ndarray, case: dict) -> np.ndarray:
    """The turn's audio as the processor released it (its envelope is the reference).

    ``source`` is absent for ``pcm`` itself; ``{"kind": "repeat", "from", "length", "total"}``
    tiles ``pcm[from:from + length]`` to ``total`` samples (a periodic envelope: a second peak as
    high as the first); ``{"kind": "hold", "factor", "total"}`` holds every sample ``factor`` times
    (a slow envelope: a blunt peak).
    """
    spec = case.get("source")
    if spec is None:
        return pcm
    n = np.arange(spec["total"])
    if spec["kind"] == "repeat":
        return pcm[spec["from"] + n % spec["length"]]
    if spec["kind"] == "hold":
        return pcm[n // spec["factor"]]
    raise ValueError(spec)


def lcg_noise(count: int, amplitude: int, seed: int) -> np.ndarray:
    """Uniform integer noise in [-amplitude, amplitude] from a 32-bit LCG (Numerical Recipes'
    constants), ``(state >> 16) % (2 * amplitude + 1) - amplitude``, state updated first."""
    out = np.empty(count, dtype=np.int64)
    state = seed
    for i in range(count):
        state = (1664525 * state + 1013904223) % 2**32
        out[i] = (state >> 16) % (2 * amplitude + 1) - amplitude
    return out


def build_received(source: np.ndarray, case: dict) -> np.ndarray:
    """The received track of ``case``: from each segment's ``from`` sample on, the source delayed by
    that segment's ``lead`` samples (zeros where it has no sample), ``samples`` long in all, plus
    the case's ``noise`` (LCG, see :func:`lcg_noise`) clipped to the int16 range.

    The source is reversed when the case's ``reverse`` is set, and all zeros when ``silent`` is. A
    segment may set its own ``reverse`` (other audio for that stretch) and ``shift`` (its samples
    arithmetic-shifted right by that many bits: each bit is 6 dB quieter, so shift 2 is 12 dB).
    """
    src = (
        np.zeros_like(source)
        if case.get("silent")
        else source[::-1]
        if case.get("reverse")
        else source
    )
    out = np.zeros(case["samples"], dtype=np.float64)
    bounds = [s["from"] for s in case["segments"]] + [case["samples"]]
    for seg, end in zip(case["segments"], bounds[1:]):
        n = np.arange(seg["from"], end)
        k = n - seg["lead"]
        own = src[::-1] if seg.get("reverse") else src
        inside = (k >= 0) & (k < len(own))
        out[n[inside]] = own[k[inside]].astype(np.int64) >> seg.get("shift", 0)
    if "noise" in case:
        noise = lcg_noise(case["samples"], case["noise"]["amplitude"], case["noise"]["seed"])
        out = np.clip(out + noise, -32768, 32767)
    return out


def number(x):
    """A float JSON can carry: infinities as the strings ``"Infinity"`` / ``"-Infinity"``."""
    if x is None or isinstance(x, bool):
        return x
    x = float(x)
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x


def result(res) -> dict:
    """An ``AlignResult`` as JSON (the port's field names)."""
    c = res.checks
    return {
        "offset_ms": number(res.offset_ms),
        "accepted": c.accepted,
        "speech": c.speech,
        "window_ok": c.window_ok,
        "peak": number(c.peak),
        "ratio": number(c.ratio),
        "sharpness": number(c.sharpness),
        "half_deviation_ms": number(c.half_deviation_ms),
        "edge": c.edge,
        "candidate_ms": number(c.candidate_ms),
        "peak_ok": c.peak_ok,
        "ratio_ok": c.ratio_ok,
        "sharpness_ok": c.sharpness_ok,
        "halves_ok": c.halves_ok,
        "interior_ok": c.interior_ok,
    }


def half_deviations(dsp, received_db, reference, prior_index, params):
    """Each half's own deviation from the full window's peak, ms (``inf`` at the edge of its
    +/- 40 ms search, ``None`` without speech): what ``dsp.align`` takes the maximum of, built
    from the same public blocks, so a case can say which half decided."""
    hop = params.hop_ms
    rec = np.asarray(received_db, dtype=np.float64)
    jt = len(rec)
    m_full = round(params.window_ms / hop)
    m_half = m_full // 2
    floor = dsp.floor_db(rec[max(0, jt - int(params.floor_history_ms / hop)) : jt], params)
    prior_ms = prior_index * hop
    lo_ms, hi_ms = prior_ms - params.search_ms, prior_ms + params.search_ms
    margin = params.half_search_ms + 2 * hop
    lags = np.arange(
        int(np.floor((lo_ms - margin) / hop)), int(np.ceil((hi_ms + margin) / hop)) + 1
    )
    lags_ms = lags * hop
    ref_db = dsp.dequantise(reference)
    pad = max(lags[-1] - (jt - m_full), (jt - lags[0]) - len(ref_db), 0) + 1
    ref_padded = dsp.pad_reference(ref_db, pad)

    def curve(length, end):
        window = rec[end - length : end]
        return dsp.ncc_curve(
            window, ref_padded, pad + end - length, lags, floor, params.gate_std_db
        )

    est = dsp.peak_features(lags_ms, curve(m_full, jt), lo_ms, hi_ms)[3]
    out = []
    for c in (curve(m_half, jt - m_half), curve(m_half, jt)):
        if c is None:
            out.append(None)
            continue
        _, _, _, half_est, edge = dsp.peak_features(
            lags_ms, c, est - params.half_search_ms, est + params.half_search_ms
        )
        out.append(number(np.inf if edge else abs(half_est - est)))
    return out


def ms(value: float, rate: int = RATE) -> int:
    """Samples in ``value`` ms at ``rate`` (every value used here is exact)."""
    return round(value * rate / 1000)


def gate_window(half_range_db: float) -> list[float]:
    """A received envelope (dB) whose last 1 s alternates ``-48 -/+ half_range_db``: its standard
    deviation is exactly ``half_range_db``. Two seconds of -90 dB before it keep the floor at
    -70 dB, below every value of the window, so the gate sees the window unclipped."""
    return [-90.0] * 400 + [
        -48.0 + (half_range_db if i % 2 else -half_range_db) for i in range(200)
    ]


# The received tracks. "lead" is where the source's sample 0 sits in the received track; the
# reference's index 0 therefore sits at lead / 24 ms, and the prior is in hops of 5 ms.
CASES = [
    # The reference 500 + 137.25 ms into the track (not on the 5 ms grid), prior 12.25 ms off.
    {"name": "shifted", "segments": [{"from": 0, "lead": ms(637.25)}], "samples": ms(2400),
     "prior_index": 125},
    # The same with a fractional prior: the lag range's floor and ceil differ.
    {"name": "shifted-fractional-prior", "segments": [{"from": 0, "lead": ms(637.25)}],
     "samples": ms(2400), "prior_index": 131.7},
    # The audio jumps by 40 ms at 1.9 s, the middle of the last 1 s: the halves disagree.
    {"name": "halves-disagree",
     "segments": [{"from": 0, "lead": ms(637.25)}, {"from": ms(1900), "lead": ms(677.25)}],
     "samples": ms(2400), "prior_index": 125},
    # Other audio (the source reversed): nothing should be accepted.
    {"name": "other-audio", "segments": [{"from": 0, "lead": ms(637.25)}], "samples": ms(2400),
     "prior_index": 125, "reverse": True},
    # The reference 932 ms in, the search 325..925 ms: the peak sits at the edge.
    {"name": "edge", "segments": [{"from": 0, "lead": ms(932)}], "samples": ms(3000),
     "prior_index": 125},
    # Silence: the speech gate.
    {"name": "silent", "segments": [{"from": 0, "lead": 0}], "samples": ms(2000),
     "prior_index": 125, "silent": True},
    # Less than one 1 s window of envelope: not enough to decide.
    {"name": "short", "segments": [{"from": 0, "lead": ms(137.25)}], "samples": ms(900),
     "prior_index": 25},
    # Room noise at about -43 dB: the floor is the received envelope's 5th percentile + 6 dB over
    # the last 3 s (here the whole 2.4 s), well above -70, and it clips both sides.
    {"name": "room-noise", "segments": [{"from": 0, "lead": ms(637.25)}], "samples": ms(2400),
     "prior_index": 125, "noise": {"amplitude": 400, "seed": 20261103}},
    # A 251 ms period: a side lobe 0.3 % below the peak. Fails on the ratio alone. (A 250 ms period,
    # a whole number of hops, makes the side lobe tie the peak exactly, and which of two equal
    # floats is the maximum is up to the platform's arithmetic.)
    {"name": "ratio-only", "source": {"kind": "repeat", "from": 0, "length": ms(251), "total": ms(2000)},
     "segments": [{"from": 0, "lead": ms(637.25)}], "samples": ms(2400), "prior_index": 125},
    # Every sample held 6 times: syllables six times longer, a blunt peak. Fails on sharpness alone.
    {"name": "sharpness-only", "source": {"kind": "hold", "factor": 6, "total": ms(4000)},
     "segments": [{"from": 0, "lead": ms(637.25)}], "samples": ms(2400), "prior_index": 125},
    # The audio 50 ms later until 1.85 s: the full window follows the late audio (+34 ms), so the
    # on-time newer half is the one that peaks past its +/- 40 ms search.
    {"name": "late-until-1.85",
     "segments": [{"from": 0, "lead": ms(687.25)}, {"from": ms(1850), "lead": ms(637.25)}],
     "samples": ms(2400), "prior_index": 125},
    # The audio 50 ms later from 2.0 s: the newer half peaks past its +/- 40 ms search.
    {"name": "newer-half-jump",
     "segments": [{"from": 0, "lead": ms(637.25)}, {"from": ms(2000), "lead": ms(687.25)}],
     "samples": ms(2400), "prior_index": 125},
    # Other audio (reversed, 12 dB down) from 1.425 to 1.675 s, inside the older half only: the
    # older half peaks 16 ms off, the newer one 4 ms. With the peak threshold relaxed to 0.80
    # (the full window loses some correlation to the stretch) the halves check alone rejects it,
    # and it is the older half that does.
    {"name": "older-half-only",
     "segments": [{"from": 0, "lead": ms(637.25)},
                  {"from": ms(1425), "lead": ms(637.25), "reverse": True, "shift": 2},
                  {"from": ms(1675), "lead": ms(637.25)}],
     "samples": ms(2400), "prior_index": 125, "params": {"min_peak": 0.8}},
    # Room noise over 3.4 s: the floor's 3 s history is not the whole track.
    {"name": "room-noise-long", "segments": [{"from": 0, "lead": ms(1637.25)}], "samples": ms(3400),
     "prior_index": 325, "noise": {"amplitude": 400, "seed": 7}},
    # The same samples read at 48 kHz (half the length): the rule at the browser's usual rate.
    {"name": "shifted-48k", "rate": 48000,
     "segments": [{"from": 0, "lead": ms(637.25, 48000)}], "samples": ms(2000, 48000),
     "prior_index": 125},
    # The speech gate on its boundary: std 1.5 dB (silence), 1.999 dB (silence, though the
    # sample standard deviation would be 2.004) and exactly 2 dB (speech: the gate is "< 2").
    # "gate_only": the window's correlation is flat by construction (best and second-best lags
    # within 1e-17 of each other), so where its peak lies is decided by rounding and differs
    # between platforms: only the gate and the outcome are compared (probe_check.mjs, and the
    # regeneration test in test_probe_page.py).
    {"name": "gate-1.5", "received_db": gate_window(1.5), "prior_index": 125, "gate_only": True},
    {"name": "gate-1.999", "received_db": gate_window(1.999), "prior_index": 125, "gate_only": True},
    {"name": "gate-2.0", "received_db": gate_window(2.0), "prior_index": 125, "gate_only": True},
]  # fmt: skip

# Received hop in samples per rate (dsp.envelope_db: Python's round, half to even); 44 100 Hz is
# a tie (220.5 -> 220) and 44 140 Hz is not (220.7 -> 221).
HOP_RATES = (16000, 22050, 24000, 44100, 44140, 48000)


def build() -> dict:
    """The fixture, as a JSON-ready dict."""
    dsp = load_oracle()
    pcm = speechlike(SECONDS, SEED)
    cases = []
    for case in CASES:
        rate = case.get("rate", RATE)
        source = build_source(pcm, case)
        reference = dsp.envelope_bytes(source, rate)
        entry = dict(case)
        params = dataclasses.replace(dsp.DEFAULTS, **case.get("params", {}))
        prior = case["prior_index"]
        if "received_db" in case:
            received_db = np.asarray(case["received_db"])
            entry["oracle_no_hp"] = result(dsp.align(received_db, reference, prior, params))
            cases.append(entry)
            continue
        received = build_received(source, case)
        plain = dsp.envelope_db(received, rate, 0.005)
        filtered = dsp.envelope_db(received, rate, 0.005, dc_block=True)
        entry["oracle_no_hp"] = result(dsp.align(plain, reference, prior, params))
        entry["oracle_hp"] = result(dsp.align(filtered, reference, prior, params))
        if "half" in case["name"] or "until" in case["name"]:
            # [older, newer]: which half decided.
            entry["oracle_half_deviations_ms"] = half_deviations(
                dsp, plain, reference, prior, params
            )
        if case["name"] == "shifted":
            # The received envelope itself, so a mismatch shows up before the rule runs.
            entry["received_db_no_hp"] = [float(v) for v in plain]
        cases.append(entry)
    return {
        "about": "written by make_oracle_fixture.py from hack/sync-sim/dsp.py; do not edit",
        "rate": RATE,
        "pcm_int16_le": base64.b64encode(pcm.astype("<i2").tobytes()).decode(),
        # The same samples read at other rates: the hop and window lengths change, the rule not.
        "envelope_bytes": {
            str(rate): base64.b64encode(dsp.envelope_bytes(pcm, rate).tobytes()).decode()
            for rate in ENVELOPE_RATES
        },
        "received_hops": {str(rate): round(0.005 * rate) for rate in HOP_RATES},
        "cases": cases,
    }


def main() -> int:
    """Rewrite ``oracle.json``."""
    OUT.write_text(json.dumps(build(), separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
