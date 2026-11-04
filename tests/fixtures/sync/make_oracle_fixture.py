"""Write ``oracle.json``: the normative oracle's own numbers for the probe page's port of the rule.

``examples/probe/sync.js`` ports ``envelope_bytes`` and ``align`` from ``hack/sync-sim/dsp.py``;
``tests/probe_check.mjs`` checks the port against this file. It holds one stretch of speech-like
int16 audio and, for each case, a recipe that builds the received track from it with integer sample
delays only, so the browser code can rebuild every received track exactly from the same samples
instead of the file carrying them. For each case the oracle's ``align`` result is stored twice: on
the envelope without a high-pass (the port must match it to 1e-6) and with the oracle's zero-phase
30 Hz high-pass (the port's causal one must land within 0.5 ms of its offset).

usage: ``uv run python tests/fixtures/sync/make_oracle_fixture.py`` (rewrites ``oracle.json`` next
to this file; ``tests/test_probe_page.py`` fails when the committed file differs from a fresh run).
"""

import base64
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


def build_received(pcm: np.ndarray, case: dict) -> np.ndarray:
    """The received track of ``case``: from each segment's ``from`` sample on, the source delayed by
    that segment's ``lead`` samples (zeros where it has no sample), ``samples`` long in all.

    The source is ``pcm`` reversed when ``reverse`` is set, and all zeros when ``silent`` is.
    """
    src = np.zeros_like(pcm) if case.get("silent") else pcm[::-1] if case.get("reverse") else pcm
    out = np.zeros(case["samples"], dtype=np.float64)
    bounds = [s["from"] for s in case["segments"]] + [case["samples"]]
    for seg, end in zip(case["segments"], bounds[1:]):
        for n in range(seg["from"], end):
            k = n - seg["lead"]
            if 0 <= k < len(src):
                out[n] = src[k]
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


def ms(value: float) -> int:
    """Samples in ``value`` ms at :data:`RATE` (every value used here is exact)."""
    return round(value * RATE / 1000)


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
]  # fmt: skip


def build() -> dict:
    """The fixture, as a JSON-ready dict."""
    dsp = load_oracle()
    pcm = speechlike(SECONDS, SEED)
    reference = dsp.envelope_bytes(pcm, RATE)
    cases = []
    for case in CASES:
        received = build_received(pcm, case)
        plain = dsp.envelope_db(received, RATE, 0.005)
        filtered = dsp.envelope_db(received, RATE, 0.005, dc_block=True)
        entry = dict(case)
        entry["oracle_no_hp"] = result(dsp.align(plain, reference, case["prior_index"]))
        entry["oracle_hp"] = result(dsp.align(filtered, reference, case["prior_index"]))
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
        "cases": cases,
    }


def main() -> int:
    """Rewrite ``oracle.json``."""
    OUT.write_text(json.dumps(build(), separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
