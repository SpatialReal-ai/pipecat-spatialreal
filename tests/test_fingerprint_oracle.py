"""The package's envelope against the normative oracle, ``hack/sync-sim/dsp.py``.

The oracle (numpy, dev only) is what the web SDK ports; the package computes the same bytes
without numpy. These tests pin the two together bit for bit and check the alignment rule on
the cases that must always hold.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from pipecat_spatialreal import fingerprint
from pipecat_spatialreal.fingerprint import Envelope

np = pytest.importorskip("numpy")

_ORACLE = Path(__file__).resolve().parent.parent / "hack" / "sync-sim" / "dsp.py"
if not _ORACLE.exists():
    pytest.skip("hack/sync-sim is not in the sdist", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("sync_sim_dsp", _ORACLE)
dsp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dsp)

RATE = 24000


def _speechlike(seconds: float, seed: int = 7):
    """Noise shaped by syllable-length bursts and short pauses, as int16: an envelope with
    structure on the 100-300 ms scale, like speech."""
    rng = np.random.default_rng(seed)
    n = round(seconds * RATE)
    env = np.zeros(n)
    pos = 0
    while pos < n:
        syllable = round(rng.uniform(0.08, 0.25) * RATE)
        pause = round(rng.uniform(0.02, 0.12) * RATE)
        level = rng.uniform(0.05, 0.6)
        t = np.linspace(0, np.pi, syllable)
        env[pos : pos + syllable] = level * np.sin(t)[: n - pos]
        pos += syllable + pause
    return np.round(rng.standard_normal(n) * env * 32767 * 0.5).astype(np.int16)


def test_package_envelope_matches_the_oracle_bit_exactly():
    pcm = _speechlike(3.0)
    oracle = dsp.envelope_bytes(pcm, RATE)
    assert oracle.dtype == np.uint8
    assert len(oracle) == 599  # (3000 - 10) / 5 + 1 windows fully inside 3 s

    data = pcm.astype("<i2").tobytes()
    assert Envelope(RATE).feed(data) == oracle.tobytes()

    # The same values however the audio is cut, odd byte boundaries included.
    env = Envelope(RATE)
    pieces = [data[i : i + 1001] for i in range(0, len(data), 1001)]
    assert b"".join(env.feed(piece) for piece in pieces) == oracle.tobytes()


@pytest.mark.parametrize("rate", [16000, 22050, 24000, 44100, 48000])
def test_package_envelope_matches_the_oracle_at_other_rates_and_levels(rate):
    rng = np.random.default_rng(rate)
    for amplitude in (0, 1, 30, 1000, 32767):
        pcm = np.clip(np.round(rng.standard_normal(rate) * amplitude), -32768, 32767)
        pcm = pcm.astype(np.int16)
        data = pcm.astype("<i2").tobytes()
        expected = dsp.envelope_bytes(pcm, rate).tobytes()
        assert Envelope(rate).feed(data) == expected

        # Cut at random byte positions (odd ones split a sample): the values do not change.
        env = Envelope(rate)
        out, pos = [], 0
        while pos < len(data):
            step = int(rng.integers(1, 5000))
            out.append(env.feed(data[pos : pos + step]))
            pos += step
        assert b"".join(out) == expected


def test_oracle_quantisation_pins():
    assert dsp.quantise(np.array([-200.0, -96.0, 0.0, 12.0])).tolist() == [0, 0, 255, 255]
    assert dsp.envelope_bytes(np.zeros(RATE, dtype=np.int16), RATE).tolist() == [0] * 199


@pytest.mark.parametrize(
    ("db", "byte"),
    [
        # Exact ties on the byte scale (float arithmetic gives x.5 exactly). Half-up rounds them
        # up; round-half-even would give 42 and 212 for the odd-below ones, so these two tell the
        # rules apart. -48 dB (127.5) is a tie both rules send to 128: it pins nothing.
        (-80.0, 43),
        (-16.0, 213),
        (-48.0, 128),
    ],
)
def test_quantiser_rounds_ties_up_in_oracle_and_package(db, byte):
    assert dsp.quantise(np.array([db])).tolist() == [byte]
    assert fingerprint._to_byte(db) == byte


def _received(reference_pcm, delay_ms, lead_s=0.5, seconds=2.4):
    """The reference audio as a receiver sees it: a quiet lead, then the reference delayed by
    ``delay_ms``; returns the received envelope (dB) over the first ``seconds``."""
    lead = np.zeros(round(lead_s * RATE))
    delayed = dsp.frac_delay(np.concatenate([lead, reference_pcm.astype(float)]), delay_ms * 24)
    cut = delayed[: round(seconds * RATE)]
    return dsp.envelope_db(cut, RATE, 0.005, dc_block=True)


def test_oracle_locks_on_a_shifted_copy():
    pcm = _speechlike(3.0)
    reference = dsp.envelope_bytes(pcm, RATE)
    received = _received(pcm, 137.3)

    # The reference starts 0.5 s + 137.3 ms into the received track; the prior is 12 ms off.
    result = dsp.align(received, reference, prior_index=125)

    assert result.offset_ms == pytest.approx(637.3, abs=0.5)
    assert result.checks.accepted
    assert result.checks.speech and result.checks.window_ok
    assert result.checks.peak_ok and result.checks.ratio_ok and result.checks.sharpness_ok
    assert result.checks.halves_ok and result.checks.interior_ok
    assert result.checks.peak >= 0.95
    assert result.checks.ratio >= 1.2
    assert result.checks.sharpness >= 0.03
    assert result.checks.half_deviation_ms <= 10
    assert not result.checks.edge


def test_oracle_holds_on_silence():
    pcm = _speechlike(3.0)
    reference = dsp.envelope_bytes(pcm, RATE)

    quiet = dsp.envelope_db(np.zeros(2 * RATE), RATE, 0.005, dc_block=True)
    result = dsp.align(quiet, reference, prior_index=125)
    assert result.offset_ms is None
    assert not result.checks.speech and not result.checks.accepted

    # Speech on the track but a silent reference: nothing to correlate with.
    received = _received(pcm, 137.3)
    flat = dsp.envelope_bytes(np.zeros(3 * RATE, dtype=np.int16), RATE)
    assert dsp.align(received, flat, prior_index=125).offset_ms is None

    # Less than one window received: not enough to decide.
    assert dsp.align(received[:100], reference, prior_index=125).offset_ms is None


def test_oracle_holds_when_the_audio_is_not_the_reference():
    reference = dsp.envelope_bytes(_speechlike(3.0, seed=7), RATE)
    received = _received(_speechlike(3.0, seed=8), 137.3)
    assert dsp.align(received, reference, prior_index=125).offset_ms is None


def test_oracle_rejects_halves_that_disagree():
    pcm = _speechlike(3.0)
    reference = dsp.envelope_bytes(pcm, RATE)

    def delayed(delay_ms):
        return dsp.frac_delay(
            np.concatenate([np.zeros(round(0.5 * RATE)), pcm.astype(float)]), delay_ms * 24
        )

    # The audio jumps by 40 ms exactly at the middle of the last 1 s: the older half sits at
    # one offset, the newer half at another (a jitter-buffer insertion inside the window).
    total, cut = round(2.4 * RATE), round(1.9 * RATE)
    spliced = np.concatenate([delayed(137.3)[:cut], delayed(177.3)[cut:total]])
    received = dsp.envelope_db(spliced, RATE, 0.005, dc_block=True)

    result = dsp.align(received, reference, prior_index=125)

    assert result.offset_ms is None
    assert result.checks.speech
    assert not result.checks.halves_ok
    assert result.checks.half_deviation_ms > 10
    assert not result.checks.accepted


def test_oracle_rejects_a_peak_at_the_edge_of_the_search_range():
    pcm = _speechlike(3.0)
    reference = dsp.envelope_bytes(pcm, RATE)

    # The reference starts 932 ms into the track; the search covers 325 to 925 ms around the
    # prior at 625 ms, so the best lag is the last one inside the range, with the true peak just
    # beyond it. Everything else about the tick looks like a clean lock.
    received = _received(pcm, 432.0, seconds=3.0)
    result = dsp.align(received, reference, prior_index=125)

    assert result.offset_ms is None
    assert result.checks.edge
    assert result.checks.candidate_ms == pytest.approx(925.0)
    assert result.checks.peak_ok and result.checks.ratio_ok and result.checks.sharpness_ok
    assert result.checks.halves_ok
    assert not result.checks.interior_ok


@pytest.fixture
def sim(monkeypatch):
    """The simulation module (it imports ``dsp`` by name, so its directory goes on sys.path)."""
    monkeypatch.syspath_prepend(str(_ORACLE.parent))
    monkeypatch.delitem(sys.modules, "sim", raising=False)
    monkeypatch.delitem(sys.modules, "dsp", raising=False)
    return importlib.import_module("sim")


@pytest.mark.parametrize("delay_ms", [137.3, 211.9])
@pytest.mark.parametrize("other_audio", [False, True])
def test_align_equals_the_simulation_path(sim, delay_ms, other_audio):
    """``dsp.align`` is what RESULTS.md measured: the simulation's tick (``sim.tick_curves`` and
    ``sim.features``) gives the same peak, ratio, sharpness, half deviation, edge and estimate."""
    pcm = _speechlike(3.0)
    reference = dsp.envelope_bytes(pcm, RATE)
    heard = _speechlike(3.0, seed=8) if other_audio else pcm
    # A 1 s pre-roll puts lag 0 where the simulation expects the turn message to arrive.
    pre_s = sim.PRE
    received = _received(heard, delay_ms, lead_s=pre_s, seconds=3.2)
    prior_ms = 1.0e3 * pre_s + 150.0  # 150 ms after the message: inside the simulation's WIDE

    result = dsp.align(received, reference, prior_index=prior_ms / 5)
    c = result.checks
    assert c.speech

    # The simulation's path: one tick that sees every received value, three window lengths.
    hop = 0.005
    ticks = [100.0]
    xenv = [received]
    lags, full, g_full = sim.tick_curves([reference], xenv, hop, ticks, 1.0)
    _, older, g_older = sim.tick_curves([reference], xenv, hop, ticks, 0.5, 0.5)
    _, newer, g_newer = sim.tick_curves([reference], xenv, hop, ticks, 0.5, 0.0)
    curves = {"full": full, "h1": older, "h2": newer}
    gates = {"full": g_full, "h1": g_older, "h2": g_newer}
    prior_lag = prior_ms / 1e3 - pre_s
    peak, ratio, sharp, agree, _, est_interp, edge = sim.features(
        lags, curves, gates, 0, prior_lag - 0.3, prior_lag + 0.3
    )

    assert c.peak == pytest.approx(peak, abs=1e-9)
    assert c.ratio == pytest.approx(ratio, abs=1e-9)
    assert c.sharpness == pytest.approx(sharp, abs=1e-9)
    assert c.half_deviation_ms == pytest.approx(agree * 1e3, abs=1e-9)
    assert c.edge == edge
    assert c.candidate_ms == pytest.approx((est_interp + pre_s) * 1e3, abs=1e-9)
