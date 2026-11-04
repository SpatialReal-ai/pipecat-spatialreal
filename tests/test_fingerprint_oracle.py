"""The package's envelope against the normative oracle, ``hack/sync-sim/dsp.py``.

The oracle (numpy, dev only) is what the web SDK ports; the package computes the same bytes
without numpy. These tests pin the two together bit for bit and check the alignment rule on
the cases that must always hold.
"""

import importlib.util
from pathlib import Path

import pytest

from pipecat_spatialreal.fingerprint import Envelope

np = pytest.importorskip("numpy")

_ORACLE = Path(__file__).resolve().parent.parent / "hack" / "sync-sim" / "dsp.py"
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


@pytest.mark.parametrize("rate", [16000, 24000, 48000, 44100])
def test_package_envelope_matches_the_oracle_at_other_rates_and_levels(rate):
    rng = np.random.default_rng(rate)
    for amplitude in (0, 1, 30, 1000, 32767):
        pcm = np.clip(np.round(rng.standard_normal(rate) * amplitude), -32768, 32767)
        pcm = pcm.astype(np.int16)
        assert Envelope(rate).feed(pcm.astype("<i2").tobytes()) == (
            dsp.envelope_bytes(pcm, rate).tobytes()
        )


def test_oracle_quantisation_pins():
    assert dsp.quantise(np.array([-200.0, -96.0, 0.0, 12.0])).tolist() == [0, 0, 255, 255]
    # -48 dB is 127.5 on the byte scale: half rounds up.
    assert dsp.quantise(np.array([-48.0])).tolist() == [128]
    assert dsp.envelope_bytes(np.zeros(RATE, dtype=np.int16), RATE).tolist() == [0] * 199


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
