import math
import struct

import pytest

from pipecat_spatialreal.fingerprint import Envelope

RATE = 24000

# 1 s holds (1000 - 10) / 5 + 1 = 199 windows that lie fully inside it: window 199 would end at
# 1005 ms.
ONE_SECOND = 199


def _pcm(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _ms(rate: int, ms: float) -> int:
    return round(ms * rate / 1000)


def _expected_byte(samples: list[int]) -> int:
    """The Global Constraints formula, applied to one window of samples."""
    mean_square = sum(s * s for s in samples) / len(samples)
    db = 10 * math.log10(mean_square / 32768**2 + 1e-10)
    db = max(-96.0, min(0.0, db))
    return math.floor((db + 96) / 96 * 255 + 0.5)


def test_silence_is_zero():
    env = Envelope(RATE)
    out = env.feed(_pcm([0] * RATE))
    assert out == bytes(ONE_SECOND)
    assert env.index == ONE_SECOND


def test_full_scale_square_is_255():
    env = Envelope(RATE)
    out = env.feed(_pcm([32767 if (i // 12) % 2 == 0 else -32767 for i in range(RATE)]))
    assert len(out) == ONE_SECOND
    assert all(abs(v - 255) <= 1 for v in out)


def test_minus_20_db_sine_maps_to_the_formula():
    # A sine of amplitude A has mean square A^2 / 2; amplitude 32768 * sqrt(2) * 0.1 is -20 dBFS.
    amp = 32768 * math.sqrt(2) * 0.1
    samples = [round(amp * math.sin(2 * math.pi * 1000 * i / RATE)) for i in range(RATE)]
    env = Envelope(RATE)
    out = env.feed(_pcm(samples))
    hop, win = _ms(RATE, 5), _ms(RATE, 10)
    assert len(out) == ONE_SECOND
    for k, value in enumerate(out):
        assert value == _expected_byte(samples[k * hop : k * hop + win])
    # Sanity: about -20 dB maps to (76 / 96) * 255 = 201.9.
    assert all(abs(v - 202) <= 1 for v in out)


def test_chunking_does_not_change_values():
    samples = [round(8000 * math.sin(i * 0.05) + (i % 97) * 40) for i in range(RATE)]
    wire = _pcm(samples)
    whole = Envelope(RATE)
    expected = whole.feed(wire)
    pieces = Envelope(RATE)
    got = b"".join(pieces.feed(wire[i : i + 37]) for i in range(0, len(wire), 37))
    assert got == expected
    assert pieces.index == whole.index == ONE_SECOND


def test_windows_only_fully_inside():
    env = Envelope(RATE)
    assert len(env.feed(_pcm([1000] * _ms(RATE, 12)))) == 1
    assert env.index == 1
    assert len(env.feed(_pcm([1000] * _ms(RATE, 3)))) == 1  # now 15 ms in total
    assert env.index == 2
    # Under one window: nothing yet.
    assert Envelope(RATE).feed(_pcm([1000] * (_ms(RATE, 10) - 1))) == b""


@pytest.mark.parametrize("rate", [16000, 24000])
def test_keyframe_index(rate):
    # A loud burst that covers exactly [j * 40 ms, j * 40 ms + 10 ms) must peak at index 8 * j:
    # the neighbours overlap it by half a window and sit about 3 dB lower.
    for j in range(25):
        samples = [0] * rate
        start = _ms(rate, 40 * j)
        for i in range(start, start + _ms(rate, 10)):
            samples[i] = 32767 if i % 2 == 0 else -32767
        out = Envelope(rate).feed(_pcm(samples))
        peak = max(out)
        assert out.index(peak) == 8 * j
        assert out[8 * j] == 255
        for k in (8 * j - 1, 8 * j + 1):
            if 0 <= k < len(out):
                assert out[k] < 255


def test_odd_byte_split_and_reset():
    wire = _pcm([i * 7 % 3000 for i in range(RATE // 10)])
    whole = Envelope(RATE)
    expected = whole.feed(wire)
    split = Envelope(RATE)
    got = split.feed(wire[:1]) + split.feed(wire[1:5]) + split.feed(wire[5:])
    assert got == expected
    split.reset()
    assert split.index == 0
    assert split.feed(wire) == expected


@pytest.mark.parametrize("rate", [16000, 22050, 24000, 32000, 44100, 48000])
def test_every_supported_rate_yields_values(rate):
    out = Envelope(rate).feed(_pcm([500] * rate))
    hop, win = 5 * rate // 1000, 10 * rate // 1000  # truncated, see the module docstring
    assert len(out) == (rate - win) // hop + 1
