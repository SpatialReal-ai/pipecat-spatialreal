import math
import struct

import pytest

from pipecat_spatialreal import fingerprint
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


def test_reset_drops_a_pending_odd_byte():
    even = _pcm([i * 11 % 4000 for i in range(RATE // 20)])
    fresh = Envelope(RATE)
    expected = fresh.feed(even)
    env = Envelope(RATE)
    env.feed(b"\x01\x02\x03")  # odd length: one whole sample, the third byte stays pending
    env.reset()
    assert env.index == 0
    assert env.feed(even) == expected
    assert env.index == fresh.index


def _window_with_mean_square_near(target: float, samples: int) -> tuple[list[int], list[int]]:
    """Two sample windows whose mean squares straddle ``target`` as closely as the grid allows.

    ``a = floor(sqrt(target))`` for all samples and ``a + 1`` for ``k`` of them gives
    ``S(k) / samples = a^2 + k * (2a + 1) / samples``, a grid of steps ``(2a + 1) / samples``.
    Returns the windows for the largest ``k`` still below ``target`` and the smallest ``k`` above.
    """
    a = math.isqrt(int(target))
    k = next(
        k for k in range(samples + 1) if (samples * a * a + k * (2 * a + 1)) / samples > target
    )
    assert 1 <= k, "target must be at least a^2 + one step"

    def window(count: int) -> list[int]:
        return [a + 1] * count + [a] * (samples - count)

    return window(k - 1), window(k)


@pytest.mark.parametrize("b", [0, 1, 100, 254])
def test_rounding_is_half_up_around_x_point_5(b):
    # The byte is b + 1 exactly when (db + 96) / 96 * 255 >= b + 0.5, i.e. from
    #   db_b = (b + 0.5) * 96 / 255 - 96
    # upwards. Inverting 10 * log10(ms / 32768^2 + 1e-10) = db_b gives the mean square where the
    # byte flips:
    #   ms_b = 32768^2 * (10^(db_b / 10) - 1e-10)
    # e.g. b = 100: db_b = 100.5 * 96 / 255 - 96 = -58.1647, ms_b = 32768^2 * (1.5265e-6 - 1e-10)
    # = 1639.7. A 4800-sample window (100 ms at 48 kHz) makes the grid of reachable mean squares
    # fine enough (about 0.01 byte) to sit just below and just above ms_b.
    rate, window_ms = 48000, 100
    samples = rate * window_ms // 1000
    db_b = (b + 0.5) * 96 / 255 - 96
    ms_b = 32768**2 * (10 ** (db_b / 10) - 1e-10)
    below, above = _window_with_mean_square_near(ms_b, samples)
    assert Envelope(rate, window_ms=window_ms).feed(_pcm(below)) == bytes([b])
    assert Envelope(rate, window_ms=window_ms).feed(_pcm(above)) == bytes([b + 1])


def _db_whose_x_is_exactly(tie: float) -> float:
    """A float db for which the mapping's arithmetic yields exactly ``tie``, or ``nan``."""
    start = tie * 96 / 255 - 96
    for ulps in range(-40, 41):
        db = start
        for _ in range(abs(ulps)):
            db = math.nextafter(db, math.inf if ulps > 0 else -math.inf)
        if (db + 96.0) / 96.0 * 255 == tie:
            return db
    return math.nan


def test_rounding_at_an_exact_tie_is_half_up():
    # x = (db + 96) / 96 * 255. db = -48 gives exactly 127.5 and must map to 128, but 127.5 rounds
    # up under half-even too (128 is even). The ties n + 0.5 with even n tell the two apart, and
    # for some of them a float db exists whose x comes out exactly n + 0.5; half-up gives n + 1,
    # half-even n.
    assert fingerprint._to_byte(-48.0) == 128
    for n in (10, 20, 32, 42, 54, 64):
        db = _db_whose_x_is_exactly(n + 0.5)
        assert not math.isnan(db), f"no float db maps to exactly x = {n + 0.5}"
        assert fingerprint._to_byte(db) == n + 1


def test_hop_longer_than_window_is_rejected():
    with pytest.raises(ValueError, match="hop_ms"):
        Envelope(RATE, hop_ms=20, window_ms=10)
    Envelope(RATE, hop_ms=10, window_ms=10)  # equal is fine: windows then do not overlap
