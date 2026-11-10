import math
import struct

import pytest

from pipecat_spatialreal.onset import OnsetTracker


def pcm16(*samples):
    """PCM16 little-endian mono bytes of the given sample values."""
    return struct.pack(f"<{len(samples)}h", *samples)


def silence(ms, sr):
    """``ms`` milliseconds of digital silence."""
    return bytes(2 * (sr * ms // 1000))


def tone(ms, sr, amp=9800, freq=440):
    """``ms`` milliseconds of a cosine wave.

    It starts at its peak: the first sample is ``amp``, so the tone is audible from its very first
    sample and an onset lands exactly where the tone begins (a sine, which starts at 0, would put
    it one sample later).
    """
    count = sr * ms // 1000
    return pcm16(*(round(amp * math.cos(2 * math.pi * freq * i / sr)) for i in range(count)))


def test_leading_silence_then_tone_gives_one_onset():
    t = OnsetTracker(24000)
    assert t.feed(silence(400, 24000) + tone(1000, 24000)) == [400.0]


def test_gap_longer_than_min_silence_reanchors():
    t = OnsetTracker(24000, min_silence_ms=150)
    pcm = silence(400, 24000) + tone(1000, 24000) + silence(300, 24000) + tone(500, 24000)
    assert t.feed(pcm) == [400.0, 1700.0]


def test_short_gap_is_not_an_onset():
    t = OnsetTracker(24000, min_silence_ms=150)
    assert t.feed(tone(500, 24000) + silence(100, 24000) + tone(500, 24000)) == [0.0]


def test_chunked_feed_matches_whole():
    pcm = silence(400, 24000) + tone(1000, 24000) + silence(300, 24000) + tone(500, 24000)
    whole = OnsetTracker(24000).feed(pcm)
    t = OnsetTracker(24000)
    found = []
    for i in range(0, len(pcm), 1920):  # 40 ms pieces
        found += t.feed(pcm[i : i + 1920])
    assert found == whole and t.fed_ms == 2200.0


# --- Rule details the tests above leave unpinned. The browser applies the same
# --- rule to the same audio, so each boundary here is part of the contract.


def test_nothing_audible_gives_no_onset():
    t = OnsetTracker(24000)
    assert t.feed(silence(2000, 24000)) == []
    assert t.fed_ms == 2000.0


def test_leading_silence_shorter_than_min_silence_still_counts():
    # The start of the stream counts as a long enough silence, however short it is.
    t = OnsetTracker(24000, min_silence_ms=150)
    assert t.feed(silence(50, 24000) + tone(500, 24000)) == [50.0]


@pytest.mark.parametrize("value", [328, -328])
def test_a_sample_at_the_threshold_is_silent(value):
    assert OnsetTracker(24000).feed(silence(400, 24000) + pcm16(value) * 10) == []


@pytest.mark.parametrize("value", [329, -329, 32767, -32768])
def test_a_sample_above_the_threshold_is_audible(value):
    assert OnsetTracker(24000).feed(silence(400, 24000) + pcm16(value)) == [400.0]


def test_threshold_is_configurable():
    t = OnsetTracker(24000, threshold=1000)
    # 24 samples (1 ms) at exactly the threshold are silent; the 25th is the first audible one.
    assert t.feed(pcm16(*[1000, -1000] * 12, 1001)) == [1.0]


@pytest.mark.parametrize("min_silence_ms", [40, 150, 300])
def test_gap_of_exactly_min_silence_reanchors_one_sample_less_does_not(min_silence_ms):
    gap = 24 * min_silence_ms  # samples in min_silence_ms at 24 kHz
    burst = pcm16(*[5000] * 24)  # 1 ms
    exact = OnsetTracker(24000, min_silence_ms=min_silence_ms)
    assert exact.feed(burst + pcm16(*[0] * gap) + burst) == [0.0, 1.0 + min_silence_ms]
    short = OnsetTracker(24000, min_silence_ms=min_silence_ms)
    assert short.feed(burst + pcm16(*[0] * (gap - 1)) + burst) == [0.0]


def test_pauses_that_are_each_short_never_add_up():
    # The silence has to be consecutive: any audible sample ends the run, even one that is not an
    # onset. Here the two pauses total 4000 samples (167 ms), more than the 150 ms minimum.
    t = OnsetTracker(24000)
    burst = pcm16(5000)
    pause = pcm16(*[0] * 2000)  # 83 ms, shorter than the 150 ms minimum
    assert t.feed(burst + pause + burst + pause + burst) == [0.0]


@pytest.mark.parametrize(
    ("min_silence_ms", "needed"),
    [
        # 150 ms at 22.05 kHz is 3307.5 samples: 3307 last 149.97 ms, too short; 3308 are enough.
        pytest.param(150, 3308, id="half-a-sample-over"),
        # 161 ms at 22.05 kHz is 3550.05 samples: 3550 last 160.998 ms, too short; 3551 are enough.
        pytest.param(161, 3551, id="a-twentieth-of-a-sample-over"),
    ],
)
def test_min_silence_rounds_up_to_whole_samples(min_silence_ms, needed):
    burst = pcm16(5000)

    def onsets(zeros):
        tracker = OnsetTracker(22050, min_silence_ms=min_silence_ms)
        return tracker.feed(burst + pcm16(*[0] * zeros) + burst)

    assert onsets(needed - 1) == [0.0]
    assert onsets(needed) == [0.0, (needed + 1) * 1000 / 22050]


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 24000, 44100, 48000])
def test_offsets_are_in_ms_whatever_the_sample_rate(sample_rate):
    t = OnsetTracker(sample_rate)
    pcm = silence(400, sample_rate) + tone(100, sample_rate)
    assert t.feed(pcm) == [400.0]
    assert t.fed_ms == 500.0


def test_onset_on_a_keyframe_boundary_is_exact():
    # 192960 samples at 24 kHz are exactly 8040 ms = keyframe 201, and the browser places the
    # animation at floor(onset_ms * 25 / 1000). Computed as samples / rate * 1000 the offset is
    # 8039.999999999999, which would land on keyframe 200.
    (onset,) = OnsetTracker(24000).feed(bytes(2 * 192960) + pcm16(5000))
    assert onset == 8040.0
    assert math.floor(onset * 25 / 1000) == 201


@pytest.mark.parametrize("size", [1, 2, 3, 7, 480, 1001, 1920])
def test_any_chunk_size_gives_the_same_onsets(size):
    audio = silence(20, 24000) + tone(10, 24000) + silence(160, 24000) + tone(10, 24000)
    assert OnsetTracker(24000).feed(audio) == [20.0, 190.0]
    t = OnsetTracker(24000)
    found = []
    for i in range(0, len(audio), size):
        found += t.feed(audio[i : i + size])
    assert found == [20.0, 190.0]
    assert t.fed_ms == 200.0


def test_a_trailing_odd_byte_waits_for_the_next_feed():
    audio = silence(400, 24000) + tone(10, 24000)
    cut = 2 * 9600 + 1  # inside the first audible sample: only its low byte is in the first piece
    t = OnsetTracker(24000)
    assert t.feed(audio[:cut]) == []
    assert t.fed_ms == 400.0  # the half sample is not counted yet
    assert t.feed(audio[cut:]) == [400.0]
    assert t.fed_ms == 410.0


def test_a_lone_byte_does_not_crash():
    t = OnsetTracker(24000)
    assert t.feed(b"\x01") == []
    assert t.fed_ms == 0.0


def test_empty_feed_changes_nothing():
    t = OnsetTracker(24000)
    assert t.feed(b"") == []
    assert t.fed_ms == 0.0
    assert t.feed(pcm16(5000)) == [0.0]  # still armed, still counting from sample 0


def test_samples_are_little_endian():
    # Two bytes 02 00 read little-endian are 2 (silent); read big-endian they would be 512.
    assert OnsetTracker(24000).feed(b"\x02\x00") == []
    assert OnsetTracker(24000).feed(b"\x00\x02") == [0.0]


@pytest.mark.parametrize(
    ("args", "kwargs", "name"),
    [
        pytest.param((0,), {}, "sample_rate", id="zero-rate"),
        pytest.param((-24000,), {}, "sample_rate", id="negative-rate"),
        pytest.param((24000,), {"threshold": -1}, "threshold", id="negative-threshold"),
        pytest.param((24000,), {"min_silence_ms": 0}, "min_silence_ms", id="zero-min-silence"),
        pytest.param(
            (24000,), {"min_silence_ms": -150}, "min_silence_ms", id="negative-min-silence"
        ),
    ],
)
def test_nonsense_settings_are_refused_up_front(args, kwargs, name):
    with pytest.raises(ValueError, match=name):
        OnsetTracker(*args, **kwargs)
