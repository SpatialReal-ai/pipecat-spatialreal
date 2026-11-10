"""Audio mode: the bot's audio also goes to the client as ``audio`` messages (pipecat/v1).

A client that cannot tap the bot's audio track asks for it with an ``audio_request`` client
message. These tests run the processor with the fake driveningress, send that request as Pipecat's
RTVI processor would (an ``RTVIClientMessageFrame`` going downstream), and decode the ``audio``
messages back into PCM to check them against what inference saw.
"""

import asyncio
import base64
import io

import av
import numpy as np
import pytest
from conftest import Do, audio_at_driveningress, link_up, message_pushed, play
from pipecat.frames.frames import InterruptionFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frameworks.rtvi import RTVIClientMessageFrame, RTVIServerMessageFrame
from pipecat.tests.utils import SleepFrame
from test_processor import kinds, make_proc, messages, record_pushes, req_of, tone_frame

from pipecat_spatialreal import SpatialRealAvatarProcessor
from pipecat_spatialreal.driven import DrivenIngress
from pipecat_spatialreal.envelope import MAX_CHUNK_BYTES, TAG


@pytest.fixture(autouse=True)
def _every_test_checks_the_log(pipecat_log):
    """Every test here fails on a pipeline error or a task left running (see ``pipecat_log``)."""


def audio_request():
    """The client's request for audio mode, as Pipecat's RTVI processor pushes it."""
    return RTVIClientMessageFrame(
        msg_id="m1", type="spatialreal-avatar", data={"tag": TAG, "audio_request": {}}
    )


def audio_of(frames, turn_id):
    """The ``audio`` payloads of one turn, in the order they were pushed."""
    return [a for a in messages(frames, "audio") if a["turn_id"] == turn_id]


def stream_of(frames, turn_id):
    """The turn's Ogg Opus stream, its ``audio`` pieces concatenated."""
    return b"".join(base64.b64decode(a["data"]) for a in audio_of(frames, turn_id))


def decode(stream):
    """The stream's audio as 48 kHz PCM16 samples (the decoder trims pre-skip and padding)."""
    container = av.open(io.BytesIO(stream), format="ogg")
    chunks = [frame.to_ndarray().reshape(-1) for frame in container.decode(audio=0)]
    pcm = np.concatenate(chunks)
    if pcm.dtype.kind == "f":
        pcm = np.round(pcm * 32767)
    return pcm.astype(np.int16)


def best_correlation(decoded, original):
    """Correlation at the best small lag (the codec shifts the signal by a few samples)."""
    n = min(len(decoded), len(original)) - 400
    return max(
        float(np.corrcoef(decoded[lag : lag + n].astype(float), original[:n].astype(float))[0, 1])
        for lag in range(0, 400, 2)
    )


def turn_end_of(frames, turn_id):
    (end,) = [m for m in messages(frames, "turn_end") if m["turn_id"] == turn_id]
    return end


def turn_ids(frames):
    return [m["turn_id"] for m in messages(frames, "turn_start")]


async def test_without_a_request_nothing_changes(fake):
    down, _ = await play(
        make_proc(fake),
        [TTSStartedFrame(), tone_frame(400), TTSStoppedFrame(), SleepFrame(0.8)],
    )
    k = kinds(down)
    assert ("rtvi", "audio") not in k and ("rtvi", "audio_mode") not in k


async def test_a_requested_turn_streams_its_audio_ahead_of_the_room(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            audio_request(),
            TTSStartedFrame(),
            *[tone_frame(40) for _ in range(25)],  # one second, at TTS speed
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
    )
    turn_id = req_of(down)
    k = kinds(down)
    assert k.count(("rtvi", "audio_mode")) == 1
    assert k.index(("rtvi", "audio_mode")) < k.index(("rtvi", "audio"))
    assert messages(down, "audio_mode") == [
        {"active": True, "first_turn_id": turn_id, "sample_rate": 24000}
    ]
    pieces = audio_of(down, turn_id)
    assert [a["seq"] for a in pieces] == list(range(len(pieces)))
    assert [a["end"] for a in pieces] == [False] * (len(pieces) - 1) + [True]
    assert all(len(base64.b64decode(a["data"])) <= MAX_CHUNK_BYTES for a in pieces)
    stream = stream_of(down, turn_id)
    assert stream[:4] == b"OggS" and b"OpusHead" in stream[:64]
    # Exactly the turn's audio: as long as turn_end says, and the samples inference got.
    decoded = decode(stream)
    assert len(decoded) == round(turn_end_of(down, turn_id)["pcm_ms"] * 48)
    sent = np.frombuffer(fake.received_audio_bytes(turn_id), dtype="<i2")
    assert best_correlation(decoded[::2], sent) > 0.9  # 48 kHz back to inference's 24 kHz
    # Not paced: the whole stream is out well before the room has played the turn.
    times = {id(f): t for t, f in pushed}
    last_audio = max(
        times[id(f)] for f in down if isinstance(f, RTVIServerMessageFrame) and "audio" in f.data
    )
    end_at = times[id(next(f for f in down if isinstance(f, TTSStoppedFrame)))]
    assert last_audio < end_at - 0.5


async def test_stall_silence_is_part_of_the_audio(fake):
    down, _ = await play(
        make_proc(fake),
        [
            audio_request(),
            TTSStartedFrame(),
            tone_frame(200),
            SleepFrame(0.8),  # the TTS falls behind: the room plays stall silence meanwhile
            tone_frame(200),
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
    )
    turn_id = req_of(down)
    pcm_ms = turn_end_of(down, turn_id)["pcm_ms"]
    assert pcm_ms > 600  # the stall silence counts into the turn...
    assert len(decode(stream_of(down, turn_id))) == round(pcm_ms * 48)  # ...and into its audio


async def test_a_request_during_a_turn_starts_with_the_next_quiet_turn(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            TTSStartedFrame(),
            tone_frame(600),
            audio_at_driveningress(fake),
            audio_request(),  # the room is busy with this turn: it stays on the track only
            TTSStoppedFrame(),
            # The next turn starts while the first still plays: the client could not leave the
            # track without cutting the first one off, so this one stays on the track too.
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=2),
            # Only a turn that starts in quiet begins audio mode.
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=3),
        ],
    )
    first, second, third = turn_ids(down)
    assert audio_of(down, first) == [] and audio_of(down, second) == []
    assert audio_of(down, third)
    assert [m["first_turn_id"] for m in messages(down, "audio_mode")] == [third]


async def test_every_request_gets_its_own_answer(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            audio_request(),
            TTSStartedFrame(),
            tone_frame(200),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
            audio_request(),  # a client that rejoined, say
            TTSStartedFrame(),
            tone_frame(200),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=2),
        ],
    )
    first, second = turn_ids(down)
    assert [m["first_turn_id"] for m in messages(down, "audio_mode")] == [first, second]
    assert audio_of(down, first) and audio_of(down, second)


async def test_a_turn_without_animation_still_gets_its_audio(fake):
    proc = make_proc(fake, reconnect_backoff_s=(30,))
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            Do(lambda: fake.close_with(4001)),
            message_pushed(pushed, "state"),  # degraded from here on
            audio_request(),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
    )
    (start,) = messages(down, "turn_start")
    assert start["degraded"] is True
    pcm_ms = turn_end_of(down, start["turn_id"])["pcm_ms"]
    assert len(decode(stream_of(down, start["turn_id"]))) == round(pcm_ms * 48)


async def test_audio_waits_for_the_turn_id(fake, monkeypatch):
    real_send = DrivenIngress.send
    calls = []

    async def slow_first_send(self, pcm, *, end=False):
        calls.append(len(pcm))
        if len(calls) == 1:
            await asyncio.sleep(0.4)  # the request id comes back late
        return await real_send(self, pcm, end=end)

    monkeypatch.setattr(DrivenIngress, "send", slow_first_send)
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            audio_request(),
            TTSStartedFrame(),
            tone_frame(400),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
    )
    turn_id = req_of(down)
    pieces = audio_of(down, turn_id)
    assert pieces[0]["seq"] == 0 and [a["seq"] for a in pieces] == list(range(len(pieces)))
    assert len(decode(stream_of(down, turn_id))) == round(turn_end_of(down, turn_id)["pcm_ms"] * 48)


async def test_a_rate_opus_lacks_is_sent_at_48_khz(fake):
    rate = 22050
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            audio_request(),
            TTSStartedFrame(),
            tone_frame(400, rate=rate),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
        params=PipelineParams(audio_out_sample_rate=rate),
    )
    turn_id = req_of(down)
    assert messages(down, "audio_mode")[0]["sample_rate"] == 48000
    expected = turn_end_of(down, turn_id)["pcm_ms"] * 48
    assert abs(len(decode(stream_of(down, turn_id))) - expected) <= 48  # within 1 ms


async def test_a_turn_that_loses_its_animation_says_so_once(fake):
    fake.first_segment_ms = 120  # turn_start comes early
    fake.segment_ms = 120
    proc = make_proc(fake, reconnect_backoff_s=(30,))
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(200),
            message_pushed(pushed, "turn_start"),
            Do(lambda: fake.close_with(4001)),  # the animation source goes mid-turn
            message_pushed(pushed, "state"),
            tone_frame(400),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
    )
    turn_id = req_of(down)
    k = kinds(down)
    assert messages(down, "turn_audio_only") == [{"turn_id": turn_id}]
    assert k.index(("rtvi", "turn_audio_only")) < k.index(("rtvi", "turn_end"))


async def test_an_unannounced_turn_that_sent_audio_is_aborted(fake):
    fake.first_segment_ms = 10_000  # no animation comes back: the turn is never announced
    proc = make_proc(fake, hold_cap_s=30)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            audio_request(),
            TTSStartedFrame(),
            tone_frame(200),
            message_pushed(pushed, "audio"),
            InterruptionFrame(),
            SleepFrame(0.3),
        ],
    )
    (turn_id,) = {a["turn_id"] for a in messages(down, "audio")}
    assert messages(down, "turn_start") == []
    assert messages(down, "turn_abort") == [{"turn_id": turn_id}]
    k = kinds(down)
    abort = k.index(("rtvi", "turn_abort"))
    assert ("rtvi", "audio") not in k[abort:]


async def test_failing_open_ends_audio_mode_first(fake, monkeypatch, pipecat_log):
    async def fault(self, turn):
        raise RuntimeError("a bug")

    monkeypatch.setattr(SpatialRealAvatarProcessor, "_forward_segments", fault)
    proc = make_proc(fake)
    down, _ = await play(
        proc,
        [
            audio_request(),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(0.5),
            audio_request(),  # after failing open: refused, the client stays on the track
            SleepFrame(0.2),
        ],
    )
    k = kinds(down)
    state = k.index(("rtvi", "state"))
    modes = messages(down, "audio_mode")
    assert [m["active"] for m in modes] == [True, False, False]
    # The first "off" goes before anything else the failure pushes.
    first_off = [i for i, kind in enumerate(k) if kind == ("rtvi", "audio_mode")][1]
    assert first_off < state
    failures = [r for r in pipecat_log if "release failed" in r["message"]]
    assert len(failures) == 1
    pipecat_log[:] = [r for r in pipecat_log if r not in failures]  # expected, not a test error
