import asyncio
import math
import statistics
import struct

import pytest
from conftest import (
    Do,
    Until,
    audio_at_driveningress,
    interruption_pushed,
    link_up,
    message_pushed,
    play,
    tts_audio_released,
)
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_spatialreal import SpatialRealAvatarProcessor
from pipecat_spatialreal.driven import DrivenIngress


def make_proc(fake, **kw):
    return SpatialRealAvatarProcessor(
        api_key="k",
        app_id="a",
        avatar_id="av",
        console_endpoint_url=fake.console_url,
        ingress_endpoint_url=fake.ingress_url,
        **kw,
    )


def sine(ms, rate, freq=440):
    """``ms`` of a sine at amplitude 9800, as PCM16 sample values."""
    return [round(9800 * math.sin(2 * math.pi * freq * i / rate)) for i in range(rate * ms // 1000)]


def pcm16(samples):
    return struct.pack(f"<{len(samples)}h", *samples)


def tone_frame(ms, rate=24000, **kw):
    """``ms`` of a 440 Hz tone at amplitude 9800, as the TTS would push it."""
    return TTSAudioRawFrame(audio=pcm16(sine(ms, rate)), sample_rate=rate, num_channels=1, **kw)


def kinds(frames):
    """Each frame as ``("rtvi", <payload member>)`` for an avatar message, else its type name."""
    out = []
    for f in frames:
        if isinstance(f, RTVIServerMessageFrame):
            (member,) = [key for key in f.data if key != "tag"]
            out.append(("rtvi", member))
        else:
            out.append(type(f).__name__)
    return out


def req_of(frames):
    """The ``turn_id`` of the first ``turn_start``."""
    return messages(frames, "turn_start")[0]["turn_id"]


def messages(frames, member):
    """The payloads of the avatar messages of one kind, in order."""
    return [
        f.data[member] for f in frames if isinstance(f, RTVIServerMessageFrame) and member in f.data
    ]


def record_pushes(proc):
    """Every frame ``proc`` pushes, with the loop time it pushed it at, in order.

    Unlike the sink, this also sees what is pushed after a ``CancelFrame``, which the processors
    downstream drop.
    """
    pushed = []

    @proc.event_handler("on_after_push_frame")
    async def _record(processor, frame):
        pushed.append((asyncio.get_running_loop().time(), frame))

    return pushed


def record_segments(proc):
    """``(loop time, req_id)`` of every segment ``proc`` receives, in order."""
    received = []
    receive = proc._on_segment

    def recorded(segment):
        received.append((asyncio.get_running_loop().time(), segment.req_id))
        receive(segment)

    proc._on_segment = recorded  # setup() hands it to the driveningress client
    return received


@pytest.fixture(autouse=True)
def pipecat_log():
    """Fail a test whose pipeline reported an error or left a task running.

    Pipecat logs, rather than raises, what goes wrong inside a processor's tasks: an exception
    in a background task, an error pushed by ``process_frame``, a task still running at teardown.
    The one error every test expects is the setup notice that RTVI is off, which ``run_test``
    does on purpose.
    """
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="WARNING")
    yield records
    logger.remove(sink)
    errors = [
        r["message"]
        for r in records
        if (r["level"].no >= 40 and "RTVI" not in r["message"]) or "dangling" in r["message"]
    ]
    assert errors == []


async def test_audio_is_held_until_first_segment(fake):
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            TTSStartedFrame(),
            *[tone_frame(40) for _ in range(10)],
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
        expected_down_frames=None,  # we assert on order ourselves
    )
    k = kinds(down)
    assert (
        k.index(("rtvi", "turn_start")) < k.index(("rtvi", "frames")) < k.index("TTSAudioRawFrame")
    )
    assert k[-1] == ("rtvi", "turn_end") and k.count("TTSAudioRawFrame") == 10
    assert fake.received_audio_ms(req_of(down)) == 400


async def test_released_audio_is_resampled_to_out_rate(fake):
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            TTSStartedFrame(),
            tone_frame(400, rate=22050),
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
    )
    rel = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert all(f.sample_rate == 24000 for f in rel)
    assert abs(sum(len(f.audio) for f in rel) / 2 / 24000 * 1000 - 400) <= 1
    assert fake.received_audio_bytes(req_of(down)) == b"".join(f.audio for f in rel)


async def test_stall_mirrors_silence(fake):
    proc = make_proc(fake, lead_ms=0)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(300),
            tts_audio_released(pushed),  # the room plays it for 300 ms from here
            SleepFrame(0.8),  # the TTS falls ~500 ms behind the room
            tone_frame(200),
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
    )
    (first, _), (second, _) = [(t, f) for t, f in pushed if isinstance(f, TTSAudioRawFrame)]
    stall = second - (first + 0.3)  # how long the room waited for the TTS
    sil = [f for f in down if type(f) is OutputAudioRawFrame]
    assert stall > 0.4 and all(set(f.audio) == {0} for f in sil)
    # Silence went out for the stall, one 40 ms chunk at a time and just in time (the lead is 0):
    # no more than the stall, and a good part of it however late the loop woke up.
    assert 4 <= len(sil) and len(sil) * 0.04 <= stall + 0.05
    assert fake.received_audio_ms(req_of(down)) == 500 + 40 * len(sil)


async def test_a_stall_is_not_stretched_by_the_lead(fake):
    proc = make_proc(fake)  # lead_ms 200, silence_chunk_ms 40
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(300),
            tts_audio_released(pushed),  # the room plays it for 300 ms from here
            SleepFrame(0.6),  # the TTS falls ~300 ms behind the room
            tone_frame(500),
            TTSStoppedFrame(),
            SleepFrame(1.2),
        ],
    )
    (first, _), (second, _) = [(t, f) for t, f in pushed if isinstance(f, TTSAudioRawFrame)]
    silence = [t for t, f in pushed if type(f) is OutputAudioRawFrame and first <= t <= second]
    stall = second - (first + 0.3)  # how long the room waited for the TTS
    inserted = len(silence) * 0.04
    assert stall > 0.2 and silence
    # The silence never runs further ahead of the room than one chunk plus the chunk kept queued so
    # the transport never runs dry; the 200 ms lead is not added on top. This bound holds on any
    # machine: a loop that wakes late can only make the silence shorter.
    assert inserted - stall <= 0.04 + 0.04 + 0.01
    # And it bridges the stall, a good part of it at least. How closely depends on how promptly the
    # loop wakes, which is the machine's business: while the process is not scheduled nothing can
    # be pushed, and the stall measured here grows by as much as that.
    assert inserted >= stall / 3
    # The chunk kept queued: each chunk is pushed while one chunk (40 ms) of what the room was
    # given is still to play, not when the room has run dry. Rebuild how much that was at each push
    # (the end of what was queued, less the time of the push). A late wake-up lowers its own
    # sample, and can leave the room dry; the median of the samples is what carries the pin.
    ends, left = first + 0.3, []
    for t in silence:
        left.append(ends - t)
        ends = max(ends, t) + 0.04
    assert statistics.median(left) >= 0.02
    assert fake.received_audio_ms(req_of(down)) == 800 + 40 * len(silence)


async def test_interruption_before_first_segment(fake):
    fake.first_segment_ms = 10_000
    proc = make_proc(fake)
    down, _ = await play(
        proc,
        [
            TTSStartedFrame(),
            tone_frame(100),
            # Frames queued together reach the processor's input queue together, and a system
            # frame there is taken before data frames: the interruption must come after the
            # turn exists, and after its audio has been sent for the interrupt to stop.
            audio_at_driveningress(fake),
            InterruptionFrame(),
            SleepFrame(0.5),
        ],
    )
    assert not [f for f in down if isinstance(f, TTSAudioRawFrame)]
    # A turn interrupted before its turn_start is never announced at all, so it gets no
    # turn_abort either (pipecat/v1/avatar.proto, TurnStart).
    k = kinds(down)
    assert ("rtvi", "turn_start") not in k and ("rtvi", "turn_abort") not in k
    assert "InterruptionFrame" in k
    assert fake.received[-1].WhichOneof("data") == "client_interrupt"


async def test_short_turn_uses_end_flush(fake):
    fake.first_segment_ms = 1000
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[TTSStartedFrame(), tone_frame(120), TTSStoppedFrame(), SleepFrame(0.8)],
    )
    k = kinds(down)
    assert k.index(("rtvi", "frames")) < k.index("TTSAudioRawFrame") and ("rtvi", "turn_end") in k


async def test_back_to_back_turns_chain(fake):
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(1.5),
        ],
    )
    starts = [
        f.data["turn_start"]
        for f in down
        if isinstance(f, RTVIServerMessageFrame) and "turn_start" in f.data
    ]
    assert starts[1]["prev_turn_id"] == starts[0]["turn_id"]
    assert starts[1]["prev_turn_pcm_ms"] == 300.0


async def test_driven_down_passes_audio_through(fake):
    proc = make_proc(fake, reconnect_backoff_s=(0.1,))
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(100),
            audio_at_driveningress(fake),  # on the wire, waiting for its animation
            Do(lambda: fake.close_with(4001)),
            message_pushed(pushed, "state"),
            SleepFrame(0.4),
            tone_frame(100),
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
    )
    k = kinds(down)
    assert ("rtvi", "state") in k and k.count("TTSAudioRawFrame") == 2


# --- Beyond the brief: the contract rulings and the Pipecat duties its seven tests leave open.


def avatar_messages(frames):
    """``(member, payload)`` for every avatar message, in order."""
    return [
        next((key, value) for key, value in f.data.items() if key != "tag")
        for f in frames
        if isinstance(f, RTVIServerMessageFrame)
    ]


def assert_turn_lifecycle(frames):
    """Every announced turn ends exactly once, and nothing of a turn follows its end."""
    seen, ended = set(), set()
    for member, payload in avatar_messages(frames):
        if member == "state":
            assert payload["status"] in ("STATUS_OK", "STATUS_DEGRADED")
            continue
        turn_id = payload["turn_id"]
        if member == "turn_start":
            assert turn_id not in seen
            seen.add(turn_id)
            assert payload["prev_turn_id"] or payload["prev_turn_pcm_ms"] == 0
            continue
        assert turn_id in seen and turn_id not in ended, (member, turn_id)
        if member in ("turn_end", "turn_abort"):
            ended.add(turn_id)
        if member == "turn_onsets":
            assert payload["onsets_ms"]
    assert seen == ended


async def test_every_turn_message_follows_the_lifecycle(fake):
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            TTSStartedFrame(),
            tone_frame(300),
            SleepFrame(0.3),
            tone_frame(500),
            TTSStoppedFrame(),
            TTSStartedFrame(),
            tone_frame(200),
            TTSStoppedFrame(),
            SleepFrame(1.5),
        ],
    )
    assert_turn_lifecycle(down)
    segments = [p for m, p in avatar_messages(down) if m == "frames"]
    ends = messages(down, "turn_end")
    assert len(ends) == 2
    for end in ends:
        seqs = sorted({s["segment_seq"] for s in segments if s["turn_id"] == end["turn_id"]})
        assert seqs == list(range(end["segment_count"]))


async def test_turn_end_counts_segments_and_pcm(fake):
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[TTSStartedFrame(), tone_frame(500), TTSStoppedFrame(), SleepFrame(1.0)],
    )
    (end,) = messages(down, "turn_end")
    # 500 ms: segments at 200 and 400 ms, then the end flush with the 100 ms left over.
    assert end == {"turn_id": req_of(down), "pcm_ms": 500.0, "segment_count": 3}
    seqs = [(p["segment_seq"], p["end"]) for m, p in avatar_messages(down) if m == "frames"]
    assert sorted(set(seqs)) == [(0, False), (1, False), (2, True)]


async def test_turn_start_reports_the_onsets_found_so_far(fake):
    silence = TTSAudioRawFrame(audio=bytes(2 * 4800), sample_rate=24000, num_channels=1)
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            TTSStartedFrame(),
            silence,  # 200 ms: the first segment comes back for silence alone
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
    )
    (start,) = messages(down, "turn_start")
    later = [o for p in messages(down, "turn_onsets") for o in p["onsets_ms"]]
    # The tone starts at sample 4800 (200 ms); a sine's first sample is 0, so the onset is the next.
    assert start["onsets_ms"] + later == [4801 * 1000 / 24000]


async def test_a_mid_turn_onset_is_reported_once(fake):
    silence = TTSAudioRawFrame(audio=bytes(2 * 4800), sample_rate=24000, num_channels=1)
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            TTSStartedFrame(),
            tone_frame(300),
            message_pushed(pushed, "turn_start"),  # the turn is announced before the rest comes
            silence,  # 200 ms, more than min_silence_ms (150)
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
    )
    (start,) = messages(down, "turn_start")
    assert start["onsets_ms"] == [1 * 1000 / 24000]
    # The second tone starts at sample 12000 (500 ms), audible from the next one. It is reported
    # once, with the next segment; the segments after that bring no new onset and no message.
    assert messages(down, "turn_onsets") == [
        {"turn_id": start["turn_id"], "onsets_ms": [12001 * 1000 / 24000]}
    ]
    assert len({p["segment_seq"] for m, p in avatar_messages(down) if m == "frames"}) > 2


async def test_frames_in_a_turn_keep_their_order(fake):
    text = TextFrame("hello")
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            tone_frame(100, context_id="c1"),
            text,
            tone_frame(100, context_id="c1"),
            TTSStoppedFrame(context_id="c1"),
            SleepFrame(0.8),
        ],
    )
    plain = [f for f in down if not isinstance(f, RTVIServerMessageFrame)]
    assert [type(f).__name__ for f in plain] == [
        "TTSStartedFrame",
        "TTSAudioRawFrame",
        "TextFrame",
        "TTSAudioRawFrame",
        "TTSStoppedFrame",
    ]
    assert plain[2] is text
    assert {f.context_id for f in plain if isinstance(f, TTSAudioRawFrame)} == {"c1"}


async def test_other_audio_in_a_turn_is_sent_and_counted_and_keeps_its_class(fake):
    chime = OutputAudioRawFrame(
        audio=pcm16(sine(100, 16000, 880)), sample_rate=16000, num_channels=1
    )
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            TTSStartedFrame(),
            tone_frame(200),
            chime,  # a sound effect, at a rate of its own
            tone_frame(200),
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
    )
    audio = [f for f in down if isinstance(f, OutputAudioRawFrame)]
    names = [type(f).__name__ for f in audio]
    assert names[0] == names[-1] == "TTSAudioRawFrame"
    assert set(names[1:-1]) == {"OutputAudioRawFrame"}  # not counted as the bot speaking
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in audio)
    chime_ms = sum(len(f.audio) for f in audio[1:-1]) * 1000 / (2 * 24000)
    assert abs(chime_ms - 100) <= 1  # its resampler's tail comes out before the next tone
    # Inference gets the same samples as the room, in the same order.
    assert fake.received_audio_bytes(req_of(down)) == b"".join(f.audio for f in audio)
    (end,) = messages(down, "turn_end")
    assert end["pcm_ms"] == 400 + chime_ms


async def test_audio_for_another_track_keeps_its_place_and_is_not_counted(fake):
    music = OutputAudioRawFrame(
        audio=pcm16(sine(100, 24000, 220)), sample_rate=24000, num_channels=1
    )
    music.transport_destination = "music"
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            TTSStartedFrame(),
            tone_frame(200),
            music,
            tone_frame(200),
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
    )
    audio = [f for f in down if isinstance(f, OutputAudioRawFrame)]
    assert audio[1] is music and len(audio) == 3
    assert fake.received_audio_ms(req_of(down)) == 400
    assert messages(down, "turn_end")[0]["pcm_ms"] == 400.0


async def test_stereo_tts_audio_is_downmixed(fake):
    left = sine(200, 24000)
    stereo = pcm16([sample for value in left for sample in (value, 0)])  # right channel silent
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            TTSStartedFrame(),
            TTSAudioRawFrame(audio=stereo, sample_rate=24000, num_channels=2),
            TTSStoppedFrame(),
            SleepFrame(0.8),
        ],
    )
    (released,) = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert released.num_channels == 1 and released.sample_rate == 24000
    assert released.audio == pcm16([value // 2 for value in left])  # the two channels averaged
    assert fake.received_audio_bytes(req_of(down)) == released.audio
    assert messages(down, "turn_end")[0]["pcm_ms"] == 200.0


async def test_turns_never_interleave(fake):
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            tone_frame(300, context_id="c1"),
            TTSStoppedFrame(context_id="c1"),
            TTSStartedFrame(context_id="c2"),
            tone_frame(100, context_id="c2"),
            tone_frame(100, context_id="c2"),
            TTSStoppedFrame(context_id="c2"),
            SleepFrame(1.5),
        ],
    )
    order = [
        (type(f).__name__, f.context_id)
        for f in down
        if isinstance(f, (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame))
    ]
    assert order == [
        ("TTSStartedFrame", "c1"),
        ("TTSAudioRawFrame", "c1"),
        ("TTSStoppedFrame", "c1"),
        ("TTSStartedFrame", "c2"),
        ("TTSAudioRawFrame", "c2"),
        ("TTSAudioRawFrame", "c2"),
        ("TTSStoppedFrame", "c2"),
    ]
    assert_turn_lifecycle(down)


# A turn is chained only when it follows the previous one back to back; any other waits until the
# room has been quiet for min_silence_ms (150) and one chunk (40), so the browser finds its onset.
GAP_S = (150 + 40) / 1000

# The tests that need a turn to be ready after the previous one has played out, but before the room
# has been quiet for min_silence_ms, use a min_silence_ms of 400: the window is 400 ms wide, not
# 150, which a loaded machine would miss. The rule they pin does not depend on the numbers.
WIDE_MIN_SILENCE_MS = 400
WIDE_GAP_S = (WIDE_MIN_SILENCE_MS + 40) / 1000


def audio_pushes(pushed):
    """``(time, frame)`` of every audio frame pushed, stall silence included."""
    return [(t, f) for t, f in pushed if isinstance(f, OutputAudioRawFrame)]


async def test_a_turn_after_a_short_gap_is_not_chained_and_waits_for_a_findable_onset(fake):
    proc = make_proc(fake, min_silence_ms=WIDE_MIN_SILENCE_MS)
    pushed = record_pushes(proc)
    segments = record_segments(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            tts_audio_released(pushed),  # turn 1 plays for 300 ms from here
            SleepFrame(0.4),  # turn 2's TTS starts ~100 ms after turn 1 has played out
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
    )
    (t1, _), (t2, second_audio) = audio_pushes(pushed)
    end1 = t1 + 0.3  # turn 1's playout end
    _, second = messages(down, "turn_start")
    ready = min(t for t, req_id in segments if req_id == second["turn_id"])
    assert 0 < ready - end1 < WIDE_MIN_SILENCE_MS / 1000  # inside the gap the browser can't see
    assert second["prev_turn_id"] == "" and second["prev_turn_pcm_ms"] == 0
    assert t2 - end1 >= WIDE_GAP_S - 0.002
    assert type(second_audio) is TTSAudioRawFrame  # no silence was pushed for the gap
    # Its inference stream starts at its own first sample, as for any turn.
    assert fake.received_audio_ms(second["turn_id"]) == 300
    assert second["onsets_ms"] == [1000 / 24000]
    assert_turn_lifecycle(down)


async def test_a_turn_ready_only_after_the_previous_one_played_out_is_not_chained(
    fake, monkeypatch
):
    animate = fake._animate
    asked = set()

    async def slow(ws, connection_id, req_id, *, end, with_animation):
        if req_id not in asked:
            asked.add(req_id)
            await asyncio.sleep(0.3)  # the first answer takes the time inference takes
        await animate(ws, connection_id, req_id, end=end, with_animation=with_animation)

    monkeypatch.setattr(fake, "_animate", slow)
    proc = make_proc(fake, min_silence_ms=WIDE_MIN_SILENCE_MS)
    pushed = record_pushes(proc)
    segments = record_segments(proc)
    tts_starts = []
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(600),
            TTSStoppedFrame(),
            tts_audio_released(pushed),  # turn 1 plays for 600 ms from here
            SleepFrame(0.35),  # turn 2's TTS starts before turn 1 has played out...
            Do(lambda: tts_starts.append(asyncio.get_running_loop().time())),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(1.5),
        ],
    )
    (t1, _), (t2, second_audio) = audio_pushes(pushed)
    end1 = t1 + 0.6
    _, second = messages(down, "turn_start")
    ready = min(t for t, req_id in segments if req_id == second["turn_id"])
    # The case under test: the TTS came in time, but its animation only comes back after.
    assert tts_starts[0] < end1
    assert 0 < ready - end1 < WIDE_MIN_SILENCE_MS / 1000
    assert second["prev_turn_id"] == "" and second["prev_turn_pcm_ms"] == 0
    assert t2 - end1 >= WIDE_GAP_S - 0.002 and type(second_audio) is TTSAudioRawFrame
    assert_turn_lifecycle(down)


async def test_audio_between_two_turns_breaks_the_chain(fake):
    chime = OutputAudioRawFrame(
        audio=pcm16(sine(100, 24000, 880)), sample_rate=24000, num_channels=1
    )
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await run_test(
        proc,
        frames_to_send=[
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            chime,  # after turn 1 and before turn 2, part of neither
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(1.5),
        ],
    )
    first, second = messages(down, "turn_start")
    assert second["prev_turn_id"] == ""  # the previous turn's length no longer places it
    (t1, _), (_, played), (t2, _) = audio_pushes(pushed)
    assert played is chime
    assert t2 - (t1 + 0.3 + 0.1) >= GAP_S - 0.002  # quiet long enough after the chime
    assert fake.received_audio_ms(first["turn_id"]) == 300  # the chime is in neither turn
    assert_turn_lifecycle(down)


def turn_start_time(pushed, turn_id):
    """When the ``turn_start`` of ``turn_id`` was pushed."""
    return next(
        t
        for t, f in pushed
        if isinstance(f, RTVIServerMessageFrame)
        and f.data.get("turn_start", {}).get("turn_id") == turn_id
    )


async def test_an_unchained_turn_is_announced_when_its_release_starts(fake):
    proc = make_proc(fake, min_silence_ms=WIDE_MIN_SILENCE_MS)
    pushed = record_pushes(proc)
    segments = record_segments(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            tts_audio_released(pushed),  # turn 1 plays for 300 ms from here
            SleepFrame(0.4),  # turn 2 is ready ~100 ms after turn 1 has played out
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(1.2),
        ],
    )
    _, second = messages(down, "turn_start")
    turn_id = second["turn_id"]
    (t1, _), (t2, _) = audio_pushes(pushed)
    end1 = t1 + 0.3
    ready = min(t for t, req_id in segments if req_id == turn_id)
    assert 0 < ready - end1 < WIDE_MIN_SILENCE_MS / 1000 and second["prev_turn_id"] == ""
    announced = turn_start_time(pushed, turn_id)
    # Not when its animation is in hand: when its release begins, after 440 ms of quiet. Its audio
    # follows in the same step (the 200 ms leave room for a loaded machine between two pushes).
    assert announced - end1 >= WIDE_GAP_S - 0.002
    assert 0 <= t2 - announced < 0.2
    # At its release: turn_start, then every segment already in hand, in order, then its audio.
    at_release = [f for t, f in pushed if t >= announced]
    first_audio = next(i for i, f in enumerate(at_release) if isinstance(f, TTSAudioRawFrame))
    head = kinds(at_release[: first_audio + 1])
    assert head[0] == ("rtvi", "turn_start") and head[-1] == "TTSAudioRawFrame"
    assert set(head[1:-1]) == {("rtvi", "frames"), "TTSStartedFrame"}
    assert head.index("TTSStartedFrame") == len(head) - 2  # all the frames go first
    seqs = [
        f.data["frames"]["segment_seq"]
        for f in at_release[1:first_audio]
        if isinstance(f, RTVIServerMessageFrame) and "frames" in f.data
    ]
    in_hand = len([t for t, req_id in segments if req_id == turn_id and t < announced])
    assert in_hand >= 1 and sorted(set(seqs)) == list(range(in_hand)) and seqs == sorted(seqs)


async def test_audio_released_before_an_unchained_turn_comes_before_its_announcement(fake):
    chime = OutputAudioRawFrame(
        audio=pcm16(sine(100, 24000, 880)), sample_rate=24000, num_channels=1
    )
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    segments = record_segments(proc)
    down, _ = await run_test(
        proc,
        frames_to_send=[
            TTSStartedFrame(),
            tone_frame(600),  # the chime goes out ~400 ms in, long after turn 2's animation
            TTSStoppedFrame(),
            chime,
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(1.8),
        ],
    )
    _, second = messages(down, "turn_start")
    assert second["prev_turn_id"] == ""
    chime_pushed = next(t for t, f in pushed if f is chime)
    # Turn 2's animation was in hand before the chime played...
    assert min(t for t, req_id in segments if req_id == second["turn_id"]) < chime_pushed
    # ...and still the browser hears of turn 2 only after the chime, which it could otherwise
    # take for turn 2's onset.
    assert turn_start_time(pushed, second["turn_id"]) > chime_pushed


async def test_a_chained_turn_is_still_announced_ahead_of_the_previous_turns_end(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await run_test(
        proc,
        frames_to_send=[
            TTSStartedFrame(),
            tone_frame(600),
            TTSStoppedFrame(),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(1.8),
        ],
    )
    first, second = messages(down, "turn_start")
    assert second["prev_turn_id"] == first["turn_id"]
    (t1, _), _ = audio_pushes(pushed)
    assert turn_start_time(pushed, second["turn_id"]) < t1 + 0.6  # before turn 1 has played out


async def test_end_frame_waits_for_the_open_turn(fake):
    # No sleep after the turn: run_test's EndFrame arrives while the audio is still held.
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[TTSStartedFrame(), tone_frame(300), TTSStoppedFrame()],
    )
    k = kinds(down)
    assert k.count("TTSAudioRawFrame") == 1 and k[-1] == ("rtvi", "turn_end")


async def test_interruption_during_release_aborts_the_turn(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            TTSStartedFrame(),
            *[tone_frame(100) for _ in range(10)],
            tts_audio_released(pushed),  # the release is under way: 1 s of audio, 200 ms ahead
            SleepFrame(0.3),
            InterruptionFrame(),
            SleepFrame(0.5),
        ],
    )
    k = kinds(down)
    assert ("rtvi", "turn_end") not in k
    assert k.index(("rtvi", "turn_abort")) < k.index("InterruptionFrame")
    assert "TTSAudioRawFrame" not in k[k.index("InterruptionFrame") :]
    assert 0 < k.count("TTSAudioRawFrame") < 10
    assert messages(down, "turn_abort") == [{"turn_id": req_of(down)}]
    assert fake.received[-1].WhichOneof("data") == "client_interrupt"
    assert_turn_lifecycle(down)


async def test_interruption_after_tts_stopped_aborts_the_turn(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            TTSStartedFrame(),
            *[tone_frame(100) for _ in range(10)],
            TTSStoppedFrame(),
            tts_audio_released(pushed),
            SleepFrame(0.3),
            InterruptionFrame(),
            SleepFrame(0.5),
        ],
    )
    k = kinds(down)
    assert ("rtvi", "turn_abort") in k and ("rtvi", "turn_end") not in k
    assert "TTSStoppedFrame" not in k  # it was still held, behind audio, and is dropped with it
    assert 0 < k.count("TTSAudioRawFrame") < 10
    # The request was closed already; the interrupt still goes out to stop its inference.
    assert fake.received[-1].WhichOneof("data") == "client_interrupt"
    assert_turn_lifecycle(down)


async def test_interruption_while_the_last_audio_plays_aborts_the_turn(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            TTSStartedFrame(),
            tone_frame(800),  # released at once: the hold is drained, the audio is still playing
            TTSStoppedFrame(),
            tts_audio_released(pushed),
            SleepFrame(0.3),
            InterruptionFrame(),
            SleepFrame(0.8),
        ],
    )
    k = kinds(down)
    assert ("rtvi", "turn_abort") in k and ("rtvi", "turn_end") not in k
    assert_turn_lifecycle(down)


class InterruptUpstream(FrameProcessor):
    """Turns a ``UserStartedSpeakingFrame`` into an ``InterruptionFrame`` sent back upstream.

    It stands for a processor after the avatar processor that interrupts the bot, the way
    Pipecat's ``broadcast_interruption`` sends one copy each way.
    """

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            await self.push_frame(InterruptionFrame(), FrameDirection.UPSTREAM)


async def test_an_interruption_from_downstream_aborts_the_turn(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, up = await play(
        Pipeline([proc, InterruptUpstream()]),
        [
            TTSStartedFrame(),
            *[tone_frame(100) for _ in range(10)],
            tts_audio_released(pushed),
            SleepFrame(0.3),
            UserStartedSpeakingFrame(),
            SleepFrame(0.5),
        ],
    )
    k = kinds(down)
    assert ("rtvi", "turn_abort") in k and ("rtvi", "turn_end") not in k
    assert 0 < k.count("TTSAudioRawFrame") < 10
    assert [type(f) for f in up] == [InterruptionFrame]  # passed on in its own direction
    assert fake.received[-1].WhichOneof("data") == "client_interrupt"


def uninterruptible(frame):
    frame.interruptible = False
    return frame


async def test_an_uninterruptible_context_plays_through_an_interruption(fake):
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            uninterruptible(TTSStartedFrame()),  # a backchannel, say
            *[uninterruptible(tone_frame(100)) for _ in range(5)],
            uninterruptible(TTSStoppedFrame()),
            SleepFrame(0.1),
            InterruptionFrame(),
            SleepFrame(0.8),
        ],
    )
    k = kinds(down)
    assert k.count("TTSAudioRawFrame") == 5 and "TTSStoppedFrame" in k
    assert ("rtvi", "turn_abort") not in k and k[-1] == ("rtvi", "turn_end")
    assert not [m for m in fake.received if m.WhichOneof("data") == "client_interrupt"]
    assert_turn_lifecycle(down)


async def test_a_failed_send_plays_the_turn_unanimated(fake, monkeypatch):
    async def failing_send(self, pcm, *, end=False):
        await asyncio.sleep(0.05)  # the write fails a little later, as a broken socket does
        self._lose(self._link, "connectionClosed")
        raise RuntimeError("driveningress not connected")

    monkeypatch.setattr(DrivenIngress, "send", failing_send)
    down, _ = await run_test(
        make_proc(fake, reconnect_backoff_s=(5,)),
        frames_to_send=[TTSStartedFrame(), tone_frame(200), TTSStoppedFrame(), SleepFrame(0.6)],
    )
    assert messages(down, "state") == [{"status": "STATUS_DEGRADED", "reason": "connectionClosed"}]
    (start,) = messages(down, "turn_start")
    assert start["degraded"] is True
    assert kinds(down).count("TTSAudioRawFrame") == 1
    assert messages(down, "turn_end")[0]["segment_count"] == 0
    assert_turn_lifecycle(down)


async def test_a_turn_after_an_interruption_is_a_new_request(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            TTSStartedFrame(),
            tone_frame(600),
            message_pushed(pushed, "turn_start"),  # turn 1 is on the air...
            InterruptionFrame(),
            interruption_pushed(pushed),  # ...and cut off before turn 2 begins
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
    )
    first, second = messages(down, "turn_start")
    assert first["turn_id"] != second["turn_id"]
    assert second["prev_turn_id"] == "" and second["prev_turn_pcm_ms"] == 0
    assert fake.received_audio_ms(second["turn_id"]) == 300
    assert_turn_lifecycle(down)


async def test_uninterruptible_frames_survive_an_interruption(fake):
    fake.first_segment_ms = 10_000
    keep = TextFrame("function result")
    keep.interruptible = False
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[
            TTSStartedFrame(),
            tone_frame(100),
            keep,
            SleepFrame(0.1),
            InterruptionFrame(),
            SleepFrame(0.3),
        ],
    )
    k = kinds(down)
    assert keep in down and "TTSAudioRawFrame" not in k
    assert down.index(keep) > k.index("InterruptionFrame")


async def test_hold_cap_starts_a_degraded_turn(fake):
    fake.first_segment_ms = 10_000
    proc = make_proc(fake, hold_cap_s=0.3)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(200),
            message_pushed(pushed, "turn_start"),  # the cap has run out, the TTS still goes on
            TTSStoppedFrame(),
            SleepFrame(0.8),
        ],
    )
    (start,) = messages(down, "turn_start")
    assert (
        start["degraded"] is True and start["turn_id"] == fake.received[0].client_audio_input.req_id
    )
    k = kinds(down)
    assert k.index(("rtvi", "turn_start")) < k.index("TTSAudioRawFrame")
    # The request is still closed, so the next one gets a new id, and the end flush that comes back
    # before the turn ends is delivered.
    assert fake.received[-1].client_audio_input.end
    (end,) = messages(down, "turn_end")
    assert end["segment_count"] == 1 and k[-1] == ("rtvi", "turn_end")
    assert_turn_lifecycle(down)


async def test_turn_ends_when_the_end_segment_never_comes(fake, monkeypatch):
    animate = fake._animate

    async def no_end(ws, connection_id, req_id, *, end, with_animation):
        if not end:
            await animate(ws, connection_id, req_id, end=end, with_animation=with_animation)

    monkeypatch.setattr(fake, "_animate", no_end)
    proc = make_proc(fake, hold_cap_s=0.3)
    down, _ = await play(
        proc,
        [link_up(proc), TTSStartedFrame(), tone_frame(200), TTSStoppedFrame(), SleepFrame(1.0)],
    )
    (end,) = messages(down, "turn_end")
    assert end["segment_count"] == 1 and end["pcm_ms"] == 200.0
    assert_turn_lifecycle(down)


async def test_a_segment_after_turn_end_is_dropped(fake, monkeypatch):
    animate = fake._animate

    async def late_end(ws, connection_id, req_id, *, end, with_animation):
        if end:
            await asyncio.sleep(0.8)
        await animate(ws, connection_id, req_id, end=end, with_animation=with_animation)

    monkeypatch.setattr(fake, "_animate", late_end)
    proc = make_proc(fake, hold_cap_s=0.2)
    down, _ = await play(
        proc,
        [link_up(proc), TTSStartedFrame(), tone_frame(300), TTSStoppedFrame(), SleepFrame(1.5)],
    )
    assert len(fake.animation_frames) == 2  # the late end did reach the processor
    (end,) = messages(down, "turn_end")
    assert end["segment_count"] == 1
    assert kinds(down)[-1] == ("rtvi", "turn_end")
    assert_turn_lifecycle(down)


async def test_turn_end_waits_for_an_end_segment_that_comes_after_the_audio(fake, monkeypatch):
    animate = fake._animate

    async def slow_end(ws, connection_id, req_id, *, end, with_animation):
        if end:
            await asyncio.sleep(0.4)  # the 300 ms of audio have played by then
        await animate(ws, connection_id, req_id, end=end, with_animation=with_animation)

    monkeypatch.setattr(fake, "_animate", slow_end)
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[TTSStartedFrame(), tone_frame(300), TTSStoppedFrame(), SleepFrame(1.2)],
    )
    (end,) = messages(down, "turn_end")
    assert end["segment_count"] == 2
    assert [p["end"] for m, p in avatar_messages(down) if m == "frames"][-1] is True
    assert kinds(down)[-1] == ("rtvi", "turn_end")


async def test_a_turn_that_loses_its_link_is_released_at_once(fake):
    proc = make_proc(fake, reconnect_backoff_s=(5,))
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(100),
            # The turn's request is on the wire, waiting for its animation, when the link goes;
            # cut earlier, the turn would meet the loss in its first send, which is another path.
            audio_at_driveningress(fake),
            Do(lambda: fake.close_with(4001)),
            message_pushed(pushed, "state"),
            SleepFrame(0.4),  # the TTS pauses; the hold cap is 1.5 s
            tone_frame(100),
            TTSStoppedFrame(),
            SleepFrame(0.6),
        ],
    )
    k = kinds(down)
    first, second = [i for i, kind in enumerate(k) if kind == "TTSAudioRawFrame"]
    # It plays from the moment the link goes, not once the hold cap (1.5 s) runs out: the TTS's
    # pause is already bridged with silence.
    assert "OutputAudioRawFrame" in k[first:second]
    assert messages(down, "turn_start")[0]["degraded"] is True


async def test_degraded_state_is_reported_once_and_cleared(fake):
    proc = make_proc(fake, reconnect_backoff_s=(0.2,))
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(100),
            audio_at_driveningress(fake),  # on the wire, waiting for its animation
            Do(lambda: fake.close_with(4001)),
            message_pushed(pushed, "state"),
            SleepFrame(0.25),
            tone_frame(100),
            tone_frame(100),
            TTSStoppedFrame(),
            message_pushed(pushed, "state", count=2),  # reconnected: the status is OK again
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            SleepFrame(1.0),
        ],
    )
    states = messages(down, "state")
    assert states == [
        {"status": "STATUS_DEGRADED", "reason": "creditsExhausted (4001)"},
        {"status": "STATUS_OK", "reason": ""},
    ]
    first, second = messages(down, "turn_start")
    assert first["degraded"] is True and second["degraded"] is False
    assert [m for m, _ in avatar_messages(down)].count("frames") >= 1
    assert fake.connections == 2
    assert_turn_lifecycle(down)


async def test_a_turn_while_down_is_announced_degraded_and_sends_nothing(fake):
    proc = make_proc(fake, reconnect_backoff_s=(5,))
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            Do(lambda: fake.close_with(4001)),
            message_pushed(pushed, "state"),
            TTSStartedFrame(),
            tone_frame(200),
            TTSStoppedFrame(),
            SleepFrame(0.5),
        ],
    )
    (start,) = messages(down, "turn_start")
    assert start["degraded"] is True and start["onsets_ms"] == [1000 / 24000]
    (end,) = messages(down, "turn_end")
    assert end == {"turn_id": start["turn_id"], "pcm_ms": 200.0, "segment_count": 0}
    assert kinds(down).count("TTSAudioRawFrame") == 1
    assert not [m for m in fake.received if m.WhichOneof("data") == "client_audio_input"]


async def test_a_turn_while_down_is_not_held(fake):
    proc = make_proc(fake, reconnect_backoff_s=(5,))
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            Do(lambda: fake.close_with(4001)),
            message_pushed(pushed, "state"),
            TTSStartedFrame(),
            tone_frame(100),
            SleepFrame(0.4),  # the hold cap is 1.5 s
            TTSStoppedFrame(),
            SleepFrame(0.5),
        ],
    )
    k = kinds(down)
    # Released as it comes: the audio already played and silence bridges the TTS's pause.
    assert "OutputAudioRawFrame" in k[k.index("TTSAudioRawFrame") : k.index("TTSStoppedFrame")]


async def test_system_frames_overtake_a_held_turn(fake):
    fake.first_segment_ms = 10_000
    down, _ = await run_test(
        make_proc(fake, hold_cap_s=0.3),
        frames_to_send=[
            TTSStartedFrame(),
            tone_frame(100),
            SleepFrame(0.1),  # the turn is held when the system frame arrives
            UserStartedSpeakingFrame(),
            SleepFrame(0.5),
        ],
    )
    k = kinds(down)
    # The turn itself comes out on the hold cap, after the system frame.
    assert k.index("UserStartedSpeakingFrame") < k.index("TTSStartedFrame")


async def test_upstream_frames_pass_at_once(fake):
    _, up = await run_test(
        make_proc(fake),
        frames_to_send=[TextFrame("upstream")],
        frames_to_send_direction=FrameDirection.UPSTREAM,
    )
    assert [type(f).__name__ for f in up] == ["TextFrame"]


async def test_a_cancel_mid_turn_stops_the_release_and_the_link(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    at_cancel = {}

    @proc.event_handler("on_after_push_frame")
    async def _at_cancel(processor, frame):
        if isinstance(frame, CancelFrame):
            at_cancel.update(release=processor._driver, link=processor._link_task)

    await play(
        proc,
        [
            TTSStartedFrame(),
            *[tone_frame(100) for _ in range(20)],  # 2 s of audio, still being released
            tts_audio_released(pushed),
            SleepFrame(0.3),
            CancelFrame(),
            SleepFrame(0.5),
        ],
        send_end_frame=False,
    )
    assert at_cancel == {"release": None, "link": None}
    assert isinstance(pushed[-1][1], CancelFrame)  # nothing is pushed after it
    assert 0 < sum(isinstance(f, TTSAudioRawFrame) for _, f in pushed) < 20


async def test_nothing_follows_the_end_frame(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    at_end = {}

    @proc.event_handler("on_after_push_frame")
    async def _lose_the_link_as_the_pipeline_ends(processor, frame):
        if isinstance(frame, EndFrame):
            at_end["link"] = processor._link_task
            processor._on_down("closed")  # a loss reported right then

    # No sleep: the EndFrame waits behind the turn and goes out from the release task.
    await run_test(proc, frames_to_send=[TTSStartedFrame(), tone_frame(300), TTSStoppedFrame()])
    assert at_end == {"link": None}  # no reconnect either
    assert isinstance(pushed[-1][1], EndFrame)  # no state message after it


async def test_a_failing_release_lets_the_audio_through(fake, monkeypatch, pipecat_log):
    async def fault(self, turn):
        raise RuntimeError("a bug")

    monkeypatch.setattr(SpatialRealAvatarProcessor, "_forward_segments", fault)
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            # The release fails, sets proc._broken and then pushes turn 1's frames. Turn 2 must
            # wait for the last of them: once the flag is set, its frames pass straight through,
            # ahead of any of turn 1's that are still to come.
            Until(
                "turn 1's frames to be pushed",
                lambda: any(isinstance(f, TTSStoppedFrame) for _, f in pushed),
            ),
            TTSStartedFrame(),
            tone_frame(200),
            TTSStoppedFrame(),
            SleepFrame(0.3),
        ],
    )
    assert kinds(down) == [
        ("rtvi", "turn_start"),
        ("rtvi", "state"),
        ("rtvi", "turn_abort"),  # the announced turn still ends, once
        "TTSStartedFrame",
        "TTSAudioRawFrame",
        "TTSStoppedFrame",
        "TTSStartedFrame",  # and the next turn passes straight through
        "TTSAudioRawFrame",
        "TTSStoppedFrame",
    ]
    assert messages(down, "state") == [{"status": "STATUS_DEGRADED", "reason": "processor error"}]
    audio_requests = {
        m.client_audio_input.req_id
        for m in fake.received
        if m.WhichOneof("data") == "client_audio_input"
    }
    assert audio_requests == {req_of(down)}  # the second turn never reached driveningress
    failures = [r for r in pipecat_log if "release failed" in r["message"]]
    assert len(failures) == 1 and failures[0]["exception"] is not None
    pipecat_log[:] = [r for r in pipecat_log if r not in failures]  # expected, not a test error


@pytest.mark.parametrize(
    "overrides",
    [
        {"threshold": -1},
        {"min_silence_ms": 0},
        {"lead_ms": -1},
        {"hold_cap_s": 0},
        {"silence_chunk_ms": 0},
        {"reconnect_backoff_s": ()},
        {"reconnect_backoff_s": (1, -1)},
        {"api_key": ""},
    ],
)
def test_bad_settings_fail_at_construction(overrides):
    arguments = {"api_key": "k", "app_id": "a", "avatar_id": "av", **overrides}
    with pytest.raises(ValueError):
        SpatialRealAvatarProcessor(**arguments)


def test_exported():
    import pipecat_spatialreal

    assert "SpatialRealAvatarProcessor" in pipecat_spatialreal.__all__
