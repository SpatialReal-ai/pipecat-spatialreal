import asyncio
import base64
import contextlib
import itertools
import math
import statistics
import struct

import pytest
from conftest import (
    Do,
    Until,
    audio_at_driveningress,
    half_open,
    interruption_pushed,
    link_up,
    message_pushed,
    play,
    tts_audio_released,
)
from fake_driveningress import FakeDriveningress
from pipecat.audio.utils import create_stream_resampler
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
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_spatialreal import SpatialRealAvatarProcessor
from pipecat_spatialreal.driven import DrivenIngress
from pipecat_spatialreal.fingerprint import Envelope
from pipecat_spatialreal.processor import _Turn


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


# The processor releases audio in whole output chunks (output_chunk_ms, 40 by default) and pads
# every turn to a whole number of them, so a turn of 300 ms is released as 8 frames of 40 ms and
# counts 320 ms, in the room and at inference alike.
CHUNK_MS = 40
CHUNK_BYTES = 2 * 24000 * CHUNK_MS // 1000


def padded(ms):
    """The length, in ms, of a turn of ``ms`` of audio once padded to whole chunks."""
    return math.ceil(ms / CHUNK_MS) * CHUNK_MS


def chunks(ms):
    """How many frames a turn of ``ms`` of audio is released as."""
    return padded(ms) // CHUNK_MS


# A loaded machine can only make a wait longer. A bound on when the processor did something allows
# this much lateness on top of the time the rule names.
LATE_S = 0.5

# A push is stamped as it begins (see ``record_pushes``), right after the processor read its
# clock, so a turn's playout end rebuilt from the stamp of its first chunk is the processor's own
# to within the stamp's cost. The release timer may still fire a millisecond early (the
# processor's own tolerance), so a gap measured from that end can come out shorter than the rule
# by that much. This much is allowed.
GAP_SLACK_S = 0.01


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


def fill_frames(frames):
    """The silence the processor keeps the room fed with after a turn: plain zero audio pushed
    outside every turn (after a ``TTSStoppedFrame``, before the next ``TTSStartedFrame``)."""
    fill, inside = [], False
    for f in frames:
        if isinstance(f, TTSStartedFrame):
            inside = True
        elif isinstance(f, TTSStoppedFrame):
            inside = False
        elif not inside and type(f) is OutputAudioRawFrame and not any(f.audio):
            fill.append(f)
    return fill


def without_fill(frames):
    """``frames`` without the fill (see ``fill_frames``): what the turns themselves pushed."""
    fill = {id(f) for f in fill_frames(frames)}
    return [f for f in frames if id(f) not in fill]


def record_pushes(proc):
    """Every frame ``proc`` pushes, with the loop time it pushed it at, in order.

    The time is taken as the push begins: nothing in a push gives the loop up, so the order is the
    order the frames went out in, and the stamp of an audio chunk is the clock the processor
    placed it by, not that clock plus however long the process was held up inside the push.

    Unlike the sink, this also sees what is pushed after a ``CancelFrame``, which the processors
    downstream drop.
    """
    pushed = []

    @proc.event_handler("on_before_push_frame")
    async def _record(processor, frame):
        pushed.append((asyncio.get_running_loop().time(), frame))

    return pushed


@contextlib.asynccontextmanager
async def heartbeat(period_s=0.01):
    """Loop times at which the loop was running, taken every ``period_s`` while the block runs.

    Time the process was held up (not scheduled, frozen) leaves no ticks, so the ticks between two
    moments tell how long the loop was actually there to act in between: see ``live_s``.
    """
    ticks = []
    loop = asyncio.get_running_loop()

    async def beat():
        while True:
            await asyncio.sleep(period_s)
            ticks.append(loop.time())

    task = asyncio.create_task(beat())
    try:
        yield ticks
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def live_s(ticks, start, end, period_s=0.01):
    """How long the loop was running between ``start`` and ``end``, from ``heartbeat``'s ticks.

    At most the wall-clock time between the two, less the time nothing in the loop could run: a
    tick comes no earlier than its period after the previous one, so this never over-reports.
    """
    return sum(1 for t in ticks if start < t < end) * period_s


def first_audio_pushes(pushed):
    """``(time, frame)`` of the first audio pushed after each ``TTSStartedFrame``, silence included.

    The time is when the room starts playing that turn; the turn plays from there for its padded
    length, stall silence included.
    """
    found, awaiting = [], False
    for t, f in pushed:
        if isinstance(f, TTSStartedFrame):
            awaiting = True
        elif awaiting and isinstance(f, OutputAudioRawFrame):
            found.append((t, f))
            awaiting = False
    return found


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
def _every_test_checks_the_log(pipecat_log):
    """Every test here fails on a pipeline error or a task left running (see ``pipecat_log``)."""


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
    k = kinds(without_fill(down))  # the fill after the turn: see the R47 tests
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
    assert all(f.sample_rate == 24000 and len(f.audio) == CHUNK_BYTES for f in rel)
    total_ms = sum(len(f.audio) for f in rel) / 2 / 24000 * 1000
    assert 400 <= total_ms <= padded(401)  # the resampler's 400 ms, give or take, then the padding
    assert fake.received_audio_bytes(req_of(down)) == b"".join(f.audio for f in rel)
    assert messages(down, "turn_start")[0]["sample_rate"] == 24000


@pytest.mark.parametrize("rate", [16000, 48000], ids=["16 kHz", "48 kHz"])
async def test_the_pipelines_output_rate_is_what_inference_and_the_room_get(fake, rate):
    """The TTS speaks at 24 kHz and the pipeline's ``audio_out_sample_rate`` is another: that one
    is the rate of everything the processor makes, the session with inference and the samples it
    gets, the chunks the room gets, and what ``turn_start`` reports."""
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [TTSStartedFrame(), tone_frame(400), TTSStoppedFrame(), message_pushed(pushed, "turn_end")],
        params=PipelineParams(audio_out_sample_rate=rate),
    )
    released = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    chunk_bytes = 2 * rate * CHUNK_MS // 1000
    assert released and all(
        f.sample_rate == rate and f.num_channels == 1 and len(f.audio) == chunk_bytes
        for f in released
    )
    pcm = b"".join(f.audio for f in released)
    total_ms = len(pcm) * 1000 / (2 * rate)
    assert 400 <= total_ms <= padded(401)  # the resampler's 400 ms, give or take, then the padding
    assert fake.configure.sample_rate == rate  # the session was opened at that rate...
    assert fake.received_audio_bytes(req_of(down)) == pcm  # ...and got the room's samples
    (start,) = messages(down, "turn_start")
    assert start["sample_rate"] == rate
    assert messages(down, "turn_end")[0]["pcm_ms"] == total_ms


async def test_stall_mirrors_silence(fake):
    proc = make_proc(fake, lead_ms=0)
    pushed = record_pushes(proc)
    async with heartbeat() as ticks:
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
    _, room_end1, second = _around_the_stall(pushed)
    stall = second - room_end1  # how long the room waited for the TTS
    (audio,) = turn_audio(down)  # the stall's silence is the turn's; the fill after it is not
    sil = [f for f in audio if type(f) is OutputAudioRawFrame]
    assert stall > 0.4 and all(set(f.audio) == {0} for f in sil)
    # Silence went out for the stall, one 40 ms chunk at a time and just in time (the lead is 0):
    # no more than the stall, and up to the audio that ended it.
    assert len(sil) * 0.04 <= stall + 0.05
    assert_bridged([t for t, f in pushed if any(f is g for g in sil)], second, ticks)
    # The same silence went to inference, and it is part of the turn's length.
    length = padded(500) + 40 * len(sil)
    assert fake.received_audio_ms(req_of(down)) == length
    assert messages(down, "turn_end")[0]["pcm_ms"] == length


def _around_the_stall(pushed):
    """When the first tone's chunks started playing, when they ran out, and when the TTS was back.

    The first tone (300 ms) is released as 7 whole chunks; its last 20 ms wait for the next tone,
    so the room runs dry 280 ms after the first chunk, and the stall silence starts there.
    """
    audio = [(t, f) for t, f in pushed if isinstance(f, OutputAudioRawFrame)]
    first_silence = next(i for i, (_, f) in enumerate(audio) if type(f) is OutputAudioRawFrame)
    run = audio[:first_silence]
    assert len(run) == 300 // CHUNK_MS and all(isinstance(f, TTSAudioRawFrame) for _, f in run)
    first = run[0][0]
    second = next(t for t, f in audio[first_silence:] if isinstance(f, TTSAudioRawFrame))
    return first, first + len(run) * CHUNK_MS / 1000, second


# A stall is bridged to its end: each chunk of silence is pushed while the room has one chunk left
# to play, so the TTS audio that ends the stall is pushed within a chunk of the last silence. A
# loaded machine can push either later; this much later is allowed on the clock. The stalls in
# these tests are half a second and longer, so a bridge that stops short of their end by more than
# this is caught on the clock alone.
BRIDGE_LATE_S = 0.25

# The same gap, in time the loop was running (``live_s``): a process that is not scheduled pushes
# nothing and is not blamed, but a loop that was there to push silence and did not is. Nominally
# the gap is under a chunk (40 ms); this much is allowed, so a bridge that stops short of a stall's
# end by more than it is caught however short the stall, whatever the machine.
BRIDGE_IDLE_S = 0.1


def assert_bridged(silence, resumed, ticks):
    """The silence pushed for a stall (its push times) went on until the audio that ended it.

    ``ticks`` are the ``heartbeat`` ticks of the run: the gap between the last silence and the
    audio that resumed is bounded both on the clock and in the time the loop was running.
    """
    assert silence and resumed - silence[-1] <= CHUNK_MS / 1000 + BRIDGE_LATE_S
    assert live_s(ticks, silence[-1], resumed) <= BRIDGE_IDLE_S


def queued_ahead(pushes, ends=None):
    """How much the room still had to play when each of ``pushes`` ((time, audio frame)) came.

    Rebuilt from the pushes: the room plays what it is given back to back, from the first push,
    or from ``ends`` when it was playing something until then. A push that comes late finds less
    queued, and one that finds the room dry gives how long it has been dry, negative.
    """
    left = []
    for t, f in pushes:
        if ends is None:
            ends = t
        left.append(ends - t)
        ends = max(ends, t) + len(f.audio) / (2 * f.sample_rate * f.num_channels)
    return left


async def test_a_stall_is_not_stretched_by_the_lead(fake):
    proc = make_proc(fake)  # lead_ms 200, silence_chunk_ms 40
    pushed = record_pushes(proc)
    async with heartbeat() as ticks:
        down, _ = await play(
            proc,
            [
                link_up(proc),
                TTSStartedFrame(),
                tone_frame(300),
                tts_audio_released(pushed),  # the room plays it for 300 ms from here
                SleepFrame(1.0),  # the TTS falls ~700 ms behind the room
                tone_frame(500),
                TTSStoppedFrame(),
                SleepFrame(1.2),
            ],
        )
    first, room_end1, second = _around_the_stall(pushed)
    silence = [(t, f) for t, f in pushed if type(f) is OutputAudioRawFrame and first <= t <= second]
    stall = second - room_end1  # how long the room waited for the TTS
    inserted = len(silence) * 0.04
    assert stall > 0.4 and silence
    # The silence never runs further ahead of the room than one chunk plus the chunk kept queued so
    # the transport never runs dry; the 200 ms lead is not added on top. This bound holds on any
    # machine: a loop that wakes late can only make the silence shorter.
    assert inserted - stall <= 0.04 + 0.04 + 0.01
    # And it bridges the stall to its end. Not judged by how much silence went out in all: while
    # the process is not scheduled nothing can be pushed, the stall measured here grows by as much,
    # and the silence does not make that up (the room was dry meanwhile). Judged by when the last
    # of it went out: within a chunk of the audio that ended the stall, however late the loop
    # woke before that.
    assert_bridged([t for t, _ in silence], second, ticks)
    # The chunk kept queued: each chunk is pushed while one chunk (40 ms) of what the room was
    # given is still to play, not when the room has run dry. Rebuild how much that was at each push
    # (the end of what was queued, less the time of the push). A late wake-up lowers its own
    # sample, and can leave the room dry; the median of the samples is what carries the pin.
    assert statistics.median(queued_ahead(silence, ends=room_end1)) >= 0.02
    assert fake.received_audio_ms(req_of(down)) == padded(800) + 40 * len(silence)


async def test_a_turn_chained_after_a_stall_is_placed_after_the_silence(fake):
    """The silence that bridged a stall is part of the turn's length: ``turn_end`` counts it, and
    the turn that follows back to back is placed from that length, after the silence."""
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(300),
            tts_audio_released(pushed),  # the room plays it for 300 ms from here
            SleepFrame(0.8),  # the TTS falls ~500 ms behind the room
            tone_frame(500),
            TTSStoppedFrame(),
            TTSStartedFrame(),  # turn 2 follows back to back: ready while turn 1 still releases
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=2),
        ],
    )
    _, room_end1, second_tone = _around_the_stall(pushed)
    assert second_tone - room_end1 > 0.4
    silence = [f for f in turn_audio(down)[0] if type(f) is OutputAudioRawFrame]  # not the fill
    length = padded(800) + 40 * len(silence)  # 800: turn 1's own audio ends on a chunk
    first, second = messages(down, "turn_start")
    assert second["prev_turn_id"] == first["turn_id"]
    assert second["prev_turn_pcm_ms"] == length
    assert [e["pcm_ms"] for e in messages(down, "turn_end")] == [length, padded(300)]
    assert fake.received_audio_ms(first["turn_id"]) == length  # what inference saw, too
    assert_turn_lifecycle(down)


LEAD_S = 0.2  # lead_ms, the default


async def test_audio_is_released_lead_ms_ahead_of_the_room(fake):
    """Each chunk is pushed when the room has at most ``lead_ms`` left to play and, with the audio
    in hand, little less: the lead is what keeps the transport fed, and all that is still audible
    after a barge-in."""
    fake.first_segment_ms = 2000  # the whole turn is in hand when its release starts
    proc = make_proc(fake, hold_cap_s=5.0)  # lead_ms 200
    pushed = record_pushes(proc)
    await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            *[tone_frame(100) for _ in range(20)],
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
    )
    fill = fill_frames([f for _, f in pushed])  # after the turn, not part of it
    audio = [(t, f) for t, f in audio_pushes(pushed) if not any(f is g for g in fill)]
    assert len(audio) == chunks(2000) and all(isinstance(f, TTSAudioRawFrame) for _, f in audio)
    ahead = queued_ahead(audio)
    # Never more than the lead queued when a chunk is pushed, give or take the clock's tolerance
    # (a chunk may go a millisecond early)...
    assert max(ahead) <= LEAD_S + 0.05
    # ...and, once the first chunks have filled the lead, about the lead. A late wake-up can only
    # lower a sample; the median of the samples is what carries the pin.
    assert statistics.median(ahead) >= LEAD_S - 0.1


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


async def test_a_segment_that_comes_back_after_an_interruption_is_dropped(fake, monkeypatch):
    """The clause the test above cannot reach: inference answers the aborted turn after the
    interrupt. Nothing is pushed for it, and the turn under way by then does not take it."""
    animate = fake._animate
    aborted = []  # the first request's id; its answers come late

    async def late(ws, connection_id, req_id, *, end, with_animation):
        if not aborted:
            aborted.append(req_id)
        if req_id == aborted[0]:
            await asyncio.sleep(0.4)
        await animate(ws, connection_id, req_id, end=end, with_animation=with_animation)

    monkeypatch.setattr(fake, "_animate", late)
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    segments = record_segments(proc)
    down, _ = await play(
        proc,
        [
            TTSStartedFrame(),
            tone_frame(200),  # one segment's worth: its answer is on its way, late
            Until("the turn's audio to be at driveningress", lambda: len(fake.received) >= 5),
            InterruptionFrame(),
            interruption_pushed(pushed),
            TTSStartedFrame(),  # turn 2 is under way when the late answer comes
            tone_frame(1000),
            TTSStoppedFrame(),
            Until("the late segment", lambda: any(req_id == aborted[0] for _, req_id in segments)),
            message_pushed(pushed, "turn_end"),
        ],
    )
    (start,) = messages(down, "turn_start")  # turn 2's: the aborted turn was never announced
    assert start["turn_id"] != aborted[0]
    frames = [p for m, p in avatar_messages(down) if m == "frames"]
    assert {p["turn_id"] for p in frames} == {start["turn_id"]}
    # Turn 2's own animation and nothing more: five full segments of its 1000 ms, and the end.
    (end,) = messages(down, "turn_end")
    assert end["segment_count"] == 6 == len({p["segment_seq"] for p in frames})
    assert_turn_lifecycle(down)


async def test_short_turn_uses_end_flush(fake):
    fake.first_segment_ms = 1000
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[TTSStartedFrame(), tone_frame(120), TTSStoppedFrame(), SleepFrame(0.8)],
    )
    k = kinds(down)
    # Its first animation is the end flush; it is still announced before that, as any turn is.
    assert (
        k.index(("rtvi", "turn_start")) < k.index(("rtvi", "frames")) < k.index("TTSAudioRawFrame")
    )
    assert ("rtvi", "turn_end") in k
    assert_turn_lifecycle(down)


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
    assert starts[1]["prev_turn_pcm_ms"] == padded(300)  # 320: the first turn ends on a chunk


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
    assert ("rtvi", "state") in k and k.count("TTSAudioRawFrame") == chunks(200)


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
    # 500 ms, padded to 520: segments at 200 and 400 ms, then the end flush with what is left.
    assert end == {"turn_id": req_of(down), "pcm_ms": padded(500), "segment_count": 3}
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


async def test_the_threshold_setting_reaches_the_onset_tracker(fake):
    """The tone the default threshold (328) hears is silence to one set above its amplitude."""
    proc = make_proc(fake, threshold=10_000)  # the test tone peaks at 9800
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [TTSStartedFrame(), tone_frame(300), TTSStoppedFrame(), message_pushed(pushed, "turn_end")],
    )
    (start,) = messages(down, "turn_start")
    assert start["onsets_ms"] == [] and messages(down, "turn_onsets") == []


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
    plain = [f for f in without_fill(down) if not isinstance(f, RTVIServerMessageFrame)]
    # The first tone's 100 ms give two whole chunks; its last 20 ms wait for the second tone, so
    # the text comes right after what was released before it, 20 ms early at most.
    assert [type(f).__name__ for f in plain] == [
        "TTSStartedFrame",
        *["TTSAudioRawFrame"] * 2,
        "TextFrame",
        *["TTSAudioRawFrame"] * (chunks(200) - 2),
        "TTSStoppedFrame",
    ]
    assert plain[3] is text
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
    audio = [f for f in without_fill(down) if isinstance(f, OutputAudioRawFrame)]
    assert all(
        f.sample_rate == 24000 and f.num_channels == 1 and len(f.audio) == CHUNK_BYTES
        for f in audio
    )
    released = b"".join(f.audio for f in audio)
    # The chime's samples, resampled as the processor resampled them (its resampler's tail
    # comes out before the next tone), are in the room whole and in their place.
    resampler = create_stream_resampler(clear_after_secs=None)
    chime_pcm = await resampler.resample(chime.audio, 16000, 24000) + await resampler.flush()
    assert abs(len(chime_pcm) * 1000 / (2 * 24000) - 100) <= 1
    assert released.index(chime_pcm) == 2 * 24000 * 200 // 1000  # right after the first tone
    # A released chunk keeps the class of the audio its first sample came from: the chime's
    # chunks are not counted as the bot speaking, the tones' are.
    chime_chunks = math.ceil(len(chime_pcm) / CHUNK_BYTES)
    assert [type(f).__name__ for f in audio] == (
        ["TTSAudioRawFrame"] * chunks(200)
        + ["OutputAudioRawFrame"] * chime_chunks
        + ["TTSAudioRawFrame"] * (len(audio) - chunks(200) - chime_chunks)
    )
    # Inference gets the same samples as the room, in the same order, padding included.
    assert fake.received_audio_bytes(req_of(down)) == released
    (end,) = messages(down, "turn_end")
    assert end["pcm_ms"] == len(released) * 1000 / (2 * 24000)
    assert 400 + 99 <= end["pcm_ms"] <= padded(400 + 101) and end["pcm_ms"] % CHUNK_MS == 0


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
    audio = [f for f in without_fill(down) if isinstance(f, OutputAudioRawFrame)]
    assert audio[chunks(200)] is music and len(audio) == 2 * chunks(200) + 1
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
    released = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert all(f.num_channels == 1 and f.sample_rate == 24000 for f in released)
    mono = b"".join(f.audio for f in released)
    assert mono == pcm16([value // 2 for value in left])  # the two channels averaged
    assert fake.received_audio_bytes(req_of(down)) == mono
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
    assert [(key, len(list(run))) for key, run in itertools.groupby(order)] == [
        (("TTSStartedFrame", "c1"), 1),
        (("TTSAudioRawFrame", "c1"), chunks(300)),
        (("TTSStoppedFrame", "c1"), 1),
        (("TTSStartedFrame", "c2"), 1),
        (("TTSAudioRawFrame", "c2"), chunks(200)),
        (("TTSStoppedFrame", "c2"), 1),
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


def room_quiet(pushed, quiet, at, sound_end):
    """How long the room was quiet before a turn whose first chunk was pushed at ``at``.

    Rebuilt from the pushes of ``quiet``, the silence played between the last sound (which
    ended at ``sound_end``) and the turn: the room plays them back to back from ``sound_end``,
    and a push that finds it dry plays from then on. The turn plays once all of it has.
    """
    ends = sound_end
    for t, f in pushed:
        if any(f is q for q in quiet):
            ends = max(ends, t) + len(f.audio) / (2 * f.sample_rate)
    return max(ends, at) - sound_end


def ran_dry(pushed, quiet, at, sound_end):
    """Whether the room ran dry inside the quiet: a push of it, or the turn's first chunk, came
    after everything before it had played (``room_quiet``'s rebuild)."""
    ends = sound_end
    for t, f in pushed:
        if any(f is q for q in quiet):
            if t > ends + 0.001:
                return True
            ends += len(f.audio) / (2 * f.sample_rate)
    return at > ends + 0.001


def assert_quiet_before(pushed, quiet, at, sound_end, need_s):
    """The room was quiet for at least ``need_s`` before the turn pushed at ``at``.

    While the room stayed fed, the quiet is the silence queued: at least ``need_s`` of it, and
    the turn pushed with at most one chunk of it left to play, as a stall resumes, not up to the
    lead early. On a loaded machine the room may run dry inside the quiet; the quiet is then
    measured on the clock, and the turn pushed no earlier than ``need_s`` after the last sound.
    """
    if ran_dry(pushed, quiet, at, sound_end):
        assert at - sound_end >= need_s - GAP_SLACK_S
        return
    total = room_quiet(pushed, quiet, at, sound_end)
    assert total >= need_s - GAP_SLACK_S
    assert at - sound_end >= total - CHUNK_MS / 1000 - GAP_SLACK_S


def quiet_ms(quiet):
    """The length in ms of silence frames, as the processor computes ``gap_ms``."""
    return (sum(len(f.audio) for f in quiet) // 2) * 1000 / 24000


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
    (t1, _), (t2, second_audio) = first_audio_pushes(pushed)
    end1 = t1 + padded(300) / 1000  # turn 1's playout end
    _, second = messages(down, "turn_start")
    ready = min(t for t, req_id in segments if req_id == second["turn_id"])
    assert 0 < ready - end1 < WIDE_MIN_SILENCE_MS / 1000  # inside the gap the browser can't see
    assert second["prev_turn_id"] == "" and second["prev_turn_pcm_ms"] == 0
    # The quiet is the fill played after turn 1 (R47); the turn starts once enough of it is in.
    _, _, quiet = between_turns(down)
    assert_quiet_before(pushed, quiet, t2, end1, WIDE_GAP_S)
    assert second["gap_ms"] in (0, quiet_ms(quiet))  # 0 only where the room ran dry
    assert type(second_audio) is TTSAudioRawFrame  # the quiet comes before its TTSStartedFrame
    # Its inference stream starts at its own first sample, as for any turn.
    assert fake.received_audio_ms(second["turn_id"]) == padded(300)
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
    (t1, _), (t2, second_audio) = first_audio_pushes(pushed)
    end1 = t1 + 0.6  # 600 ms is a whole number of chunks: nothing is padded
    _, second = messages(down, "turn_start")
    ready = min(t for t, req_id in segments if req_id == second["turn_id"])
    # The case under test: the TTS came in time, but its animation only comes back after.
    assert tts_starts[0] < end1
    assert 0 < ready - end1 < WIDE_MIN_SILENCE_MS / 1000
    assert second["prev_turn_id"] == "" and second["prev_turn_pcm_ms"] == 0
    _, _, quiet = between_turns(down)
    assert_quiet_before(pushed, quiet, t2, end1, WIDE_GAP_S)
    assert type(second_audio) is TTSAudioRawFrame
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
    (t1, _), (t2, _) = first_audio_pushes(pushed)
    fill = fill_frames([f for _, f in pushed])  # the silence after a turn (R47)
    played = [
        f
        for _, f in audio_pushes(pushed)
        if not isinstance(f, TTSAudioRawFrame) and not any(f is g for g in fill)
    ]
    assert played == [chime]  # released as it is, after turn 1's padded audio
    # Between the chime's end and turn 2's start: the quiet, which is the fill.
    _, _, between = between_turns(down)
    quiet = [f for f in between if f is not chime]
    assert_quiet_before(pushed, quiet, t2, t1 + padded(300) / 1000 + 0.1, GAP_S)
    assert second["gap_ms"] == 0  # the chime is in the gap: the envelopes cannot be joined
    assert fake.received_audio_ms(first["turn_id"]) == padded(300)  # the chime is in neither turn
    assert_turn_lifecycle(down)


async def test_audio_released_between_two_drained_turns_breaks_the_chain(fake):
    """The chime's other path: it arrives once turn 1's hold has drained (its last chunk pushed,
    ``lead_ms`` before its playout end) and goes straight through, and turn 2 is ready before
    turn 1 has played out. Chained, turn 2 would be placed early by the chime's length."""
    chime = OutputAudioRawFrame(
        audio=pcm16(sine(100, 24000, 880)), sample_rate=24000, num_channels=1
    )
    proc = make_proc(fake, lead_ms=1000)  # turn 1 drains a second before it has played out
    pushed = record_pushes(proc)
    segments = record_segments(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(1000),
            TTSStoppedFrame(),
            Until(
                "turn 1 to drain", lambda: any(isinstance(f, TTSStoppedFrame) for _, f in pushed)
            ),
            chime,  # no turn holds it: straight through, while turn 1 still plays
            TTSStartedFrame(),  # turn 2, ready long before turn 1 has played out
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=2),
        ],
    )
    first, second = messages(down, "turn_start")
    assert second["prev_turn_id"] == "" and second["prev_turn_pcm_ms"] == 0
    (t1, _), (t2, second_audio) = first_audio_pushes(pushed)
    end1 = t1 + 1.0  # 1000 ms is a whole number of chunks: nothing is padded
    chime_at = next(t for t, f in pushed if f is chime)
    ready2 = min(t for t, req_id in segments if req_id == second["turn_id"])
    # The case under test: the chime went out, and turn 2's animation was in hand, while turn 1
    # still played. Ready only after that, turn 2 would be unchained by the quiet instead.
    assert chime_at < ready2 < end1
    fill = fill_frames([f for _, f in pushed])  # the silence after a turn (R47)
    played = [
        f
        for _, f in audio_pushes(pushed)
        if not isinstance(f, TTSAudioRawFrame) and not any(f is g for g in fill)
    ]
    assert played == [chime]
    # Unchained, turn 2 waits for the quiet after the chime and starts with its own audio.
    _, _, between = between_turns(down)
    quiet = [f for f in between if f is not chime]
    assert_quiet_before(pushed, quiet, t2, end1 + 0.1, GAP_S)
    assert type(second_audio) is TTSAudioRawFrame and second["gap_ms"] == 0
    assert fake.received_audio_ms(first["turn_id"]) == 1000  # the chime is in neither turn
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
    (t1, _), (t2, _) = first_audio_pushes(pushed)
    end1 = t1 + padded(300) / 1000
    ready = min(t for t, req_id in segments if req_id == turn_id)
    assert 0 < ready - end1 < WIDE_MIN_SILENCE_MS / 1000 and second["prev_turn_id"] == ""
    announced = turn_start_time(pushed, turn_id)
    # Not when its animation is in hand: when its release begins, after 440 ms of quiet (the
    # fill, R47), with one chunk of that left to play. Its audio follows in the same step (the
    # order below pins that; the clock bound only leaves room for a loaded machine between two
    # pushes, as any wait here may be longer).
    _, _, quiet = between_turns(down)
    assert_quiet_before(pushed, quiet, announced, end1, WIDE_GAP_S)
    assert 0 <= t2 - announced < LATE_S
    # At its release: turn_start, then every segment already in hand, in order, then its audio.
    at_release = [f for t, f in pushed if t >= announced]
    first_audio = next(i for i, f in enumerate(at_release) if isinstance(f, TTSAudioRawFrame))
    head = kinds(at_release[: first_audio + 1])
    assert head[0] == ("rtvi", "turn_start") and head[-1] == "TTSAudioRawFrame"
    assert set(head[1:-1]) == {("rtvi", "frames"), "TTSStartedFrame", ("rtvi", "turn_envelope")}
    assert head.index("TTSStartedFrame") == len(head) - 3  # all the frames go first
    assert head[-2] == ("rtvi", "turn_envelope")  # and the audio's envelope right before it
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
    (t1, _), _ = first_audio_pushes(pushed)
    assert turn_start_time(pushed, second["turn_id"]) < t1 + 0.6  # before turn 1 has played out


async def test_end_frame_waits_for_the_open_turn(fake):
    # No sleep after the turn: run_test's EndFrame arrives while the audio is still held.
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[TTSStartedFrame(), tone_frame(300), TTSStoppedFrame()],
    )
    k = kinds(down)
    assert k.count("TTSAudioRawFrame") == chunks(300) and k[-1] == ("rtvi", "turn_end")


# --- A TTS context that never says it stopped. A service built with push_stop_frames=False
# --- (Deepgram, ElevenLabs, Inworld, Soniox and Bland in Pipecat 1.15) pushes its TTSStoppedFrame
# --- on the provider's own flush message; lose that, and the context ends on the service's quiet
# --- timeout without one. The turn is stopped by the next context's TTSStartedFrame, or by the
# --- EndFrame. A regression holds the turn for good and parks the EndFrame: play's bound is the
# --- failure then.


async def test_a_context_that_never_stopped_is_stopped_by_the_next_one(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            TTSStartedFrame(context_id="c1"),
            tone_frame(300, context_id="c1"),  # and no TTSStoppedFrame for c1
            TTSStartedFrame(context_id="c2"),
            tone_frame(300, context_id="c2"),
            TTSStoppedFrame(context_id="c2"),
            message_pushed(pushed, "turn_end", count=2),
        ],
        wait_s=5.0,
    )
    order = [
        (type(f).__name__, f.context_id)
        for f in down
        if isinstance(f, (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame))
    ]
    assert [(key, len(list(run))) for key, run in itertools.groupby(order)] == [
        (("TTSStartedFrame", "c1"), 1),
        (("TTSAudioRawFrame", "c1"), chunks(300)),
        (("TTSStartedFrame", "c2"), 1),
        (("TTSAudioRawFrame", "c2"), chunks(300)),
        (("TTSStoppedFrame", "c2"), 1),
    ]
    assert [e["pcm_ms"] for e in messages(down, "turn_end")] == [padded(300)] * 2
    assert_turn_lifecycle(down)


async def test_a_context_that_never_stopped_is_stopped_by_the_end_frame(fake):
    # No TTSStoppedFrame: play's EndFrame follows the tone.
    down, _ = await play(make_proc(fake), [TTSStartedFrame(), tone_frame(300)], wait_s=5.0)
    k = kinds(down)
    assert k.count("TTSAudioRawFrame") == chunks(300) and k[-1] == ("rtvi", "turn_end")
    assert messages(down, "turn_end")[0]["pcm_ms"] == padded(300)
    assert_turn_lifecycle(down)


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
    assert 0 < k.count("TTSAudioRawFrame") < chunks(1000)
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
    assert 0 < k.count("TTSAudioRawFrame") < chunks(1000)
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
    assert 0 < k.count("TTSAudioRawFrame") < chunks(1000)
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
    k = kinds(without_fill(down))
    assert k.count("TTSAudioRawFrame") == chunks(500) and "TTSStoppedFrame" in k
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
    assert kinds(down).count("TTSAudioRawFrame") == chunks(200)
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
    assert fake.received_audio_ms(second["turn_id"]) == padded(300)
    assert_turn_lifecycle(down)


async def test_uninterruptible_frames_survive_an_interruption(fake):
    fake.first_segment_ms = 10_000  # the hold is never released: only the abort's rescue frees it
    keep = TextFrame("function result")
    keep.interruptible = False
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            TTSStartedFrame(),
            tone_frame(100),
            keep,
            # The rescue under test is the processor's: the frame must be in the turn's hold when
            # the interruption comes. Still on its way, it would be kept by Pipecat's own input
            # queue instead, and the test would pass for that reason.
            Until(
                "the frame to be held",
                lambda: proc._turns and any(item is keep for item in proc._turns[-1].hold),
            ),
            InterruptionFrame(),
            Until("the rescued frame to be pushed", lambda: any(f is keep for _, f in pushed)),
        ],
    )
    k = kinds(down)
    assert keep in down and "TTSAudioRawFrame" not in k and "TTSStartedFrame" not in k
    assert down.index(keep) > k.index("InterruptionFrame")
    # Rescued by the abort itself: pushed right behind the InterruptionFrame, in the same step.
    order = [f for _, f in pushed]
    assert isinstance(order[order.index(keep) - 1], InterruptionFrame)


async def test_hold_cap_starts_a_degraded_turn(fake):
    fake.first_segment_ms = 10_000
    proc = make_proc(fake, hold_cap_s=0.3)
    pushed = record_pushes(proc)
    at = {}

    def mark(name):
        at[name] = asyncio.get_running_loop().time()

    down, _ = await play(
        proc,
        [
            link_up(proc),
            Do(lambda: mark("audio")),
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
    # Announced at the cap. A slower cap would still come out within play's bound: when the
    # announcement came is what pins it.
    announced = turn_start_time(pushed, start["turn_id"])
    assert 0.3 - 0.01 <= announced - at["audio"] <= 0.3 + LATE_S
    k = kinds(without_fill(down))
    assert k.index(("rtvi", "turn_start")) < k.index("TTSAudioRawFrame")
    # The request is still closed, so the next one gets a new id, and the end flush that comes back
    # before the turn ends is delivered.
    assert fake.received[-1].client_audio_input.end
    (end,) = messages(down, "turn_end")
    assert end["segment_count"] == 1 and k[-1] == ("rtvi", "turn_end")
    assert_turn_lifecycle(down)


async def test_the_hold_cap_counts_from_the_turns_first_audio(fake):
    """A TTS that keeps streaming does not keep the cap from running out: the turn is announced
    ``hold_cap_s`` after its first audio, not after its latest."""
    fake.first_segment_ms = 10_000
    proc = make_proc(fake, hold_cap_s=0.3)
    pushed = record_pushes(proc)
    at = {}

    def mark(name):
        at[name] = asyncio.get_running_loop().time()

    down, _ = await play(
        proc,
        [
            link_up(proc),
            Do(lambda: mark("audio")),
            TTSStartedFrame(),
            # A second of audio at real time, 40 ms at a time: audio still comes well past the cap.
            *[step for _ in range(25) for step in (tone_frame(40), SleepFrame(0.04))],
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
    )
    (start,) = messages(down, "turn_start")
    assert start["degraded"] is True
    announced = turn_start_time(pushed, start["turn_id"])
    assert 0.3 - 0.01 <= announced - at["audio"] <= 0.3 + LATE_S
    assert kinds(down).count("TTSAudioRawFrame") == chunks(1000)
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
    assert kinds(without_fill(down))[-1] == ("rtvi", "turn_end")
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
    assert kinds(without_fill(down))[-1] == ("rtvi", "turn_end")


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
    tts = [i for i, kind in enumerate(k) if kind == "TTSAudioRawFrame"]
    # It plays from the moment the link goes, not once the hold cap (1.5 s) runs out: the TTS's
    # pause is already bridged with silence, between the first tone's chunks and the second's.
    assert "OutputAudioRawFrame" in k[tts[0] : tts[-1]]
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
    assert kinds(down).count("TTSAudioRawFrame") == chunks(200)
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
    assert 0 < sum(isinstance(f, TTSAudioRawFrame) for _, f in pushed) < chunks(2000)


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
        *["TTSAudioRawFrame"] * chunks(300),  # held in chunks, pushed as they are
        "TTSStoppedFrame",
        "TTSStartedFrame",  # and the next turn passes straight through, its frames untouched
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
        {"silence_chunk_ms": 60},  # not a multiple of output_chunk_ms
        {"output_chunk_ms": 0},
        {"output_chunk_ms": 25},  # not a multiple of 10 ms
        {"wire_timeout_s": 0},
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


# --- A half-open link: the socket takes nothing more and says nothing about it. Nothing the voice
# --- depends on waits for it; a write that does not return within wire_timeout_s is a loss.

WIRE_S = 0.5  # wire_timeout_s in these tests; a timer fires at its deadline or a hair early


def freeze(fake, proc):
    """Leave the processor's link half-open from here on (see ``conftest.half_open``)."""
    half_open(fake, proc._driven)


async def test_a_hung_write_never_stops_the_voice(fake):
    proc = make_proc(fake, wire_timeout_s=WIRE_S, reconnect_backoff_s=(5,))
    pushed = record_pushes(proc)
    at = {}

    def mark(name):
        at[name] = asyncio.get_running_loop().time()

    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),  # turn 1, with the link up
            tone_frame(200),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
            Do(lambda: freeze(fake, proc)),  # the link goes half-open, silently
            Do(lambda: mark("frozen")),
            TTSStartedFrame(),  # turn 2: its first write hangs on the socket...
            tone_frame(300),
            tts_audio_released(pushed, count=chunks(200) + 1),  # ...and it plays regardless
            Do(lambda: mark("released")),
            SleepFrame(0.5),  # the TTS is behind the room: silence bridges the gap
            tone_frame(200),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=2),
            TTSStartedFrame(),  # turn 3, started during the freeze
            tone_frame(200),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=3),
        ],
        wait_s=6.0,  # on a link nothing bounds, the failure is this timeout, not a hang
    )
    # The loss is found when the write runs out of time, and the turn plays from then on.
    assert WIRE_S - 0.01 <= at["released"] - at["frozen"] <= WIRE_S + LATE_S
    assert messages(down, "state") == [{"status": "STATUS_DEGRADED", "reason": "send timed out"}]
    starts, ends = messages(down, "turn_start"), messages(down, "turn_end")
    assert [s["degraded"] for s in starts] == [False, True, True]
    assert [e["segment_count"] for e in ends][1:] == [0, 0]
    k = kinds(down)
    assert k.count("TTSAudioRawFrame") == chunks(200) + chunks(500) + chunks(200)
    second = k.index("TTSStartedFrame", k.index("TTSStartedFrame") + 1)
    assert "OutputAudioRawFrame" in k[second : k.index("TTSStoppedFrame", second)]  # the bridge
    assert ends[1]["pcm_ms"] > padded(500)  # which inference would have seen too
    assert_turn_lifecycle(down)  # and the pipeline's EndFrame left: play() returned


async def test_a_barge_in_during_a_hung_write_is_not_held_back(fake):
    proc = make_proc(fake, wire_timeout_s=WIRE_S, reconnect_backoff_s=(5,))
    pushed = record_pushes(proc)
    at = {}

    def mark(name):
        at[name] = asyncio.get_running_loop().time()

    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            *[tone_frame(100) for _ in range(5)],
            tts_audio_released(pushed),  # the turn is announced and playing
            Do(lambda: freeze(fake, proc)),
            *[tone_frame(100) for _ in range(5)],  # the writer's next write hangs
            # A write on a healthy loopback link never yields, so the poll only ever sees the
            # writer on a write that is stuck.
            Until("the writer to be stuck on the socket", lambda: proc._writing is not None),
            Do(lambda: mark("interrupted")),
            InterruptionFrame(),
            interruption_pushed(pushed),
            Do(lambda: mark("passed on")),
            message_pushed(pushed, "state"),  # the loss, when the hung write runs out of time
        ],
        wait_s=6.0,
    )
    k = kinds(down)
    assert k.index(("rtvi", "turn_abort")) < k.index("InterruptionFrame")
    aborted = next(
        t for t, f in pushed if isinstance(f, RTVIServerMessageFrame) and "turn_abort" in f.data
    )
    assert aborted - at["interrupted"] < 0.3 and at["passed on"] - at["interrupted"] < 0.3
    assert messages(down, "turn_abort") == [{"turn_id": req_of(down)}]
    assert messages(down, "state") == [{"status": "STATUS_DEGRADED", "reason": "send timed out"}]
    assert_turn_lifecycle(down)


async def test_the_hold_cap_announces_a_turn_whose_first_write_hangs(fake):
    """The cap counts from the first audio whatever the link does: a turn whose first write is
    stuck on the socket plays, unanimated, when the cap runs out, before the write times out."""
    proc = make_proc(fake, hold_cap_s=0.3, wire_timeout_s=2.0, reconnect_backoff_s=(5,))
    pushed = record_pushes(proc)
    at = {}

    def mark(name):
        at[name] = asyncio.get_running_loop().time()

    down, _ = await play(
        proc,
        [
            link_up(proc),
            Do(lambda: freeze(fake, proc)),
            Do(lambda: mark("audio")),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_start"),
            Do(lambda: mark("announced")),
            message_pushed(pushed, "state"),  # the loss, at the wire timeout
            Do(lambda: mark("lost")),
        ],
        wait_s=6.0,
    )
    (start,) = messages(down, "turn_start")
    assert start["degraded"] is True
    assert 0.3 - 0.01 <= at["announced"] - at["audio"] <= 0.3 + LATE_S  # the cap, not the timeout
    assert at["lost"] - at["audio"] >= 2.0 - 0.01
    assert kinds(down).count("TTSAudioRawFrame") == chunks(300)
    (end,) = messages(down, "turn_end")
    assert end["segment_count"] == 0
    assert_turn_lifecycle(down)


# --- A turn that follows a turn without animation is not chained to it: the browser has no frame
# --- to place it from. It takes the unchained path, after enough quiet for its own onset.


async def test_a_turn_after_a_degraded_turn_is_not_chained(fake):
    fake.first_segment_ms = 10_000  # turn 1 gets no animation before its link goes
    proc = make_proc(fake, reconnect_backoff_s=(0.1,))
    pushed = record_pushes(proc)
    async with heartbeat() as ticks:
        down, _ = await play(
            proc,
            [
                link_up(proc),
                TTSStartedFrame(),
                tone_frame(1000),
                audio_at_driveningress(fake),
                Do(
                    lambda: fake.close_with(4001)
                ),  # turn 1 is degraded: it plays at once, unanimated
                message_pushed(pushed, "state"),
                message_pushed(pushed, "state", count=2),  # the link is back while it still plays
                Do(lambda: setattr(fake, "first_segment_ms", 200)),
                TTSStoppedFrame(),
                TTSStartedFrame(),  # turn 2, ready long before turn 1 has played out
                tone_frame(300),
                TTSStoppedFrame(),
                SleepFrame(2.0),
            ],
        )
    first, second = messages(down, "turn_start")
    assert first["degraded"] is True and second["degraded"] is False
    assert second["prev_turn_id"] == "" and second["prev_turn_pcm_ms"] == 0
    ends = messages(down, "turn_end")
    (t1, _), (t2, second_audio) = first_audio_pushes(pushed)
    end1 = t1 + ends[0]["pcm_ms"] / 1000  # turn 1's playout end
    # Turn 2 was ready while turn 1 still played, so the quiet is played as silence, right
    # behind turn 1's last sample: the room's gap is that silence, exactly (gap_ms).
    _, _, gap = between_turns(down)
    assert all(type(f) is OutputAudioRawFrame and set(f.audio) == {0} for f in gap)
    # Announced at its release, after the quiet: when the room has one chunk of it left to play.
    announced = turn_start_time(pushed, second["turn_id"])
    assert all(announced > t for t, f in pushed if any(f is g for g in gap))
    assert_quiet_before(pushed, gap, announced, end1, GAP_S)
    if held_up(ticks, end1 - CHUNK_MS / 1000, t2):  # the room may have run dry (accepted)
        assert second["gap_ms"] in (0, quiet_ms(gap))
    else:
        assert second["gap_ms"] == quiet_ms(gap) == EMITTED_GAP_MS
    assert type(second_audio) is TTSAudioRawFrame and 0 <= t2 - announced < LATE_S
    assert_turn_lifecycle(down)


# --- The plan's session ceiling: the server ends the session with close code 4002, on schedule.
# --- The reconnect skips the first backoff step then; any other loss waits it out.


def _token_times(monkeypatch):
    """Loop times of the fake's token requests, which start every connect; patched before the
    fake is entered, since its route is bound then."""
    times = []
    token = FakeDriveningress._token

    async def recorded(self, request):
        times.append(asyncio.get_running_loop().time())
        return await token(self, request)

    monkeypatch.setattr(FakeDriveningress, "_token", recorded)
    return times


@pytest.mark.parametrize(
    ("code", "waits"), [(4002, False), (4001, True)], ids=["cut on schedule", "a fault"]
)
async def test_the_reconnect_after_a_scheduled_cut_skips_the_first_backoff(
    monkeypatch, code, waits
):
    asked = _token_times(monkeypatch)
    backoff = 1.0
    async with FakeDriveningress() as fake:
        proc = make_proc(fake, reconnect_backoff_s=(backoff,))
        pushed = record_pushes(proc)
        lost = []
        on_down = proc._on_down

        def recorded(reason):
            lost.append(asyncio.get_running_loop().time())
            on_down(reason)

        proc._on_down = recorded  # setup() hands it to the link
        down, _ = await play(
            proc,
            [
                link_up(proc),
                Do(lambda: fake.close_with(code)),
                message_pushed(pushed, "state"),
                message_pushed(pushed, "state", count=2),  # connected again
            ],
        )
    assert fake.connections == 2 and len(asked) == 2 and len(lost) == 1
    reconnect = asked[1] - lost[0]
    if waits:
        assert reconnect >= backoff - 0.01
    else:
        assert reconnect < 0.1
    assert [s["status"] for s in messages(down, "state")] == ["STATUS_DEGRADED", "STATUS_OK"]


# --- Released chunks and the clock they keep, and the chunk the transport expects.


async def test_released_chunks_carry_the_time_of_the_tts_frame_they_start_in(fake):
    frames = []
    for i in range(3):
        frame = tone_frame(100)
        frame.pts = 1_000_000_000 + i * 100_000_000  # 100 ms apart on the TTS's clock, in ns
        frames.append(frame)
    down, _ = await run_test(
        make_proc(fake),
        frames_to_send=[TTSStartedFrame(), *frames, TTSStoppedFrame(), SleepFrame(1.0)],
    )
    released = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    # Chunk k starts 40k ms into the turn: in frame 40k // 100, 40k % 100 ms in; the padded last
    # chunk starts 280 ms in, 80 ms into the third frame.
    assert [f.pts for f in released] == [1_000_000_000 + k * 40_000_000 for k in range(chunks(300))]
    assert all(f.pts is None for f in down if isinstance(f, (TTSStartedFrame, TTSStoppedFrame)))


class Chunky(FrameProcessor):
    """Stands for an output transport that writes chunks of another size."""

    audio_chunk_size = 960  # 20 ms at 24 kHz: audio_out_10ms_chunks=2

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


async def test_a_transport_with_another_chunk_size_is_reported(fake, pipecat_log):
    await run_test(Pipeline([make_proc(fake), Chunky()]), frames_to_send=[])
    reported = [r for r in pipecat_log if "output_chunk_ms" in r["message"]]
    assert len(reported) == 1 and "1920 B" in reported[0]["message"]
    assert "960 B" in reported[0]["message"] and reported[0]["level"].name == "ERROR"
    pipecat_log[:] = [r for r in pipecat_log if r not in reported]  # expected, not a test error


# --- The turn envelope: the fingerprint of every sample the room hears, sent ahead of it, and the
# --- quiet between two turns emitted as real silence where that makes the gap exact.

HOP_MS, WINDOW_MS = 5, 10

# The quiet an unchained turn waits for, min_silence_ms (150) and one silence chunk (40), in whole
# output chunks: 190 ms is 4.75 chunks, so 5 chunks, 200 ms.
EMITTED_GAP_MS = math.ceil((150 + 40) / CHUNK_MS) * CHUNK_MS


def values_for(samples, rate=24000):
    """How many envelope values ``samples`` released samples give: whole windows only."""
    window, hop = WINDOW_MS * rate // 1000, HOP_MS * rate // 1000
    return 0 if samples < window else (samples - window) // hop + 1


def envelope_runs(frames, turn_id):
    """The ``turn_envelope`` payloads of one turn, in the order they were pushed."""
    return [
        p for m, p in avatar_messages(frames) if m == "turn_envelope" and p["turn_id"] == turn_id
    ]


def envelope_values(frames, turn_id):
    """The turn's envelope, the runs' values joined; each run must start where the last ended."""
    values = b""
    for run in envelope_runs(frames, turn_id):
        assert run["first_index"] == len(values), "the runs are not contiguous from 0"
        assert run["hop_ms"] == HOP_MS and run["window_ms"] == WINDOW_MS
        values += base64.b64decode(run["values"])
    return values


def turn_audio(frames):
    """Per turn, in order: the audio frames released between its TTSStartedFrame and its stop."""
    turns, current = [], None
    for f in frames:
        if isinstance(f, TTSStartedFrame):
            current = []
            turns.append(current)
        elif isinstance(f, TTSStoppedFrame):
            current = None
        elif current is not None and isinstance(f, OutputAudioRawFrame):
            current.append(f)
    return turns


def fingerprint(audio_frames, rate=24000):
    """The envelope recomputed from the bytes the room got."""
    return Envelope(rate).feed(b"".join(f.audio for f in audio_frames))


def described_before_released(frames):
    """Check that every audio frame of a turn is described before it goes; return how many.

    When a turn's frame is pushed, the turn's envelope already holds every window that ends
    inside what the room has been given of the turn, and every envelope came after its turn's
    ``turn_start``. Turn ``n`` is the one whose ``turn_start`` is ``n``-th: turns are announced
    in their order, which is the order they play in.
    """
    turn_ids = [s["turn_id"] for s in messages(frames, "turn_start")]
    announced, sent = set(), dict.fromkeys(turn_ids, 0)
    turn, inside, released, checked = -1, False, 0, 0
    for f in frames:
        if isinstance(f, RTVIServerMessageFrame) and "turn_start" in f.data:
            announced.add(f.data["turn_start"]["turn_id"])
        elif isinstance(f, RTVIServerMessageFrame) and "turn_envelope" in f.data:
            run = f.data["turn_envelope"]
            assert run["turn_id"] in announced  # never ahead of its turn_start
            sent[run["turn_id"]] += len(base64.b64decode(run["values"]))
        elif isinstance(f, TTSStartedFrame):
            turn, inside, released = turn + 1, True, 0
        elif isinstance(f, TTSStoppedFrame):
            inside = False  # what plays until the next turn starts is no turn's
        elif inside and isinstance(f, OutputAudioRawFrame):
            released += len(f.audio) // 2
            assert sent[turn_ids[turn]] >= values_for(released), (turn, released)
            checked += 1
    return checked


def between_turns(frames):
    """The audio frames pushed after the first turn's TTSStoppedFrame and before the second's
    TTSStartedFrame: what the room plays between the two turns' samples."""
    stopped = next(i for i, f in enumerate(frames) if isinstance(f, TTSStoppedFrame))
    started = next(
        i for i, f in enumerate(frames) if isinstance(f, TTSStartedFrame) and i > stopped
    )
    return (
        stopped,
        started,
        [f for f in frames[stopped:started] if isinstance(f, OutputAudioRawFrame)],
    )


async def test_turn_envelope_precedes_the_audio_it_describes(fake):
    """Every released chunk is described before it goes: when it is pushed, the turn's envelope
    already holds every window that ends inside what the room has been given, and every envelope
    comes after its turn's ``turn_start``."""
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    # The TTS streams at about twice real time, so the hold grows while the turn releases.
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            *[step for _ in range(10) for step in (tone_frame(100), SleepFrame(0.05))],
            TTSStoppedFrame(),
            TTSStartedFrame(),  # a second turn, chained: announced early, released after
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=2),
        ],
    )
    assert len(messages(down, "turn_start")) == 2
    assert kinds(down).count("TTSAudioRawFrame") == chunks(1000) + chunks(300)
    # Every chunk of both turns, and any stall silence a slow machine put between them.
    assert described_before_released(down) == sum(map(len, turn_audio(down)))


async def test_turn_envelope_indices_are_contiguous_and_match_released_bytes(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            *[step for _ in range(5) for step in (tone_frame(100), SleepFrame(0.05))],
            tone_frame(30),  # 530 ms: the turn is padded to 560
            TTSStoppedFrame(),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=2),
        ],
    )
    ends = messages(down, "turn_end")
    # A slow machine may add stall silence to the first: it is in the turn, and its envelope.
    assert ends[0]["pcm_ms"] >= padded(530) and ends[1]["pcm_ms"] == padded(300)
    for end, audio in zip(ends, turn_audio(down), strict=True):
        values = envelope_values(down, end["turn_id"])  # contiguous, 0-based, every turn
        assert len(values) == math.floor((end["pcm_ms"] - WINDOW_MS) / HOP_MS) + 1
        assert len(values) == values_for(sum(len(f.audio) for f in audio) // 2)


async def test_a_turn_held_whole_sends_its_envelope_in_one_message(fake):
    """The envelope goes per run of released chunks, not per chunk: a turn whose audio is all in
    hand when its release starts is described by one message, ahead of its first chunk."""
    fake.first_segment_ms = 2000
    proc = make_proc(fake, hold_cap_s=5.0)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            *[tone_frame(100) for _ in range(20)],
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
    )
    (run,) = envelope_runs(down, req_of(down))
    assert run["first_index"] == 0 and len(base64.b64decode(run["values"])) == values_for(48000)
    k = kinds(down)
    assert k.index(("rtvi", "turn_start")) < k.index(("rtvi", "turn_envelope"))
    assert k.index(("rtvi", "turn_envelope")) < k.index("TTSAudioRawFrame")


async def test_stall_silence_is_in_the_envelope_in_room_order(fake):
    proc = make_proc(fake, lead_ms=0)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(300),
            tts_audio_released(pushed),
            SleepFrame(0.8),  # the TTS falls behind the room: silence bridges the stall
            tone_frame(200),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
    )
    (audio,) = turn_audio(down)
    silence = [i for i, f in enumerate(audio) if type(f) is OutputAudioRawFrame]
    assert silence and silence == list(range(silence[0], silence[-1] + 1))  # one run, mid-turn
    assert isinstance(audio[0], TTSAudioRawFrame) and isinstance(audio[-1], TTSAudioRawFrame)
    values = envelope_values(down, req_of(down))
    assert values == fingerprint(audio)  # the room's samples, the silence in its place
    assert described_before_released(down) == len(audio)  # each chunk, silence too, ahead
    # The windows that lie wholly in the stall are the floor, between the tone's values.
    start = sum(len(f.audio) for f in audio[: silence[0]]) // 2
    end = start + sum(len(audio[i].audio) for i in silence) // 2
    hop, window = HOP_MS * 24, WINDOW_MS * 24
    inside = [k for k in range(len(values)) if start <= k * hop and k * hop + window <= end]
    assert inside and {values[k] for k in inside} == {0}
    assert values[inside[0] - 2] > 0 and values[inside[-1] + 2] > 0
    assert len(values) == values_for(messages(down, "turn_end")[0]["pcm_ms"] * 24)


async def test_unchained_turn_emits_its_gap_as_silence_and_reports_gap_ms(fake):
    """A turn unchained while the room still plays the turn before it (here because that turn
    had no animation, R28): the quiet it waits for is released as silence right behind the
    previous turn's last sample, so its length is exact, and ``turn_start`` reports it. The
    silence belongs to neither turn: not to an envelope, not to inference."""
    fake.first_segment_ms = 10_000  # turn 1 gets no animation before its link goes
    proc = make_proc(fake, reconnect_backoff_s=(0.1,))
    pushed = record_pushes(proc)
    async with heartbeat() as ticks:
        down, _ = await play(
            proc,
            [
                link_up(proc),
                TTSStartedFrame(),
                tone_frame(1000),
                audio_at_driveningress(fake),
                Do(lambda: fake.close_with(4001)),  # turn 1 is degraded: it plays at once
                message_pushed(pushed, "state"),
                message_pushed(pushed, "state", count=2),  # the link is back while it plays
                Do(lambda: setattr(fake, "first_segment_ms", 200)),
                TTSStoppedFrame(),
                TTSStartedFrame(),  # turn 2, ready long before turn 1 has played out
                tone_frame(300),
                TTSStoppedFrame(),
                message_pushed(pushed, "turn_end", count=2),
            ],
        )
    first, second = messages(down, "turn_start")
    assert first["degraded"] is True and second["prev_turn_id"] == ""
    stopped, started, gap = between_turns(down)
    t2 = next(t for t, f in pushed if f is turn_audio(down)[1][0])
    assert first["gap_ms"] == 0
    if held_up(ticks, t2 - 1.0, t2):  # the room may have run dry (an accepted limit)
        assert second["gap_ms"] in (0, quiet_ms(gap))
    else:
        assert second["gap_ms"] == quiet_ms(gap) == EMITTED_GAP_MS
    # Right behind turn 1's last sample (its TTSStoppedFrame follows it directly): only silence,
    # in whole output chunks, as long as gap_ms, in the class stall silence has.
    assert isinstance(down[stopped - 1], TTSAudioRawFrame)
    assert all(type(f) is OutputAudioRawFrame and set(f.audio) == {0} for f in gap)
    assert all(len(f.audio) == CHUNK_BYTES for f in gap)
    # Announced once its release starts, after the silence: nothing it could take for an onset.
    announce = next(
        i
        for i, f in enumerate(down)
        if isinstance(f, RTVIServerMessageFrame)
        and f.data.get("turn_start", {}).get("turn_id") == second["turn_id"]
    )
    assert down.index(gap[-1]) < announce < started
    # The silence went out without a gap of its own: pushed one chunk ahead, up to turn 2's audio.
    assert_bridged([t for t, f in pushed if any(f is g for g in gap)], t2, ticks)
    # Neither turn's envelope or inference stream has it.
    assert envelope_values(down, second["turn_id"]) == fingerprint(turn_audio(down)[1])
    assert fake.received_audio_ms(second["turn_id"]) == padded(300)
    sent = {m.client_audio_input.req_id for m in fake.received if m.HasField("client_audio_input")}
    assert sent == {first["turn_id"], second["turn_id"]}  # no request of its own either
    assert described_before_released(down) == sum(map(len, turn_audio(down)))
    # Turn 1 played unanimated and is described all the same.
    assert envelope_values(down, first["turn_id"]) == fingerprint(turn_audio(down)[0])
    assert_turn_lifecycle(down)


async def test_chained_turn_reports_gap_zero_and_no_silence(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(600),
            TTSStoppedFrame(),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=2),
        ],
    )
    first, second = messages(down, "turn_start")
    assert second["prev_turn_id"] == first["turn_id"]
    assert first["gap_ms"] == 0 and second["gap_ms"] == 0
    _, _, between = between_turns(down)
    assert between == []  # nothing between them: no fill, no silence
    assert all(type(f) is not OutputAudioRawFrame for audio in turn_audio(down) for f in audio)
    for start, audio in zip((first, second), turn_audio(down), strict=True):
        assert envelope_values(down, start["turn_id"]) == fingerprint(audio)


async def test_no_turn_envelope_after_an_abort(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            *[tone_frame(100) for _ in range(10)],
            tts_audio_released(pushed),  # the release is under way, its envelope sent
            SleepFrame(0.3),
            InterruptionFrame(),
            interruption_pushed(pushed),
            TTSStartedFrame(),  # the next turn starts over
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
    )
    first, second = messages(down, "turn_start")
    k = kinds(down)
    aborted = k.index(("rtvi", "turn_abort"))
    runs = [(i, p) for i, (m, p) in enumerate(zip(k, down)) if m == ("rtvi", "turn_envelope")]
    before = [p.data["turn_envelope"] for i, p in runs if i < aborted]
    after = [p.data["turn_envelope"] for i, p in runs if i > aborted]
    assert before and {p["turn_id"] for p in before} == {first["turn_id"]}
    assert after and {p["turn_id"] for p in after} == {second["turn_id"]}  # nothing of turn 1
    (_, audio) = turn_audio(down)
    assert envelope_values(down, second["turn_id"]) == fingerprint(audio)  # from index 0 again
    assert_turn_lifecycle(down)


async def test_degraded_turn_still_sends_its_envelope(fake):
    """No animation within the hold cap: the turn plays unanimated and is still described, so a
    client can line up whatever frames do arrive."""
    fake.first_segment_ms = 10_000
    proc = make_proc(fake, hold_cap_s=0.3)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(500),
            message_pushed(pushed, "turn_start"),  # the cap ran out; the end flush comes after
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
    )
    (start,) = messages(down, "turn_start")
    assert start["degraded"] is True
    (audio,) = turn_audio(down)
    values = envelope_values(down, start["turn_id"])
    assert values == fingerprint(audio) and len(values) == values_for(24 * padded(500))


async def test_a_turn_while_down_still_sends_its_envelope(fake):
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
            message_pushed(pushed, "turn_end"),
        ],
    )
    (start,) = messages(down, "turn_start")
    assert start["degraded"] is True
    (audio,) = turn_audio(down)
    assert envelope_values(down, start["turn_id"]) == fingerprint(audio) != b""


async def test_one_chunk_turn_has_seven_values(fake):
    """A turn of one output chunk (40 ms): windows start at 0, 5, ..., 30 ms, 7 values, the first
    at index 0, where keyframe 0 sits; sent before the chunk."""
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [TTSStartedFrame(), tone_frame(40), TTSStoppedFrame(), message_pushed(pushed, "turn_end")],
    )
    (run,) = envelope_runs(down, req_of(down))
    assert run["first_index"] == 0 and len(base64.b64decode(run["values"])) == 7
    k = kinds(down)
    assert k.count("TTSAudioRawFrame") == 1
    assert k.index(("rtvi", "turn_start")) < k.index(("rtvi", "turn_envelope"))
    assert k.index(("rtvi", "turn_envelope")) < k.index("TTSAudioRawFrame")


async def test_envelope_values_equal_fingerprint_on_the_pushed_bytes(fake):
    """Resampled TTS audio, a stall bridged with silence and the tail's padding: the envelope is
    the fingerprint of exactly the bytes pushed downstream for the turn, in their order."""
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(300, rate=22050),
            tts_audio_released(pushed),
            SleepFrame(0.7),  # a stall: silence goes out
            tone_frame(130, rate=22050),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
        ],
    )
    (audio,) = turn_audio(down)
    assert any(type(f) is OutputAudioRawFrame for f in audio)  # the stall's silence
    assert envelope_values(down, req_of(down)) == fingerprint(audio)
    assert described_before_released(down) == len(audio)
    (end,) = messages(down, "turn_end")
    assert end["pcm_ms"] == sum(len(f.audio) for f in audio) * 1000 / (2 * 24000)


async def test_audio_after_a_turns_end_is_not_in_its_envelope(fake):
    """A sound held behind a turn's last sample plays after it but is not the turn's: the turn's
    envelope ends with the turn's own audio, and the next turn's starts with its own."""
    chime = OutputAudioRawFrame(
        audio=pcm16(sine(100, 24000, 880)), sample_rate=24000, num_channels=1
    )
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            chime,  # held behind turn 1's end, in hand when turn 1's release starts
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=2),
        ],
    )
    assert any(f is chime for f in down)
    for start, audio in zip(messages(down, "turn_start"), turn_audio(down), strict=True):
        assert all(f is not chime for f in audio)
        assert envelope_values(down, start["turn_id"]) == fingerprint(audio)


# --- R47: after a turn's last sample the room is kept fed with silence (the fill) until the next
# --- turn starts, for at most a second, so the quiet before an unchained turn is known exactly.

FILL_CAP_MS = 1000


def held_up(ticks, start, end, more_than_s=CHUNK_MS / 1000):
    """Whether the loop was held up (not scheduled, frozen) for longer than ``more_than_s``
    between ``start`` and ``end``, from ``heartbeat``'s ticks: long enough for the room to run
    dry under a fill pushed one chunk ahead."""
    inside = [start] + [t for t in ticks if start < t < end] + [end]
    return max(b - a for a, b in itertools.pairwise(inside)) > more_than_s + 0.01


def turn_end_time(pushed, count):
    """When the ``count``-th ``turn_end`` was pushed."""
    ends = [t for t, f in pushed if isinstance(f, RTVIServerMessageFrame) and "turn_end" in f.data]
    return ends[count - 1]


async def test_an_unchained_turn_after_a_short_gap_hears_exactly_gap_ms_of_silence(fake):
    """Review Focus 3: turn 2's TTS starts 50 ms after turn 1 has played out. The room is kept
    fed after turn 1, so the quiet before turn 2 is our silence only: whole chunks, as long as
    the rule asks (min_silence_ms + one chunk, rounded up to a chunk), reported as gap_ms, and
    in neither turn's envelope or inference stream."""
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    segments = record_segments(proc)

    async def until_after_turn_1(ms):
        (t1, _), *_ = first_audio_pushes(pushed)
        loop = asyncio.get_running_loop()
        await asyncio.sleep(max(0.0, t1 + padded(300) / 1000 + ms / 1000 - loop.time()))

    async with heartbeat() as ticks:
        down, _ = await play(
            proc,
            [
                link_up(proc),
                TTSStartedFrame(),
                tone_frame(300),
                TTSStoppedFrame(),
                tts_audio_released(pushed),
                Do(lambda: until_after_turn_1(50)),
                TTSStartedFrame(),
                tone_frame(300),
                TTSStoppedFrame(),
                message_pushed(pushed, "turn_end", count=2),
            ],
        )
    first, second = messages(down, "turn_start")
    (t1, _), (t2, second_audio) = first_audio_pushes(pushed)
    end1 = t1 + padded(300) / 1000
    ready = min(t for t, req_id in segments if req_id == second["turn_id"])
    assert end1 < ready and second["prev_turn_id"] == ""  # ready after it played out: unchained
    _, _, quiet = between_turns(down)
    assert quiet and all(type(f) is OutputAudioRawFrame and not any(f.audio) for f in quiet)
    assert all(len(f.audio) == CHUNK_BYTES for f in quiet)
    assert_quiet_before(pushed, quiet, t2, end1, GAP_S)
    if held_up(ticks, end1 - CHUNK_MS / 1000, t2):
        # A loop held up for more than a chunk lets the room run dry, as for stall silence (an
        # accepted limit): the gap is then not all fill, or not known to be, and may say 0.
        assert second["gap_ms"] in (0, quiet_ms(quiet))
    else:
        # What the room hears between the two turns' samples is exactly gap_ms of silence.
        assert second["gap_ms"] == quiet_ms(quiet) == EMITTED_GAP_MS
        # (to the processor's own 1 ms tolerance on the clock)
        assert room_quiet(pushed, quiet, t2, end1) * 1000 == pytest.approx(second["gap_ms"], abs=1)
    for start, audio in zip((first, second), turn_audio(down), strict=True):
        assert envelope_values(down, start["turn_id"]) == fingerprint(audio)
        assert fake.received_audio_ms(start["turn_id"]) == padded(300)
    sent = {m.client_audio_input.req_id for m in fake.received if m.HasField("client_audio_input")}
    assert sent == {first["turn_id"], second["turn_id"]}
    assert type(second_audio) is TTSAudioRawFrame
    assert_turn_lifecycle(down)


async def test_a_turn_after_an_idle_longer_than_the_fill_reports_gap_zero(fake):
    """The fill stops a second after the turn's last sample; a turn 1.5 s later starts after the
    transport's own idle, whose length is not known, and reports no gap."""
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    async with heartbeat() as ticks:
        down, _ = await play(
            proc,
            [
                link_up(proc),
                TTSStartedFrame(),
                tone_frame(300),
                TTSStoppedFrame(),
                message_pushed(pushed, "turn_end"),
                SleepFrame(1.5),
                TTSStartedFrame(),
                tone_frame(300),
                TTSStoppedFrame(),
                message_pushed(pushed, "turn_end", count=2),
            ],
        )
    _, second = messages(down, "turn_start")
    assert second["prev_turn_id"] == "" and second["gap_ms"] == 0
    (t1, _), (t2, _) = first_audio_pushes(pushed)
    end1 = t1 + padded(300) / 1000
    _, _, fill = between_turns(down)
    assert all(type(f) is OutputAudioRawFrame and not any(f.audio) for f in fill)
    assert quiet_ms(fill) <= FILL_CAP_MS  # never more than the cap
    # The last fill chunk goes one chunk before the cap ends, counted from turn 1's last sample;
    # turn 1's turn_end, pushed at its playout end, bounds that from above on any machine.
    last = max(t for t, f in pushed if any(f is g for g in fill))
    assert last < turn_end_time(pushed, 1) + FILL_CAP_MS / 1000 < t2
    if not held_up(ticks, end1 - CHUNK_MS / 1000, end1 + FILL_CAP_MS / 1000):
        assert quiet_ms(fill) == FILL_CAP_MS  # and, with the loop there, went on up to it
    assert_turn_lifecycle(down)


async def test_an_interruption_during_the_fill_stops_it(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    down, _ = await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
            SleepFrame(0.2),  # the fill is under way
            InterruptionFrame(),
            interruption_pushed(pushed),
            SleepFrame(0.5),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end", count=2),
        ],
    )
    order = [f for _, f in pushed]
    barge = next(i for i, f in enumerate(order) if isinstance(f, InterruptionFrame))
    restart = next(i for i, f in enumerate(order) if isinstance(f, TTSStartedFrame) and i > barge)
    fill = fill_frames(order)
    assert any(order.index(f) < barge for f in fill)  # it was running...
    assert not [f for f in order[barge:restart] if isinstance(f, OutputAudioRawFrame)]  # ...no more
    _, second = messages(down, "turn_start")
    assert second["gap_ms"] == 0 and second["prev_turn_id"] == ""
    assert_turn_lifecycle(down)


async def test_an_end_frame_during_the_fill_ends_the_pipeline(fake):
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(300),
            TTSStoppedFrame(),
            message_pushed(pushed, "turn_end"),
            SleepFrame(0.3),  # the fill is under way when the EndFrame comes
        ],
    )  # play() returns only once the pipeline has ended
    order = [f for _, f in pushed]
    end = next(i for i, f in enumerate(order) if isinstance(f, EndFrame))
    fill = fill_frames(order)
    assert fill and all(order.index(f) < end for f in fill)
    assert order[end + 1 :] == []  # nothing after the EndFrame, the fill least of all
    assert quiet_ms(fill) < FILL_CAP_MS


async def test_the_fill_is_plain_output_audio_on_the_turns_track(fake):
    """The fill must not count as the bot speaking: it is not TTS audio, it is silence in whole
    output chunks, on the track of the turn it follows."""
    proc = make_proc(fake)
    pushed = record_pushes(proc)
    started = TTSStartedFrame()
    started.transport_destination = "voice"
    tone = tone_frame(300)
    tone.transport_destination = "voice"
    stopped = TTSStoppedFrame()
    stopped.transport_destination = "voice"
    down, _ = await play(
        proc,
        [
            link_up(proc),
            started,
            tone,
            stopped,
            message_pushed(pushed, "turn_end"),
            SleepFrame(0.3),
        ],
    )
    fill = fill_frames(down)
    assert fill
    assert all(type(f) is OutputAudioRawFrame and not isinstance(f, TTSAudioRawFrame) for f in fill)
    assert all(len(f.audio) == CHUNK_BYTES and not any(f.audio) for f in fill)
    assert {f.transport_destination for f in fill} == {"voice"}
    assert {f.sample_rate for f in fill} == {24000} and {f.num_channels for f in fill} == {1}


async def test_an_end_frame_waiting_for_the_last_turn_stops_the_fill(fake):
    """An EndFrame that arrives while the last turn still plays waits for that turn to end, as
    before, and no fill is started behind it: the pipeline ends at the turn's end."""
    proc = make_proc(fake, lead_ms=1000)  # the turn drains a second before it has played out
    pushed = record_pushes(proc)
    await play(
        proc,
        [
            link_up(proc),
            TTSStartedFrame(),
            tone_frame(600),
            TTSStoppedFrame(),
            Until(
                "the turn to drain", lambda: any(isinstance(f, TTSStoppedFrame) for _, f in pushed)
            ),
        ],
    )  # the EndFrame comes now, long before the turn has played out
    order = [f for _, f in pushed]
    end = next(i for i, f in enumerate(order) if isinstance(f, EndFrame))
    assert isinstance(order[end - 1], RTVIServerMessageFrame) and "turn_end" in order[end - 1].data
    assert fill_frames(order) == [] and order[end + 1 :] == []


class _Clock:
    """A loop stand-in whose time a test sets."""

    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now


def test_an_unchained_turn_starts_with_one_chunk_of_its_quiet_left(fake):
    """The fill already holds the quiet the rule asks for when the turn is ready, two chunks
    ahead of the room: the turn waits until one chunk of it is left, as a stall resumes, rather
    than going out up to the lead early. The clock is stepped by hand: in a pipeline the fill's
    own pacing hides the difference except inside one chunk."""
    proc = make_proc(fake)  # min_silence_ms 150 + silence_chunk_ms 40: 190 ms of quiet
    proc._loop = _Clock(10.0)
    proc._stall_ahead_s = CHUNK_MS / 1000
    turn = _Turn(TTSStartedFrame(), rate=24000, chunk_bytes=CHUNK_BYTES, tracker=None, sends=True)
    turn.ready = True
    # The last sound ended at 10.0; 200 ms of fill is queued behind it, without a break.
    proc._sound_end, proc._playout_end, proc._fed = 10.0, 10.2, True
    proc._loop.now = 10.12  # the fill's last chunk just went: two chunks left to play
    deadlines = []
    assert proc._may_start(turn, deadlines) is False
    assert deadlines == [pytest.approx(10.16)]
    proc._loop.now = 10.16  # one chunk left
    assert proc._may_start(turn, []) is True
    proc._sound_end, proc._playout_end, proc._loop.now = 10.0, 10.12, 10.08  # 120 ms: too little
    deadlines = []
    assert proc._may_start(turn, deadlines) is False and deadlines == [pytest.approx(10.12)]
