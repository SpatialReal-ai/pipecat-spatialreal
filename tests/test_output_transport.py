"""The processor's audio through Pipecat's real output transport.

``BaseOutputTransport`` cuts the audio it is given into chunks of ``audio_out_10ms_chunks`` x
10 ms, keeps a partial chunk until the next audio completes it, pads the partial chunk left at a
``TTSStoppedFrame`` with silence, and discards whatever is buffered when it dequeues that frame.
The processor releases whole chunks and pads every turn to a chunk boundary, in the room and at
inference alike, so that what the room plays is what inference saw: across two turns that follow
back to back in particular, where the second is placed from the first's exact length.
"""

import asyncio
import random
import struct

import pytest
from conftest import Until, link_up, play
from pipecat.frames.frames import (
    OutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

from pipecat_spatialreal import SpatialRealAvatarProcessor

RATE = 24000


@pytest.fixture(autouse=True)
def _every_test_checks_the_log(pipecat_log):
    """Every test here fails on a pipeline error or a task left running (see ``pipecat_log``)."""


class RoomTransport(BaseOutputTransport):
    """Pipecat's output transport over a room whose writes complete in real time, as Daily's do.

    What it would put on the room's track is kept in ``written``, in order.
    """

    def __init__(self, **params):
        super().__init__(
            TransportParams(
                audio_out_enabled=True,
                audio_out_sample_rate=RATE,
                audio_out_end_silence_secs=0,  # nothing after the pipeline's own audio
                **params,
            )
        )
        self.written = bytearray()
        self._play_end: float | None = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._play_end is None or now > self._play_end:
            self._play_end = now  # the room ran dry: this chunk plays from now
        self.written += frame.audio
        self._play_end += len(frame.audio) / (2 * RATE)
        await asyncio.sleep(max(0.0, self._play_end - loop.time()))  # done when it has played
        return True


def noise(ms, seed):
    """``ms`` of random samples: no two stretches alike, so a slice is found where it is."""
    rng = random.Random(seed)
    samples = [rng.randint(-9800, 9800) for _ in range(RATE * ms // 1000)]
    samples[0] = 5000  # never a zero sample first: padding is zeros
    return struct.pack(f"<{len(samples)}h", *samples)


def tts_frames(pcm, context_id, frame_ms):
    size = 2 * RATE * frame_ms // 1000
    return [
        TTSAudioRawFrame(
            audio=pcm[i : i + size], sample_rate=RATE, num_channels=1, context_id=context_id
        )
        for i in range(0, len(pcm), size)
    ]


def messages(frames, member):
    return [
        f.data[member] for f in frames if isinstance(f, RTVIServerMessageFrame) and member in f.data
    ]


async def test_two_turns_back_to_back_reach_the_room_as_inference_saw_them(fake):
    """Neither turn is a whole number of chunks, so the transport would pad the first and cut
    the second; the second's animation is in hand while the first still plays, so it chains."""
    proc = SpatialRealAvatarProcessor(
        api_key="k",
        app_id="a",
        avatar_id="av",
        console_endpoint_url=fake.console_url,
        ingress_endpoint_url=fake.ingress_url,
    )
    room = RoomTransport()
    pushed = []

    @proc.event_handler("on_after_push_frame")
    async def _record(processor, frame):
        pushed.append(frame)

    pcm_a, pcm_b = noise(700, 1), noise(500, 2)
    down, _ = await play(
        Pipeline([proc, room]),
        [
            link_up(proc),
            TTSStartedFrame(context_id="A"),
            *tts_frames(pcm_a, "A", 100),
            TTSStoppedFrame(context_id="A"),
            TTSStartedFrame(context_id="B"),
            *tts_frames(pcm_b, "B", 100),
            TTSStoppedFrame(context_id="B"),
            Until(
                "both turns to end",
                lambda: sum("turn_end" in f.data for f in pushed if hasattr(f, "data")) == 2,
            ),
        ],
    )
    first, second = messages(down, "turn_start")
    assert second["prev_turn_id"] == first["turn_id"]  # chained: placed from A's length
    heard_a = fake.received_audio_bytes(first["turn_id"])
    heard_b = fake.received_audio_bytes(second["turn_id"])
    # Each turn is its audio, then silence to the chunk boundary, at inference...
    assert heard_a == pcm_a + bytes(2 * RATE * 20 // 1000)  # 700 ms -> 720
    assert heard_b == pcm_b + bytes(2 * RATE * 20 // 1000)  # 500 ms -> 520
    # ...and the room's track is exactly those two, one after the other: nothing padded by the
    # transport before B, nothing of B discarded at A's end.
    written = bytes(room.written)
    assert written == heard_a + heard_b
    assert written.index(pcm_b) == len(heard_a)
    assert second["prev_turn_pcm_ms"] == len(heard_a) * 1000 / (2 * RATE) == 720.0
    assert [e["pcm_ms"] for e in messages(down, "turn_end")] == [720.0, 520.0]


async def test_a_transport_with_another_chunk_is_reported(fake, pipecat_log):
    proc = SpatialRealAvatarProcessor(
        api_key="k",
        app_id="a",
        avatar_id="av",
        console_endpoint_url=fake.console_url,
        ingress_endpoint_url=fake.ingress_url,
    )
    await play(Pipeline([proc, RoomTransport(audio_out_10ms_chunks=2)]), [link_up(proc)])
    reported = [r for r in pipecat_log if "output_chunk_ms" in r["message"]]
    assert len(reported) == 1 and "960 B" in reported[0]["message"]
    pipecat_log[:] = [r for r in pipecat_log if r not in reported]  # expected, not a test error
