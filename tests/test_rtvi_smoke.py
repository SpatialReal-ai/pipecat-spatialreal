"""The smoke test: with RTVI on, the avatar messages reach the client.

Everywhere else the processor runs with RTVI off, and the tests stop at the
``RTVIServerMessageFrame``s it pushes. In a bot RTVI is on: ``PipelineWorker`` adds it unless told
not to, and the example bot keeps that default. Pipecat then turns each of those frames into an RTVI
``server-message``: the RTVI observer sees the frame go by and has the RTVI processor at the head
of the pipeline push it as an urgent transport message, which the output transport sends at once
instead of queueing it behind the audio. Nothing in the processor's own tests would notice a Pipecat
release that stopped doing that, and the weekly compatibility job would then certify a release on
which the avatar never moves.

Here that path runs end to end, with no key and nothing leaving the machine: a TTS service that
speaks a tone, the processor, the fake driveningress, and Pipecat's eval transport, whose serializer
writes the RTVI messages and the bot's audio to one WebSocket in the order the transport sends
them. The WebSocket is an in-memory ASGI connection, so the test reads what a client would read.

If a Pipecat release reshapes the eval transport itself, this test breaks for a reason that lies
in the test, as ``play`` in ``conftest.py`` would: adapt it to the new shape.
"""

import asyncio
import base64
import json
import math
import struct
import warnings
from collections.abc import AsyncGenerator

import pytest
from conftest import WAIT_S, Until, link_up, until
from fake_driveningress import FakeDriveningress
from loguru import logger
from pipecat.frames.frames import EndFrame, Frame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIServerMessageFrame
from pipecat.services.settings import TTSSettings
from pipecat.workers.runner import WorkerRunner
from starlette.websockets import WebSocket

from pipecat_spatialreal import SpatialRealAvatarProcessor
from pipecat_spatialreal.envelope import TAG

# Importing Pipecat's eval transport (built on the legacy websockets server) and its TTS base class
# (which defines a deprecated subclass) raises deprecation warnings that are not ours to fix. They
# are dropped here, as test_example_token_route.py drops the same ones, so the package's own warning
# filter stays as strict as it is.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from pipecat.evals.fastapi_transport import EvalFastAPIWebsocketTransport
    from pipecat.evals.serializer import EVAL_BOT_AUDIO_TYPE, EvalConnectionFlags, EvalSerializer
    from pipecat.evals.transport import EvalTransportParams
    from pipecat.services.tts_service import TTSService

RATE = 24000
FRAME_MS = 40
# Two responses back to back: the first is announced as its release starts, the second follows it
# with no gap, so it is chained to it and announced as soon as its animation is in hand. The second
# can be chained only while the first still plays, which gives its animation two seconds to come
# back on a starved machine.
TURNS_MS = (2000, 400)

# What the browser's RTVI client matches a server message on (pipecat-client-js onServerMessage).
RTVI_LABEL = "rtvi-ai"
SERVER_MESSAGE = "server-message"

# The connection the runner would hand the bot, as an ASGI scope.
SCOPE = {
    "type": "websocket",
    "scheme": "ws",
    "path": "/ws",
    "root_path": "",
    "query_string": b"",
    "headers": [],
    "client": ("127.0.0.1", 50000),
    "server": ("127.0.0.1", 7860),
    "subprotocols": [],
}


def tone(ms: int) -> bytes:
    """``ms`` of a 440 Hz tone at amplitude 9800, PCM16 mono at ``RATE``."""
    count = RATE * ms // 1000
    return struct.pack(
        f"<{count}h", *(round(9800 * math.sin(2 * math.pi * 440 * i / RATE)) for i in range(count))
    )


class ToneTTS(TTSService):
    """A TTS service shaped like Pipecat's HTTP ones that speaks a tone instead of the text.

    Every ``TTSSpeakFrame`` is a context of its own: the base class pushes its ``TTSStartedFrame`` and
    ``TTSStoppedFrame`` around the audio ``run_tts`` yields, which is the next length of
    ``turns_ms`` in 40 ms frames.
    """

    def __init__(self, turns_ms: tuple[int, ...]):
        """Speak the lengths in ``turns_ms``, one per ``TTSSpeakFrame``, in order."""
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=RATE,
            settings=TTSSettings(model=None, voice=None, language=None),
        )
        self._turns_ms = list(turns_ms)

    def can_generate_metrics(self) -> bool:
        """No metrics: nothing is measured here."""
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Yield the next turn's tone, whatever ``text`` says."""
        for _ in range(self._turns_ms.pop(0) // FRAME_MS):
            yield TTSAudioRawFrame(
                audio=tone(FRAME_MS), sample_rate=RATE, num_channels=1, context_id=context_id
            )


class Client:
    """The far end of an in-memory ASGI WebSocket: what the bot sent, in the order it sent it."""

    def __init__(self):
        """Connect: the first thing the bot reads is the connection's opening."""
        self._inbox: asyncio.Queue[dict] = asyncio.Queue()
        self._inbox.put_nowait({"type": "websocket.connect"})
        self.received: list[dict] = []  # every message the bot sent, parsed

    async def receive(self) -> dict:
        """ASGI ``receive``: the bot waits here for the client, which says nothing."""
        return await self._inbox.get()

    async def send(self, message: dict) -> None:
        """ASGI ``send``: keep every message the bot sends."""
        if message["type"] == "websocket.send":
            self.received.append(json.loads(message["text"]))

    def leave(self) -> None:
        """Hang up, for a bot still listening."""
        self._inbox.put_nowait({"type": "websocket.disconnect", "code": 1000})

    def avatar_messages(self) -> list[tuple[int, dict]]:
        """``(position, envelope)`` of every RTVI server message that carries an avatar envelope."""
        return [
            (at, message["data"])
            for at, message in enumerate(self.received)
            if message.get("label") == RTVI_LABEL
            and message.get("type") == SERVER_MESSAGE
            and isinstance(message.get("data"), dict)
            and message["data"].get("tag") == TAG
        ]

    def audio(self) -> list[tuple[int, int, int]]:
        """``(position, first byte, length)`` of every piece of the bot's audio, in the room's order."""
        pieces, offset = [], 0
        for at, message in enumerate(self.received):
            if message.get("type") == EVAL_BOT_AUDIO_TYPE:
                length = len(base64.b64decode(message["data"]["audio"]))
                pieces.append((at, offset, length))
                offset += length
        return pieces


async def reach(step: Until) -> None:
    """Wait for ``step`` the way ``play`` does, and name it if it never comes."""
    try:
        await until(step.condition, timeout=WAIT_S)
    except AssertionError:
        raise AssertionError(f"timed out waiting for {step.what}") from None


@pytest.fixture
async def backend():
    """A fake driveningress that answers a turn with two segments: its first 200 ms, then the rest.

    On a slow link every answer costs time, and the second turn can only be chained while the first
    one still plays. Two answers per turn keep the second turn's first segment that close behind
    the first turn's, however slow the machine.
    """
    async with FakeDriveningress(segment_ms=60_000) as fake:
        yield fake


@pytest.fixture
def pipecat_log():
    """Every warning and error logged during the test."""
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="WARNING")
    yield records
    logger.remove(sink)


async def test_avatar_messages_reach_the_client_through_rtvi_ahead_of_queued_audio(
    backend, pipecat_log
):
    """Every avatar message reaches the client as an RTVI server message, ahead of queued audio.

    The processor runs with a lead of the whole first turn, so the transport holds all of that
    turn's audio in its queue by the time either turn is announced. A message that went through
    that queue would reach the client after the first turn's last audio; one that is sent urgently,
    as the design relies on, reaches it within that audio.

    A turn that is not chained is announced as its release starts, and Pipecat 1.15 hands its
    ``turn_start`` to the transport just after the turn's first audio, not before it: the RTVI
    observer runs in a task of its own and its urgent frame sets off again from the head of the
    pipeline, while the audio goes straight on to the transport (here the first two 40 ms chunks
    reach the client first). That is allowed for: the web SDK accepts an onset
    up to 100 ms before the message arrives and otherwise looks back through its last second of
    audio. For that turn the test therefore asks only that the message did not wait behind the
    turn's audio. A chained turn is announced while the previous one plays, so its ``turn_start``
    must reach the client before the turn's own first audio.
    """
    client = Client()
    websocket = WebSocket(SCOPE, client.receive, client.send)
    await websocket.accept()  # the runner hands the bot an accepted WebSocket
    transport = EvalFastAPIWebsocketTransport(
        websocket,
        EvalTransportParams(
            audio_out_enabled=True, serializer=EvalSerializer(), allowed_origins=[]
        ),
        EvalConnectionFlags(capture_bot_audio=True),  # the bot's audio goes on the wire too
    )
    proc = SpatialRealAvatarProcessor(
        api_key="k",
        app_id="a",
        avatar_id="av",
        console_endpoint_url=backend.console_url,
        ingress_endpoint_url=backend.ingress_url,
        lead_ms=TURNS_MS[0],
    )
    pushed = []  # the envelopes the processor pushed, in order

    @proc.event_handler("on_after_push_frame")
    async def _record(processor, frame):
        if isinstance(frame, RTVIServerMessageFrame):
            pushed.append(frame.data)

    # RTVI as a bot gets it: PipelineWorker's own default, which the example bot keeps.
    worker = PipelineWorker(
        Pipeline([transport.input(), ToneTTS(TURNS_MS), proc, transport.output()]),
        params=PipelineParams(audio_out_sample_rate=RATE),
        cancel_on_idle_timeout=False,
    )
    try:
        rtvi = worker.rtvi
    except Exception:  # noqa: BLE001 - Pipecat raises a bare Exception when RTVI is off
        rtvi = None
    assert isinstance(rtvi, RTVIProcessor), "PipelineWorker no longer turns RTVI on by default"

    connected = asyncio.Event()

    @transport.event_handler("on_client_connected")
    async def _on_client_connected(transport, websocket):
        connected.set()

    def turns_ended() -> int:
        return sum("turn_end" in envelope for _, envelope in client.avatar_messages())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        await reach(Until("the client to connect", connected.is_set))
        await reach(link_up(proc))
        await worker.queue_frames([TTSSpeakFrame("One."), TTSSpeakFrame("Two.")])
        await reach(
            Until(
                "both turn_end messages to reach the client as RTVI server messages",
                lambda: turns_ended() == len(TURNS_MS),
            )
        )
        await worker.queue_frame(EndFrame())
        done, _ = await asyncio.wait({running}, timeout=WAIT_S)
        assert done, f"the pipeline did not end within {WAIT_S:g} s of its EndFrame"
        running.result()
    except BaseException:
        # As in play(): a second cancel cuts through a worker that waits for its EndFrame.
        for _ in range(2):
            running.cancel()
            await asyncio.wait({running}, timeout=0.5)
            if running.done():
                break
        if running.done() and not running.cancelled():
            running.exception()  # retrieved: the error that matters is the one being raised
        raise
    finally:
        client.leave()

    # Every avatar message the processor pushed reached the client as an RTVI server message, in
    # the order it was pushed, with nothing lost or changed by RTVI's serialization or the
    # transport's.
    avatar = client.avatar_messages()
    assert len(avatar) == len(pushed), f"{len(avatar)} of {len(pushed)} messages reached the client"
    assert [envelope for _, envelope in avatar] == pushed
    starts = [(at, envelope["turn_start"]) for at, envelope in avatar if "turn_start" in envelope]
    assert len(starts) == 2, "two turns, two turn_start messages"
    (start_1, first), (start_2, second) = starts
    assert first["prev_turn_id"] == "" and not first["degraded"]
    assert second["prev_turn_id"] == first["turn_id"] and not second["degraded"]
    assert second["prev_turn_pcm_ms"] == TURNS_MS[0]
    for turn in (first, second):
        kinds = [next(iter(set(e) - {"tag"})) for _, e in avatar if _turn_of(e) == turn["turn_id"]]
        assert kinds[0] == "turn_start" and kinds[-1] == "turn_end" and "frames" in kinds

    # Where each turn's audio reached the client. The second turn starts prev_turn_pcm_ms into the
    # room's audio.
    audio = client.audio()
    second_starts_at = round(second["prev_turn_pcm_ms"] * RATE / 1000) * 2

    def audio_holding(byte: int) -> int:
        return next(at for at, offset, length in audio if offset + length > byte)

    last_audio_1 = audio_holding(second_starts_at - 1)
    first_audio_2 = audio_holding(second_starts_at)
    assert start_1 < last_audio_1, "the first turn_start waited behind the turn's audio"
    assert start_2 < last_audio_1, "the second turn_start waited behind the first turn's audio"
    assert start_2 < first_audio_2, "the chained turn_start reached the client after its audio"

    problems = [
        record["message"]
        for record in pipecat_log
        if record["level"].no >= 40 or "dangling" in record["message"]
    ]
    assert problems == []


def _turn_of(envelope: dict) -> str:
    """The ``turn_id`` an envelope belongs to, or ``""`` for a ``state`` message."""
    (member,) = set(envelope) - {"tag"}
    return envelope[member].get("turn_id", "")


async def test_an_audio_request_from_the_client_turns_on_audio_mode(backend, pipecat_log):
    """A client's ``audio_request``, sent as an RTVI client message, reaches the processor.

    The processor sits after the TTS; the request comes in at the transport, where Pipecat's RTVI
    processor turns it into an ``RTVIClientMessageFrame`` and pushes it downstream. If a Pipecat
    release stopped doing that (or consumed the frame on the way), a phone would never get the
    bot's audio. Here the client sends the request over the WebSocket, as a mobile client would,
    and must get ``audio_mode`` and then the turn's ``audio`` back.
    """
    client = Client()
    websocket = WebSocket(SCOPE, client.receive, client.send)
    await websocket.accept()
    transport = EvalFastAPIWebsocketTransport(
        websocket,
        EvalTransportParams(
            audio_out_enabled=True, serializer=EvalSerializer(), allowed_origins=[]
        ),
        EvalConnectionFlags(capture_bot_audio=False),
    )
    proc = SpatialRealAvatarProcessor(
        api_key="k",
        app_id="a",
        avatar_id="av",
        console_endpoint_url=backend.console_url,
        ingress_endpoint_url=backend.ingress_url,
    )
    worker = PipelineWorker(
        Pipeline([transport.input(), ToneTTS((400,)), proc, transport.output()]),
        params=PipelineParams(audio_out_sample_rate=RATE),
        cancel_on_idle_timeout=False,
    )
    connected = asyncio.Event()

    @transport.event_handler("on_client_connected")
    async def _on_client_connected(transport, websocket):
        connected.set()

    def say(message: dict) -> None:
        client._inbox.put_nowait({"type": "websocket.receive", "text": json.dumps(message)})

    def kinds() -> list[str]:
        return [next(iter(set(e) - {"tag"})) for _, e in client.avatar_messages()]

    runner = WorkerRunner()
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        await reach(Until("the client to connect", connected.is_set))
        await reach(link_up(proc))
        say({"label": RTVI_LABEL, "type": "client-ready", "id": "r1", "data": {"version": "1.0.0"}})
        say(
            {
                "label": RTVI_LABEL,
                "type": "client-message",
                "id": "m1",
                "data": {"t": "spatialreal-avatar", "d": {"tag": TAG, "audio_request": {}}},
            }
        )
        await reach(Until("the request to reach the processor", lambda: proc._audio_pending))
        await worker.queue_frames([TTSSpeakFrame("One.")])
        await reach(Until("the turn to end", lambda: "turn_end" in kinds()))
        await worker.queue_frame(EndFrame())
        done, _ = await asyncio.wait({running}, timeout=WAIT_S)
        assert done, f"the pipeline did not end within {WAIT_S:g} s of its EndFrame"
        running.result()
    except BaseException:
        for _ in range(2):
            running.cancel()
            await asyncio.wait({running}, timeout=0.5)
            if running.done():
                break
        if running.done() and not running.cancelled():
            running.exception()
        raise
    finally:
        client.leave()

    envelopes = [e for _, e in client.avatar_messages()]
    (mode,) = [e["audio_mode"] for e in envelopes if "audio_mode" in e]
    (start,) = [e["turn_start"] for e in envelopes if "turn_start" in e]
    assert mode == {"active": True, "first_turn_id": start["turn_id"], "sample_rate": RATE}
    audio = [e["audio"] for e in envelopes if "audio" in e]
    assert audio and audio[-1]["end"] and all(a["turn_id"] == start["turn_id"] for a in audio)
    assert kinds().index("audio_mode") < kinds().index("audio")
    assert base64.b64decode(audio[0]["data"])[:4] == b"OggS"
    problems = [
        record["message"]
        for record in pipecat_log
        if record["level"].no >= 40 or "dangling" in record["message"]
    ]
    assert problems == []
