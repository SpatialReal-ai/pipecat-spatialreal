"""The Pipecat processor that drives a SpatialReal avatar from a bot's TTS audio.

It sits between the TTS service and ``transport.output()``. Each TTS context, from
``TTSStartedFrame`` to ``TTSStoppedFrame``, is one turn of the avatar envelope
(``pipecat/v1/avatar.proto``) and one driveningress request:

* The turn's audio goes to driveningress as it arrives, at TTS speed, and is held back here.
* Once the first animation segment is back, the held audio is released downstream at real time,
  ``lead_ms`` ahead so the transport never runs dry, right after the turn is announced to the
  browser (``turn_start``, then the keyframes in hand, as RTVI server messages). A turn that
  follows the previous one back to back is announced earlier, as soon as its animation is in
  hand. Keyframes that come later are pushed at once, ahead of the audio they belong to.
* When the TTS falls behind playback, the processor plays silence and sends the same silence to
  driveningress, so the audio inference saw and the audio in the room stay sample-identical.

The audio is released in whole output chunks (``output_chunk_ms``, the transport's own chunk),
and every turn ends on a chunk boundary: its tail is padded with silence that inference sees
and that counts into the turn's length. Pipecat's output transport cuts what it is given into
chunks of that size, keeps a partial chunk until the next audio completes it, pads it when the
TTS context ends and discards it when it dequeues that end. Whole chunks leave it nothing to
pad or discard, so what the room plays is the audio inference saw, and a turn that follows
back to back starts in the room exactly where ``prev_turn_pcm_ms`` says.

Pipecat's own frames keep their order: everything a turn carries (``TTSStartedFrame``, text,
``TTSStoppedFrame``, ...) reaches the transport in its original order, only later, so the bot
speaks, is interrupted and has its context aggregated as before, at the delayed playout.

One difference stays. Frames the TTS stamped with a presentation time on its own clock (word
``TTSTextFrame``s, and ``LLMFullResponseEndFrame`` from services with word timestamps) are past
that time when they are released, so the transport passes them on as soon as they arrive instead
of at their audio: up to ``min(hold, lead_ms)`` ahead of it. Transcripts and the assistant
aggregator therefore run up to ``lead_ms`` (200 ms by default) ahead of the speech they report.
A released audio chunk keeps that clock too: its presentation time is the time of the TTS frame
its first sample came from, plus that sample's offset in the frame.

Without an animation source (driveningress unreachable, or too slow) the audio still plays,
unanimated. Nothing that plays audio ever waits for the driveningress socket: what goes to
inference is handed to a writer task, which performs one bounded write at a time, and a write
that does not return within ``wire_timeout_s`` is a loss like any other.
"""

import asyncio
import dataclasses
import sys
from array import array
from collections import deque

import spatialreal
from loguru import logger
from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    SystemFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

from pipecat_spatialreal import envelope
from pipecat_spatialreal.driven import DrivenIngress, Segment, _describe, close_code
from pipecat_spatialreal.onset import OnsetTracker
from pipecat_spatialreal.proto.generated import avatar_pb2

# Longest wait for one connect attempt; the SDK's handshake has no deadline of its own.
CONNECT_TIMEOUT_S = 10.0

# Timers may fire this early; a deadline this close counts as reached.
_TOLERANCE_S = 0.001

# Writes waiting for the writer task, at most. The writer keeps up with the TTS on a healthy
# link; a backlog this long means it is stuck, and the write it is stuck on is about to time out.
_WRITE_BACKLOG = 1024

_STATUS_OK = avatar_pb2.State.STATUS_OK
_STATUS_DEGRADED = avatar_pb2.State.STATUS_DEGRADED


class _EndOfAudio:
    """Marks where a turn's own audio ends in its hold; what follows it only keeps its place."""

    def __repr__(self) -> str:
        return "<end of turn audio>"


_END_OF_AUDIO = _EndOfAudio()


class _Chunker:
    """Cuts a turn's audio into whole output chunks, in order, and carries the rest.

    What does not fill a chunk waits for the next audio to complete it, so the hold and the
    inference stream only ever get whole chunks; the turn's end pads what is left with silence.
    Each chunk remembers the frame its first byte came from and that byte's offset in the frame,
    for the chunk's presentation time.
    """

    def __init__(self, size: int):
        self.size = size
        self.carry = bytearray()
        self.source: OutputAudioRawFrame | None = None  # the frame the carry's first byte is from
        self.offset = 0  # that byte's offset in the source's audio

    def feed(
        self, pcm: bytes, source: OutputAudioRawFrame
    ) -> list[tuple[bytes, OutputAudioRawFrame, int]]:
        """Return the whole chunks that ``pcm`` completes, behind the carry; carry the rest."""
        held = len(self.carry)
        self.carry.extend(pcm)
        chunks = []
        position = 0
        while len(self.carry) - position >= self.size:
            if position < held:
                origin, offset = self.source, self.offset + position
            else:
                origin, offset = source, position - held
            assert origin is not None
            chunks.append((bytes(self.carry[position : position + self.size]), origin, offset))
            position += self.size
        del self.carry[:position]
        if self.carry:
            if position < held:
                self.offset += position
            else:
                self.source, self.offset = source, position - held
        return chunks

    def flush(self) -> tuple[bytes, OutputAudioRawFrame, int] | None:
        """Return the carry padded with silence to a whole chunk, or None when there is none."""
        if not self.carry:
            return None
        chunk = bytes(self.carry) + bytes(self.size - len(self.carry))
        self.carry.clear()
        assert self.source is not None
        return chunk, self.source, self.offset


class _Turn:
    """One TTS context: one turn of the envelope and, while it can, one driveningress request."""

    def __init__(
        self,
        started: TTSStartedFrame,
        *,
        rate: int,
        chunk_bytes: int,
        tracker: OnsetTracker,
        sends: bool,
    ):
        self.started = started
        self.rate = rate
        self.tracker = tracker
        self.chunker = _Chunker(chunk_bytes)
        # Every frame that arrived while the turn was open, in arrival order, waiting for its
        # release. _END_OF_AUDIO follows the turn's last audio once the turn has stopped.
        self.hold: deque[Frame | _EndOfAudio] = deque([started])
        # Audio not already mono at the output rate goes through one stream resampler per input
        # rate. stream_rate is the rate the last such audio came at, whose filter still holds its
        # tail; last_audio[rate] is that stream's latest frame, which dresses the tail.
        self.resamplers: dict[int, BaseAudioResampler] = {}
        self.stream_rate: int | None = None
        self.last_audio: dict[int, OutputAudioRawFrame] = {}
        self.turn_id = ""  # its id in the envelope: the request id, or one made up for it
        self.req_id = ""  # its driveningress request, once the first write returned
        self.sends = sends  # its audio still goes to driveningress
        self.sent = False  # a write was queued for it on the current session
        self.degraded = not sends  # no animation is expected for it, or no more of it
        self.ready = False  # its animation is in hand (or given up on): it may be announced
        self.announced = False  # turn_start pushed, or released without one (silent)
        self.chained = False  # its turn_start names the turn it follows back to back
        self.release_at: float | None = None  # loop time its release may start, once decided
        self.silent = False  # it stopped without audio: released, never announced
        self.stopped = False  # its audio is complete: _END_OF_AUDIO is in the hold
        self.finished = False  # its last sample has been released
        self.drained = False  # finished, and nothing of its hold is left
        self.first_audio_at: float | None = None  # loop time; the hold cap counts from here
        self.playout_end = 0.0  # loop time its last sample has played by, once finished
        self.pcm_bytes = (
            0  # its audio in whole chunks, stall silence and the tail's padding included
        )
        self.onsets: list[float] = []  # every onset found, ms from its first sample
        self.onsets_sent = 0  # how many of them the client has
        self.segments: deque[Segment] = deque()  # arrived and not pushed yet
        self.segment_count = 0  # pushed
        self.end_pushed = False  # the segment with driveningress's end flag was pushed

    @property
    def interruptible(self) -> bool:
        """Whether an interruption ends the turn; an uninterruptible TTS context is kept."""
        return self.started.interruptible

    @property
    def pcm_ms(self) -> float:
        """Exact length of the turn's audio so far, in ms."""
        return (self.pcm_bytes // 2) * 1000 / self.rate

    def on_track(self, frame: Frame | _EndOfAudio) -> bool:
        """Whether ``frame`` is audio on the turn's own track, the one the browser animates."""
        return _plays(frame) and frame.transport_destination == self.started.transport_destination

    def holds_audio_after_its_end(self) -> bool:
        """Whether audio on its track still waits in its hold behind its own last sample."""
        after = self.finished
        for item in self.hold:
            if item is _END_OF_AUDIO:
                after = True
            elif after and self.on_track(item):
                return True
        return False


@dataclasses.dataclass(frozen=True, slots=True)
class _Write:
    """Audio of a turn waiting for the writer task; ``end`` closes the turn's request."""

    turn: _Turn
    pcm: bytes
    end: bool = False


class _Interrupt:
    """The writer's order to interrupt the session's latest request."""

    def __repr__(self) -> str:
        return "<interrupt>"


_INTERRUPT = _Interrupt()


class SpatialRealAvatarProcessor(FrameProcessor):
    """Animates a SpatialReal avatar in the browser with the bot's own speech.

    Put it right after the TTS service, before ``transport.output()``::

        pipeline = Pipeline([
            transport.input(), stt, user_aggregator, llm, tts,
            SpatialRealAvatarProcessor(api_key=..., app_id=..., avatar_id=...),
            transport.output(), assistant_aggregator,
        ])

    The bot's audio keeps playing through the transport; the animation reaches the browser as
    RTVI server messages, which the SpatialReal web SDK lines up with that audio. RTVI must
    therefore be enabled (Pipecat's default). The audio goes downstream at the pipeline's
    ``audio_out_sample_rate``, which must be the transport's output rate too, in chunks of
    ``output_chunk_ms``, which must be the transport's chunk (its ``audio_out_10ms_chunks``
    times 10 ms).

    The processor opens one driveningress session for the whole pipeline, when the pipeline
    starts. While driveningress cannot be reached it lets the audio through unanimated, tells
    the browser so (``STATUS_DEGRADED``), and keeps reconnecting.

    One limit of that session is the plan's. On a plan with a session ceiling (every plan but an
    unlimited one), the server closes the connection when the session reaches it, with close
    code 4002 (``session_timeout``). The processor reconnects at once; the turn being spoken at
    the cut finishes without animation, and the browser sees ``STATUS_DEGRADED`` and then
    ``STATUS_OK`` again. A bot session longer than the ceiling sees this once per ceiling.
    """

    def __init__(
        self,
        *,
        api_key: str,
        app_id: str,
        avatar_id: str,
        environment: str = "us-west",
        console_endpoint_url: str = "",
        ingress_endpoint_url: str = "",
        threshold: int = 328,
        min_silence_ms: int = 150,
        lead_ms: int = 200,
        hold_cap_s: float = 1.5,
        silence_chunk_ms: int = 40,
        output_chunk_ms: int = 40,
        wire_timeout_s: float = 2.0,
        reconnect_backoff_s: tuple[float, ...] = (1, 2, 4, 8),
        **kwargs,
    ):
        """Check the settings; nothing connects before the pipeline sets the processor up.

        Args:
            api_key: SpatialReal API key.
            app_id: SpatialReal app id.
            avatar_id: The avatar to animate.
            environment: The SpatialReal deployment to use when the two URLs are not given.
            console_endpoint_url: Origin of the console API that issues session tokens
                (``https://<host>``; the route is appended to it); empty takes it from
                ``environment``.
            ingress_endpoint_url: The driveningress URL, ``https://<host>/v2/driveningress``;
                empty takes it from ``environment``. An origin alone (``https://<host>``) is
                accepted and given that path.
            threshold: Largest absolute sample value that still counts as silence when finding
                the audible onsets the browser anchors the animation to. Must equal the web
                SDK's value, which is fixed at 328; the web SDK takes no setting for it, so do
                not change it on its own.
            min_silence_ms: Shortest silence, in ms, after which the next sound is a new onset.
                Must equal the web SDK's value, which is fixed at 150; the web SDK takes no
                setting for it, so do not change it on its own. A turn that does not follow the previous
                one back to back starts at least this long after the last audio before it, so
                that the web SDK can find its onset; one that would have started sooner waits
                until this long plus one ``silence_chunk_ms`` have passed.
            lead_ms: How far ahead of real time the audio is released.
            hold_cap_s: Longest time a turn's audio is held for its first animation, counted
                from the turn's first audio; after it, the turn plays without animation. Also the longest wait, once a turn's audio has
                played, for the last segment of its animation.
            silence_chunk_ms: Length of each chunk of silence played while the TTS is behind. A
                multiple of ``output_chunk_ms``.
            output_chunk_ms: Length of the audio chunks released downstream, which every turn
                is padded to a whole number of. It must equal the output transport's chunk,
                ``audio_out_10ms_chunks`` times 10 ms (40 ms by default in Pipecat); the
                transport does not expose it, so it is set here. A positive multiple of 10.
            wire_timeout_s: Longest wait, in seconds, for one write to driveningress (audio, the
                end of a turn's request, an interrupt) and for closing the session. A write that
                takes longer means the link is lost: the audio plays on unanimated, the browser
                is told, and the processor reconnects.
            reconnect_backoff_s: Seconds to wait before each new attempt to reach
                driveningress, in turn; the last value repeats. A session the server ended at
                the plan's ceiling is reconnected without the first wait.
            **kwargs: Passed to ``FrameProcessor``.

        Raises:
            ValueError: If a credential is empty or a setting is out of range.
        """
        for name, value in (("api_key", api_key), ("app_id", app_id), ("avatar_id", avatar_id)):
            if not value:
                raise ValueError(f"{name} must not be empty")
        # The tracker owns the onset rule and its limits: a throwaway one rejects bad values
        # now rather than at the first turn.
        OnsetTracker(16000, threshold=threshold, min_silence_ms=min_silence_ms)
        if lead_ms < 0:
            raise ValueError(f"lead_ms must not be negative, got {lead_ms}")
        if hold_cap_s <= 0:
            raise ValueError(f"hold_cap_s must be positive, got {hold_cap_s}")
        if output_chunk_ms < 10 or output_chunk_ms % 10:
            raise ValueError(
                f"output_chunk_ms must be a positive multiple of 10, got {output_chunk_ms}"
            )
        if silence_chunk_ms < 1:
            raise ValueError(f"silence_chunk_ms must be at least 1, got {silence_chunk_ms}")
        if silence_chunk_ms % output_chunk_ms:
            raise ValueError(
                f"silence_chunk_ms must be a multiple of output_chunk_ms ({output_chunk_ms}), "
                f"got {silence_chunk_ms}"
            )
        if wire_timeout_s <= 0:
            raise ValueError(f"wire_timeout_s must be positive, got {wire_timeout_s}")
        if not reconnect_backoff_s or any(delay < 0 for delay in reconnect_backoff_s):
            raise ValueError(
                f"reconnect_backoff_s must be one or more delays >= 0, got {reconnect_backoff_s}"
            )
        super().__init__(**kwargs)
        self._credentials = {
            "api_key": api_key,
            "app_id": app_id,
            "avatar_id": avatar_id,
            "environment": environment,
            "console_endpoint_url": console_endpoint_url,
            "ingress_endpoint_url": ingress_endpoint_url,
        }
        self._threshold = threshold
        self._min_silence_ms = min_silence_ms
        self._lead_s = lead_ms / 1000
        self._hold_cap_s = hold_cap_s
        self._silence_chunk_ms = silence_chunk_ms
        self._output_chunk_ms = output_chunk_ms
        self._wire_timeout_s = wire_timeout_s
        self._backoff = tuple(reconnect_backoff_s)

        self._out_rate = 24000
        self._chunk_bytes = 0
        self._silence = b""
        self._stall_ahead_s = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._driven: DrivenIngress | None = None
        self._link_task: asyncio.Task | None = None
        self._driver: asyncio.Task | None = None
        self._finished = False  # the EndFrame or CancelFrame went out: nothing more is released
        self._broken = False  # the release failed: every frame passes straight through

        # The link. Everything written to driveningress is queued for one writer task, so
        # audio, silence and interrupts reach it in the order they were decided, and nothing
        # that handles frames or plays audio ever waits for the socket.
        self._writes: deque[_Write | _Interrupt] = deque()
        self._writes_ready = asyncio.Event()  # something is queued for the writer
        self._writer: asyncio.Task | None = None
        self._writing: _Write | _Interrupt | None = None  # what the writer is on, if anything
        self._link_settled = asyncio.Event()  # the first connect attempt is over
        self._link_lost = asyncio.Event()  # set by on_down for the reconnect loop
        self._scheduled_cut = False  # the loss was the server ending the session on schedule
        self._degraded = False  # STATUS_DEGRADED in force: no animation source
        self._status = _STATUS_OK
        self._status_reason = ""
        self._status_sent = _STATUS_OK  # nothing is sent until the status changes
        self._last_sent: _Turn | None = None  # the turn of the session's latest request
        self._first_send: _Turn | None = None  # a turn whose first write has not returned yet
        self._early: list[Segment] = []  # segments that came back before that write returned

        # The turns, oldest first, until each is closed. They release strictly in this order.
        self._turns: deque[_Turn] = deque()
        self._open: _Turn | None = None  # the turn still taking frames
        self._track: str | None = None  # the transport destination the turns' audio plays on
        self._chain: _Turn | None = None  # the latest announced turn, which the next may follow
        self._end_frame: EndFrame | None = None  # waits until every turn is closed
        self._playout_end = 0.0  # loop time by which everything released has played
        self._wake = asyncio.Event()  # something changed for the release task

    # --- Pipecat

    async def setup(self, setup: FrameProcessorSetup):
        """Start connecting to driveningress in the background.

        Args:
            setup: The pipeline's setup; ``audio_out_sample_rate`` becomes the rate of the audio
                sent to driveningress and released downstream.
        """
        await super().setup(setup)
        self._loop = asyncio.get_running_loop()
        self._out_rate = setup.audio_out_sample_rate
        # Sized as Pipecat's output transport sizes its chunk: whole 10 ms blocks at the rate.
        self._chunk_bytes = int(self._out_rate / 100) * 2 * (self._output_chunk_ms // 10)
        self._silence = bytes(self._chunk_bytes * (self._silence_chunk_ms // self._output_chunk_ms))
        # A stall is bridged one chunk ahead, not the lead ahead: the audio that ends it must not
        # queue behind more silence than keeps the transport fed. Never more than the lead.
        self._stall_ahead_s = min(self._lead_s, len(self._silence) / (2 * self._out_rate))
        if not _rtvi_enabled(setup.pipeline_worker):
            logger.error(
                f"{self}: RTVI is disabled in this pipeline, so no animation can reach the "
                "browser; the bot's audio still plays"
            )
        self._driven = DrivenIngress(
            **self._credentials,
            sample_rate=self._out_rate,
            wire_timeout_s=self._wire_timeout_s,
            on_segment=self._on_segment,
            on_down=self._on_down,
        )
        self._link_task = self.create_task(self._keep_linked(), "driveningress_link")
        self._writer = self.create_task(self._keep_writing(), "driveningress_writer")

    async def cleanup(self):
        """Stop the background tasks and close the driveningress session."""
        await super().cleanup()
        self._finished = True
        await self._stop_driver()
        await self._stop_writer()
        await self._stop_link()
        if self._driven is not None:
            await self._driven.close()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Hold back a turn's frames until its animation is in hand; pass everything else on.

        Args:
            frame: The frame to process.
            direction: The direction the frame travels in.
        """
        await super().process_frame(frame, direction)

        if self._broken:
            if isinstance(frame, (EndFrame, CancelFrame)):
                await self._finish()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):  # either way: Pipecat treats both as one
            await self._interrupt(frame, direction)
        elif direction == FrameDirection.UPSTREAM:
            await self.push_frame(frame, direction)
        elif isinstance(frame, SystemFrame):
            if isinstance(frame, CancelFrame):
                await self._stop_driver()
                await self._finish()
            await self.push_frame(frame, direction)
            if isinstance(frame, StartFrame):
                self._check_output_chunk()
                self._start_driver()
        elif isinstance(frame, TTSStartedFrame):
            await self._open_turn(frame)
        elif isinstance(frame, OutputAudioRawFrame) and self._open is not None:
            await self._add_audio(self._open, frame)  # TTS audio or not: it plays in the turn
        elif isinstance(frame, TTSStoppedFrame) and self._open is not None:
            await self._stop_turn(self._open, frame)
        elif isinstance(frame, EndFrame):
            await self._end(frame)
        else:
            await self._hold_or_push(frame)

    def _check_output_chunk(self) -> None:
        """Log an error when the transport right after this processor uses another chunk size."""
        size = getattr(self.next, "audio_chunk_size", None)
        if isinstance(size, int) and size > 0 and size != self._chunk_bytes:
            logger.error(
                f"{self}: output_chunk_ms ({self._output_chunk_ms}) gives {self._chunk_bytes} B "
                f"chunks, but the transport writes {size} B ones; set it to the transport's "
                "audio_out_10ms_chunks x 10, or turns that follow back to back will not line up"
            )

    # --- frames arriving, in order (the process task)

    async def _open_turn(self, frame: TTSStartedFrame) -> None:
        if self._open is not None:
            await self._stop_turn(self._open, None)  # a context that never said it stopped
            if self._broken:  # the release failed meanwhile
                await self.push_frame(frame)
                return
        turn = _Turn(
            frame,
            rate=self._out_rate,
            chunk_bytes=self._chunk_bytes,
            tracker=OnsetTracker(
                self._out_rate, threshold=self._threshold, min_silence_ms=self._min_silence_ms
            ),
            sends=not self._degraded,
        )
        self._turns.append(turn)
        self._open = turn
        self._track = frame.transport_destination
        self._wake.set()

    async def _add_audio(self, turn: _Turn, frame: OutputAudioRawFrame) -> None:
        """Hold audio that came while the turn is open.

        Audio on the turn's track belongs to the turn whatever produced it, since inference has
        to see every sample the room plays: it is made mono at the output rate, cut into whole
        output chunks, counted, sent, and released in its own class (Pipecat counts only TTS
        audio as the bot speaking). Audio for another track only keeps its place.
        """
        if not turn.on_track(frame):
            await self._keep(turn, frame)
            return
        if turn.stream_rate is not None and frame.sample_rate != turn.stream_rate:
            await self._flush_stream(turn)  # its tail plays before what comes next
        if frame.sample_rate == self._out_rate and frame.num_channels == 1:
            pcm, released = frame.audio, frame
        else:
            pcm = _downmix(frame.audio, frame.num_channels)
            if frame.sample_rate != self._out_rate:
                resampler = turn.resamplers.get(frame.sample_rate)
                if resampler is None:
                    # Never cleared on a pause: the TTS may stall mid-turn.
                    resampler = create_stream_resampler(clear_after_secs=None)
                    turn.resamplers[frame.sample_rate] = resampler
                pcm = await resampler.resample(pcm, frame.sample_rate, self._out_rate)
                turn.stream_rate = frame.sample_rate
                turn.last_audio[frame.sample_rate] = frame
            released = _with_audio(frame, pcm, self._out_rate)
        if pcm:  # a resampler may give nothing back while it fills its filter
            await self._add_chunks(turn, turn.chunker.feed(pcm, released))

    async def _add_chunks(
        self, turn: _Turn, chunks: list[tuple[bytes, OutputAudioRawFrame, int]]
    ) -> None:
        """Hold, count and send whole chunks of the turn's audio, in order."""
        for chunk, source, offset in chunks:
            await self._keep(turn, _with_audio(source, chunk, self._out_rate, offset))
            self._take_audio(turn, chunk)

    async def _flush_stream(self, turn: _Turn) -> None:
        """Release the tail the turn's current resampler stream still holds, in its place."""
        rate, turn.stream_rate = turn.stream_rate, None
        if rate is None:
            return
        tail = await turn.resamplers[rate].flush()
        if tail:
            source = _with_audio(turn.last_audio[rate], tail, self._out_rate)
            await self._add_chunks(turn, turn.chunker.feed(tail, source))

    async def _stop_turn(self, turn: _Turn, frame: TTSStoppedFrame | None) -> None:
        self._open = None
        await self._flush_stream(turn)
        padded = turn.chunker.flush()  # the turn ends on a chunk boundary, in both streams
        if padded is not None:
            await self._add_chunks(turn, [padded])
        if frame is not None:
            await self._keep(turn, frame)
        await self._keep(turn, _END_OF_AUDIO)
        turn.stopped = True
        if turn.sent:
            self._queue_write(turn, b"", end=True)  # the end flush returns the last animation

    async def _end(self, frame: EndFrame) -> None:
        if self._open is not None:
            await self._stop_turn(self._open, None)
        if self._turns:
            self._end_frame = frame  # it goes once every turn has played out and closed
            self._wake.set()
        else:
            await self._finish()
            await self.push_frame(frame)

    async def _hold_or_push(self, frame: Frame) -> None:
        last = self._turns[-1] if self._turns else None
        if last is not None and not last.drained:
            await self._keep(last, frame)  # nothing overtakes a turn's frames
            return
        if _plays(frame) and frame.transport_destination == self._track:
            self._playout_end = max(self._playout_end, self._now()) + _duration(frame)
        await self.push_frame(frame)

    async def _keep(self, turn: _Turn, item: Frame | _EndOfAudio) -> None:
        """Add ``item`` to the turn's hold; once the release has failed, push a frame on instead.

        The process task can be in the middle of a turn when the release fails, and must not
        then leave what it was handling in a hold nothing releases any more.
        """
        if not self._broken:
            turn.hold.append(item)
            self._wake.set()
        elif isinstance(item, Frame):
            await self.push_frame(item)

    def _take_audio(self, turn: _Turn, pcm: bytes) -> None:
        """Count a chunk that just joined the turn's hold, and queue it for driveningress."""
        turn.onsets.extend(turn.tracker.feed(pcm))
        turn.pcm_bytes += len(pcm)
        if turn.first_audio_at is None:
            turn.first_audio_at = self._now()
        self._wake.set()
        self._queue_write(turn, pcm)

    # --- driveningress: the writer task

    def _queue_write(self, turn: _Turn, pcm: bytes, *, end: bool = False) -> None:
        """Queue audio of ``turn`` for its request, unless the turn has no animation source.

        The writer sends what is queued in this order. A backlog means it is stuck on a write
        that is about to time out; what cannot be queued is lost for inference, so the turn goes
        on without animation.
        """
        if not turn.sends:
            return
        if len(self._writes) >= _WRITE_BACKLOG:
            self._lose_turn(turn)
            return
        if not turn.sent:
            turn.sent = True
            self._last_sent = turn
        self._writes.append(_Write(turn, pcm, end))
        self._writes_ready.set()

    async def _keep_writing(self) -> None:
        """Perform the queued writes, one at a time and in order, each bounded by the link."""
        while True:
            while self._writes:
                op = self._writes.popleft()
                self._writing = op
                try:
                    if isinstance(op, _Interrupt):
                        assert self._driven is not None
                        await self._driven.interrupt()
                    else:
                        await self._write(op)
                except Exception:  # noqa: BLE001 - the writer must outlive any one write
                    logger.exception(f"{self}: a write to driveningress failed unexpectedly")
                    if isinstance(op, _Write):
                        self._lose_turn(op.turn)
                finally:
                    self._writing = None
            self._writes_ready.clear()
            await self._writes_ready.wait()

    async def _write(self, op: _Write) -> None:
        """Send one queued write to the turn's request; a failure or a timeout is a loss."""
        turn = op.turn
        if not turn.sends:
            return
        assert self._driven is not None
        first = not turn.req_id
        if first and not self._driven.connected and not self._degraded:
            # The first connect is still under way: wait for it, as long as the hold cap.
            try:
                async with asyncio.timeout(self._hold_cap_s):
                    await self._link_settled.wait()
            except TimeoutError:
                pass
            if not turn.sends:  # given up meanwhile
                return
        if not self._driven.connected:
            self._lose_turn(turn)
            return
        if first:
            self._first_send = turn
        early: list[Segment] = []
        try:
            req_id = await self._driven.send(op.pcm, end=op.end)
        except RuntimeError:
            # A failed or timed-out send is a loss and on_down has run, or the session was not
            # up: either way the turn has no request any more.
            self._lose_turn(turn)
            return
        finally:
            if first:
                self._first_send = None
                early, self._early = self._early, []
                self._wake.set()  # the turn may be announced now, whatever the outcome
        if first:
            turn.req_id = req_id
            if not turn.turn_id:
                turn.turn_id = req_id  # announced already: it keeps the id the browser knows
            turn.segments.extend(segment for segment in early if segment.req_id == req_id)
            self._wake.set()

    def _on_segment(self, segment: Segment) -> None:
        turn = next((t for t in self._turns if t.req_id == segment.req_id), None)
        if turn is None:
            if self._first_send is not None:  # its request id is not known yet
                self._early.append(segment)
            return  # otherwise interrupted or closed: dropped
        turn.segments.append(segment)
        self._wake.set()

    def _on_down(self, reason: str) -> None:
        self._lose_link(reason)
        # The server ends a session at the plan's ceiling: a cut on schedule, not a fault.
        self._scheduled_cut = close_code(reason) == spatialreal.CloseCode.SESSION_TIMEOUT
        self._link_lost.set()

    def _lose_link(self, reason: str) -> None:
        """The animation source is gone: every turn counting on it goes on without it."""
        for turn in self._turns:
            self._lose_turn(turn)
        self._last_sent = None
        if not self._degraded:
            logger.warning(f"{self}: driveningress unavailable ({reason}); audio plays unanimated")
            self._degraded = True
            self._status, self._status_reason = _STATUS_DEGRADED, reason
        self._wake.set()

    def _lose_turn(self, turn: _Turn) -> None:
        if turn.sends:
            turn.sends = False
            turn.degraded = True
        self._wake.set()

    async def _keep_linked(self) -> None:
        """Connect, and connect again after every loss, waiting as the backoff says."""
        assert self._driven is not None
        failures = 0
        while True:
            if failures:
                await asyncio.sleep(self._backoff[min(failures, len(self._backoff)) - 1])
            try:
                async with asyncio.timeout(CONNECT_TIMEOUT_S):
                    await self._driven.connect()
            except Exception as exc:  # noqa: BLE001 - any failure to connect is retried
                failures += 1
                reason = "connect timed out" if isinstance(exc, TimeoutError) else _describe(exc)
                logger.debug(f"{self}: driveningress connect failed: {reason}")
                self._lose_link(reason)
                self._link_settled.set()
                continue
            failures = 0
            self._link_settled.set()
            if self._degraded:
                logger.info(f"{self}: driveningress connected again")
                self._degraded = False
                self._status, self._status_reason = _STATUS_OK, ""
                self._wake.set()
            await self._link_lost.wait()
            self._link_lost.clear()
            # A session the server cut on schedule is reconnected at once; a fault waits.
            failures = 0 if self._scheduled_cut else 1

    # --- interruption (the input task)

    async def _interrupt(self, frame: InterruptionFrame, direction: FrameDirection) -> None:
        """Abort the interruptible turns, then pass the interruption on.

        An uninterruptible TTS context (a backchannel, say) plays on, as it does in Pipecat, and
        so does every uninterruptible frame held in an aborted turn.
        """
        victims = [turn for turn in self._turns if turn.interruptible]
        if not victims:
            await self.push_frame(frame, direction)
            return
        await self._stop_driver()  # nothing is released while the holds change

        order = list(self._turns)
        unplaced: list[Frame] = []  # rescued frames no surviving turn ahead of them holds back
        for turn in victims:
            self._turns.remove(turn)
            rescued = [
                item for item in turn.hold if isinstance(item, Frame) and not item.interruptible
            ]
            keeper = next(
                (
                    t
                    for t in reversed(order[: order.index(turn)])
                    if t not in victims and not t.drained
                ),
                None,
            )
            (keeper.hold if keeper is not None else unplaced).extend(rescued)
            turn.hold.clear()
            turn.segments.clear()
        if self._open in victims:
            self._open = None
        if self._chain in victims:
            self._chain = None
        if any(turn.announced for turn in victims):  # the transport drops what it still holds
            self._playout_end = min(self._playout_end, self._now())
        interrupt = self._driven is not None and self._last_sent in victims
        if interrupt:
            self._last_sent = None

        # The interrupt is queued behind nothing of the victims' and before the pushes below,
        # so that nothing the pipeline sends after the barge-in (Pipecat has already restarted
        # the process task) can reach the socket ahead of it. The pushes never wait for the
        # socket: a congested link cannot hold the barge-in back.
        self._writes = deque(
            op for op in self._writes if isinstance(op, _Interrupt) or op.turn not in victims
        )
        if interrupt:
            self._writes.append(_INTERRUPT)
            self._writes_ready.set()
        for turn in victims:
            if turn.announced and not turn.silent:
                await self._push_message(envelope.turn_abort(turn.turn_id))
        await self.push_frame(frame, direction)
        for item in unplaced:
            if _plays(item):
                self._playout_end = max(self._playout_end, self._now()) + _duration(item)
            await self.push_frame(item)
        self._start_driver()

    # --- the release task

    def _start_driver(self) -> None:
        if self._driver is None and not (self._finished or self._broken):
            self._driver = self.create_task(self._drive(), "avatar_release")

    async def _stop_driver(self) -> None:
        driver, self._driver = self._driver, None
        if driver is not None:
            await self.cancel_task(driver)

    async def _stop_writer(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            await self.cancel_task(writer)

    async def _stop_link(self) -> None:
        task, self._link_task = self._link_task, None
        if task is not None:
            await self.cancel_task(task)

    async def _finish(self) -> None:
        """The EndFrame or CancelFrame goes out: no state follows it, and no reconnect."""
        self._finished = True
        await self._stop_link()

    async def _drive(self) -> None:
        """Push whatever is due, then sleep until the next deadline or the next change.

        Should this ever fail, the bot's audio would be held for good; the processor gives up on
        the animation instead and lets the audio through.
        """
        try:
            while True:
                self._wake.clear()
                deadlines: list[float] = []
                await self._push_status()
                await self._announce_and_forward(deadlines)
                await self._release(deadlines)
                await self._close(deadlines)
                if self._wake.is_set():
                    continue
                try:
                    async with asyncio.timeout_at(min(deadlines, default=None)):
                        await self._wake.wait()
                except TimeoutError:
                    pass
        except Exception:  # noqa: BLE001 - fail open, whatever the fault
            logger.exception(
                f"{self}: the avatar release failed; audio plays unanimated from now on"
            )
            await self._fail_open()

    async def _fail_open(self) -> None:
        """Push everything held at once, end the turns, and let every later frame through."""
        self._broken = True
        turns, self._turns = list(self._turns), deque()
        self._open = self._chain = None
        for turn in turns:
            turn.sends = False
        if not self._finished and self._status_sent != _STATUS_DEGRADED:
            self._status_sent = _STATUS_DEGRADED
            await self._push_message(envelope.state(_STATUS_DEGRADED, "processor error"))
        for turn in turns:
            # No playout end can be timed any more: the client is told to stop animating now,
            # which every announced turn needs, exactly once, to end.
            if turn.announced and not turn.silent:
                await self._push_message(envelope.turn_abort(turn.turn_id))
            for item in turn.hold:
                if isinstance(item, Frame):
                    await self.push_frame(item)
            turn.hold.clear()
        if self._end_frame is not None:
            frame, self._end_frame = self._end_frame, None
            await self._finish()
            await self.push_frame(frame)
        await self._stop_writer()
        self._writes.clear()
        await self._stop_link()
        if self._driven is not None:
            await self._driven.close()

    async def _push_status(self) -> None:
        if self._status != self._status_sent and not self._finished:
            self._status_sent = self._status
            await self._push_message(envelope.state(self._status, self._status_reason))

    async def _announce_and_forward(self, deadlines: list[float]) -> None:
        """Announce the turns that are due, in order, and push the segments that arrived.

        A turn that follows the previous one back to back is announced as soon as its animation
        is in hand, well ahead of its audio. Any other is announced only when its release starts
        (see ``_release``): the browser takes the first onset after an unchained turn_start as
        the turn's own, so nothing the room plays before the turn's audio may come after it.
        """
        for turn in list(self._turns):
            if not turn.ready:
                due = self._announcement_due(turn)
                if due is not True:
                    if due is not None:
                        deadlines.append(due)
                    return  # a later turn is never announced before this one
                turn.ready = True
                if turn.silent:
                    turn.announced = True  # nothing to announce: it is only released in order
                else:
                    prev_turn_id, prev_turn_pcm_ms = self._follows()
                    if prev_turn_id:
                        turn.chained = True
                        await self._announce(turn, prev_turn_id, prev_turn_pcm_ms)
            if not turn.announced:
                return  # unchained: announced when its release starts; later turns wait
            await self._forward_segments(turn)

    def _announcement_due(self, turn: _Turn) -> bool | float | None:
        """True when ``turn`` may be announced now, else the loop time it will be, if known.

        The hold cap counts from the turn's first audio whatever the link is doing: a turn
        whose first write is still on the wire is announced without animation when the cap runs
        out, like one whose animation is late, and gets whatever comes back later.
        """
        if turn.stopped and not turn.pcm_bytes:
            turn.silent = True
            return True
        if turn.segments or (turn.degraded and turn.pcm_bytes):
            return True
        if turn.first_audio_at is None:
            return None
        expiry = turn.first_audio_at + self._hold_cap_s
        if expiry > self._now() + _TOLERANCE_S:
            return expiry
        logger.warning(f"{self}: no animation within {self._hold_cap_s} s; the turn plays without")
        turn.degraded = True
        if not turn.sent:
            turn.sends = False  # nothing was queued for driveningress: no request to close there
        return True

    async def _announce(
        self, turn: _Turn, prev_turn_id: str = "", prev_turn_pcm_ms: float = 0.0
    ) -> None:
        turn.announced = True
        if not turn.turn_id:
            turn.turn_id = spatialreal.generate_log_id()  # it has no request id (yet)
        turn.onsets_sent = len(turn.onsets)
        self._chain = turn
        await self._push_message(
            envelope.turn_start(
                turn.turn_id,
                sample_rate=self._out_rate,
                onsets_ms=list(turn.onsets),
                prev_turn_id=prev_turn_id,
                prev_turn_pcm_ms=prev_turn_pcm_ms,
                degraded=turn.degraded,
            )
        )

    def _follows(self) -> tuple[str, float]:
        """The turn a turn announced now follows back to back, and that turn's exact PCM length.

        A turn is chained to the previous one only when its first sample will play right after
        the previous turn's last: either the previous turn is still releasing, so this one
        starts the moment it drains, or the room is still playing the previous turn's audio.
        The browser then has no silence to find the onset in, and places the turn from the
        previous turn's length instead, which is exact only with no gap at all.

        Any gap, or other audio between the two, would make that placement wrong by its
        length, and it cannot be corrected: the gap is only known once the turn is released,
        and by then its audio has long gone to inference at TTS speed, so no silence can be put
        in front of it there. A turn that does not follow back to back is therefore unchained:
        its release waits until the room has been quiet for ``min_silence_ms`` and one chunk
        (see ``_release_start``), which lets the browser find its onset itself, and it is
        announced only as that release starts, so that no sound the room plays before it can be
        taken for that onset.

        A previous turn that plays without animation gives the browser no frame to place this
        one from, so the turn is unchained then too.

        Returns:
            ``(prev_turn_id, prev_turn_pcm_ms)``, or ``("", 0.0)`` when the turn is not chained.
        """
        prev = self._chain
        if prev is None or prev.degraded or prev.holds_audio_after_its_end():
            return "", 0.0
        if prev.finished and self._playout_end > prev.playout_end + _TOLERANCE_S:
            return "", 0.0  # audio that is not the previous turn's played after it
        if prev.drained and self._now() > prev.playout_end + _TOLERANCE_S:
            return "", 0.0  # the room has gone quiet: this turn would start after a gap
        return prev.turn_id, prev.pcm_ms

    def _release_start(self, turn: _Turn) -> float:
        """The loop time ``turn`` may start releasing at; see ``_follows``.

        A turn that is chained, or that plays without animation, starts as soon as it can. Any
        other starts once the last audio released before it has been followed by
        ``min_silence_ms`` and one chunk of quiet. The room's transport fills that gap; nothing
        is sent to inference for it.
        """
        if turn.chained or turn.degraded or turn.silent:
            return 0.0
        quiet_since = self._playout_end
        if self._now() - quiet_since >= self._min_silence_ms / 1000:
            return 0.0
        return quiet_since + (self._min_silence_ms + self._silence_chunk_ms) / 1000

    async def _forward_segments(self, turn: _Turn) -> None:
        while turn.segments:
            segment = turn.segments.popleft()
            seq = turn.segment_count
            turn.segment_count += 1
            turn.end_pushed = turn.end_pushed or segment.end
            for data in envelope.frames(turn.turn_id, seq, segment.wire, end=segment.end):
                await self._push_message(data)
            new = turn.onsets[turn.onsets_sent :]
            if new:
                turn.onsets_sent = len(turn.onsets)
                await self._push_message(envelope.turn_onsets(turn.turn_id, new))

    async def _release(self, deadlines: list[float]) -> None:
        """Release the held frames that are due, turn after turn, at real time plus the lead."""
        for turn in list(self._turns):
            if turn.drained:
                continue
            if not turn.ready:
                return  # it holds everything behind it until its animation is in hand
            if turn.release_at is None:  # its turn to release has come: decide when it starts
                turn.release_at = self._release_start(turn)
            if turn.release_at > self._now() + _TOLERANCE_S:
                deadlines.append(turn.release_at)  # no silence for the gap: the room plays it
                return
            if not turn.announced:  # unchained: announced now, with the animation in hand
                await self._announce(turn)
                await self._forward_segments(turn)
                self._wake.set()  # the turn after it may be announced now
            while turn.hold or not turn.finished:
                if not turn.hold:  # the TTS is behind playback: play silence meanwhile
                    due = self._next_release(self._stall_ahead_s)
                    if due is not None:
                        deadlines.append(due)
                        return
                    await self._play_silence(turn)
                    continue
                item = turn.hold[0]
                if item is _END_OF_AUDIO:
                    turn.hold.popleft()
                    turn.finished = True
                    turn.playout_end = max(self._playout_end, self._now())
                    continue
                if turn.on_track(item):
                    due = self._next_release(self._lead_s)
                    if due is not None:
                        deadlines.append(due)
                        return
                    self._playout_end = max(self._playout_end, self._now()) + _duration(item)
                turn.hold.popleft()
                await self.push_frame(item)
            turn.drained = True  # the next turn may start at once

    def _next_release(self, ahead: float) -> float | None:
        """None when audio may go now, ``ahead`` s of playout at most, else the loop time it may."""
        due = self._playout_end - ahead
        return None if due <= self._now() + _TOLERANCE_S else due

    async def _play_silence(self, turn: _Turn) -> None:
        """Play one chunk of silence for the turn, and queue the same samples for inference."""
        pcm = self._silence
        turn.onsets.extend(turn.tracker.feed(pcm))
        turn.pcm_bytes += len(pcm)
        frame = OutputAudioRawFrame(audio=pcm, sample_rate=self._out_rate, num_channels=1)
        frame.transport_destination = turn.started.transport_destination
        frame.interruptible = turn.interruptible
        self._playout_end = max(self._playout_end, self._now()) + _duration(frame)
        self._queue_write(turn, pcm)  # the same samples inference sees; never waited for here
        await self.push_frame(frame)  # not TTS audio: Pipecat does not count it as speech

    async def _close(self, deadlines: list[float]) -> None:
        """End the turns whose audio has played out and whose animation is complete."""
        for turn in list(self._turns):
            if not turn.drained:
                break  # turns drain in order: none after this one has
            if not turn.silent:
                now = self._now() + _TOLERANCE_S
                if now < turn.playout_end:
                    deadlines.append(turn.playout_end)
                    continue
                if not (turn.end_pushed or turn.degraded):
                    give_up = turn.playout_end + self._hold_cap_s
                    if now < give_up:
                        deadlines.append(give_up)
                        continue
                    logger.debug(f"{self}: turn {turn.turn_id} ends without its last segment")
            self._turns.remove(turn)
            if not turn.silent:
                await self._push_message(
                    envelope.turn_end(
                        turn.turn_id, pcm_ms=turn.pcm_ms, segment_count=turn.segment_count
                    )
                )
        if not self._turns and self._end_frame is not None:
            frame, self._end_frame = self._end_frame, None
            await self._finish()
            await self.push_frame(frame)

    # --- helpers

    async def _push_message(self, data: dict) -> None:
        await self.push_frame(RTVIServerMessageFrame(data=data))

    def _now(self) -> float:
        assert self._loop is not None
        return self._loop.time()


def _rtvi_enabled(worker: object) -> bool:
    """Whether the pipeline has RTVI; ``PipelineWorker.rtvi`` raises when it has not."""
    try:
        return getattr(worker, "rtvi", None) is not None
    except Exception:  # noqa: BLE001
        return False


def _plays(frame: Frame) -> bool:
    """Whether ``frame`` is audio that takes time to play."""
    return isinstance(frame, OutputAudioRawFrame) and bool(frame.audio)


def _duration(frame: OutputAudioRawFrame) -> float:
    """Seconds of audio in ``frame``."""
    if frame.sample_rate <= 0 or frame.num_channels <= 0:
        return 0.0
    return len(frame.audio) / (2 * frame.num_channels * frame.sample_rate)


def _downmix(pcm: bytes, channels: int) -> bytes:
    """Average interleaved PCM16 little-endian channels into one; mono comes back unchanged.

    A trailing partial frame (fewer bytes than one sample of every channel) is dropped.
    """
    if channels <= 1:
        return pcm
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % (2 * channels)])
    if sys.byteorder == "big":
        samples.byteswap()  # array reads the machine's byte order, the audio is little-endian
    columns = [samples[channel::channels] for channel in range(channels)]
    mono = array("h", (sum(frame) // channels for frame in zip(*columns)))
    if sys.byteorder == "big":
        mono.byteswap()
    return mono.tobytes()


def _with_audio(
    frame: OutputAudioRawFrame, pcm: bytes, rate: int, offset: int = 0
) -> OutputAudioRawFrame:
    """A frame of the same class and routing as ``frame`` that carries ``pcm``, mono at ``rate``.

    ``offset`` is where ``pcm`` starts in the frame's own audio, in bytes of mono at ``rate``;
    the copy's presentation time, when the frame has one, is moved by that much.
    """
    copy = dataclasses.replace(frame, audio=pcm, sample_rate=rate, num_channels=1)
    copy.pts = None if frame.pts is None else frame.pts + offset * 1_000_000_000 // (2 * rate)
    copy.metadata = frame.metadata
    copy.transport_source = frame.transport_source
    copy.transport_destination = frame.transport_destination
    copy.interruptible = frame.interruptible
    return copy
