"""A thin asyncio layer over one SpatialReal driveningress session.

The processor streams the bot's speech to driveningress and gets the facial animation back. The
``spatialreal`` SDK does the work (the session token, the WebSocket, the protobuf messages); this
module narrows it to what the processor needs.

* Animation returns over the same WebSocket, which is what the SDK does when no egress is
  configured. Each animation message from the server reaches the caller as a :class:`Segment`
  whose ``wire`` is the SDK's bytes, untouched, so the processor can forward them without
  serializing anything.
* A session that ends is reported once, through ``on_down``, with a short readable reason.
  The SDK has no reconnect and a session object cannot be reused, so this layer has none either:
  to come back, call :meth:`DrivenIngress.connect` again, which builds a new SDK session. When
  to do that, and how often, is up to the caller.

It decides nothing about turns: which audio belongs to which request, and when a request ends,
is the caller's.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from functools import partial

import spatialreal
from loguru import logger
from spatialreal.proto.generated import message_pb2


@dataclass(frozen=True)
class Segment:
    """One animation message from driveningress.

    The server answers a request's audio with one message per inference segment, each carrying
    the keyframes for a stretch of that audio.

    Attributes:
        req_id: The request the animation belongs to: the id :meth:`DrivenIngress.send` returned
            for its audio.
        wire: The complete serialized ``driveningress.v2.Message``, exactly as received.
        end: True on the last message the server sends for the request. That message is an
            ordinary segment, usually with animation, not a marker of its own, and the server
            may send none. It sends none when the end arrives with nothing left to infer (no
            audio was sent, or the audio already filled its last segment exactly, and the
            client sends the 16 kHz the inference runs at, so no resampler tail remains) or
            when the last inference fails. A consumer must not wait for ``end`` alone to close
            a turn.
    """

    req_id: str
    wire: bytes
    end: bool


class _State(Enum):
    CONNECTING = auto()  # connect() is still opening the session
    UP = auto()  # connected: the only state in which anything is delivered or sent
    DOWN = auto()  # lost, closed, or never opened


class _Link:
    """What belongs to one SDK session.

    A session is single-use, so everything that describes it lives here and the next session
    starts from a new link. The callbacks the SDK is given carry the link, and a link that is
    DOWN drops whatever its session still reports, so a late callback from a session that was
    lost, closed or replaced cannot reach the caller.
    """

    session: spatialreal.AvatarSession  # set right after construction; the callbacks need the link

    def __init__(self) -> None:
        self.state = _State.CONNECTING
        self.last_req_id: str | None = None  # the request an interrupt would target
        self.error: Exception | None = None  # the last error the session reported


class DrivenIngress:
    """One driveningress session at a time, for the audio of one avatar.

    Use it from the event loop that owns it; nothing here is thread-safe. The callbacks run on
    that loop and must not block.
    """

    def __init__(
        self,
        *,
        api_key: str,
        app_id: str,
        avatar_id: str,
        sample_rate: int,
        environment: str = "us-west",
        console_endpoint_url: str = "",
        ingress_endpoint_url: str = "",
        token_ttl_s: int = 86400,
        on_segment: Callable[[Segment], None],
        on_down: Callable[[str], None],
    ):
        """Remember the settings; nothing connects until :meth:`connect`.

        Args:
            api_key: SpatialReal API key, exchanged for a session token on every connect.
            app_id: SpatialReal app id.
            avatar_id: The avatar to drive.
            sample_rate: Hz of the PCM16 mono audio that will be sent. It is fixed for a session
                and the server accepts only some rates (8, 16, 22.05, 24, 32, 44.1 and 48 kHz).
            environment: Which SpatialReal deployment to use, when the two URLs below are not
                given.
            console_endpoint_url: Origin of the console API that issues session tokens; empty
                takes it from ``environment``.
            ingress_endpoint_url: Origin of driveningress, without the path; empty takes it from
                ``environment``.
            token_ttl_s: Seconds each session token stays valid.
            on_segment: Called with every animation message, in the order it arrived. It is not
                called once the session is lost or closed.
            on_down: Called once for each session that ends without :meth:`close`, with a short
                reason, for example ``creditsExhausted (4001)`` (the SDK's error code, then the
                WebSocket close code when there is one) or ``closed``. The reason comes from the
                last error the session reported before it ended, or is ``closed`` when it
                reported none.
        """
        self._options = {
            "api_key": api_key,
            "app_id": app_id,
            "avatar_id": avatar_id,
            "sample_rate": sample_rate,
            "environment": environment,
            "console_endpoint_url": console_endpoint_url,
            "ingress_endpoint_url": ingress_endpoint_url,
        }
        self._token_ttl = timedelta(seconds=token_ttl_s)
        self._on_segment = on_segment
        self._on_down = on_down
        self._link: _Link | None = None

    @property
    def connected(self) -> bool:
        """Whether a session is up: connected, and neither lost nor closed since."""
        return self._link is not None and self._link.state is _State.UP

    async def connect(self) -> None:
        """Open a new session, replacing whatever is left of the last one.

        Calling it again after a loss is how the caller reconnects. A session that is still up
        is closed first, without ``on_down``. Do not call it from two tasks at once.

        Raises:
            spatialreal.AvatarSDKError: The token exchange or the handshake failed; its ``code``
                says why, and ``retryable`` whether trying again can help.
            ValueError: ``api_key``, ``app_id`` or ``avatar_id`` is empty.
            RuntimeError: :meth:`close` ran while this call was waiting and the call gave up.
                A close that cuts the handshake short raises ``AvatarSDKError`` instead. Either
                way no socket is left open.
        """
        await self.close()
        link = _Link()
        link.session = spatialreal.new_avatar_session(
            **self._options,
            expire_at=datetime.now(UTC) + self._token_ttl,
            audio_format="pcm_s16le",
            transport_frames=partial(self._on_frame, link),
            on_error=partial(self._on_error, link),
            on_close=partial(self._on_close, link),
        )
        self._link = link  # close() can reach the session from here on
        try:
            await link.session.init()
            # A close() during the token exchange must not be followed by a socket: it bills.
            if link.state is _State.CONNECTING:
                await link.session.start()
        except BaseException:  # also a cancellation: the half-open session must not outlive it
            link.state = _State.DOWN
            if self._link is link:
                self._link = None
            await link.session.close()
            raise
        if link.state is _State.DOWN:  # close() ran while this call waited
            await link.session.close()  # closes the socket too, if start() opened one
            raise RuntimeError("driveningress closed while connecting")
        link.state = _State.UP

    async def send(self, pcm: bytes, *, end: bool = False) -> str:
        """Send audio to the current request.

        The first call after a connect, after ``end=True`` or after an interrupt starts a new
        request; the id is never reused.

        Args:
            pcm: PCM16 little-endian mono audio at ``sample_rate``; may be empty.
            end: True on the request's last chunk. The server's reply to it is not guaranteed to
                be a :class:`Segment` with ``end`` set; see there.

        Returns:
            The id of the request the audio went into.

        Raises:
            RuntimeError: The session is not up, or the send failed. A failed send is a loss: it
                is reported through ``on_down`` like any other, and the original error is the
                exception's ``__cause__``.
        """
        link = self._link
        if link is None or link.state is not _State.UP:
            raise RuntimeError("driveningress not connected")
        try:
            req_id = await link.session.send_audio(pcm, end=end)
        except (spatialreal.AvatarSDKError, ValueError) as exc:
            self._lose(link, _describe(exc))
            raise RuntimeError("driveningress not connected") from exc
        link.last_req_id = req_id
        return req_id

    async def interrupt(self) -> str | None:
        """Tell the server to drop the audio it still holds for the latest request.

        The next :meth:`send` starts a new request. This never raises for a missing or broken
        session: there is then nothing left to interrupt, and the caller is usually in the
        middle of something more urgent.

        Returns:
            The id of the request interrupted, or None when nothing was sent on this session or
            the session is not up.
        """
        link = self._link
        if link is None or link.state is not _State.UP or link.last_req_id is None:
            return None
        try:
            return await link.session.interrupt()
        except (spatialreal.AvatarSDKError, ValueError) as exc:
            self._lose(link, _describe(exc))
            return None

    async def close(self) -> None:
        """Close the session; safe at any time, including when there is none, and repeatedly.

        ``on_down`` is not called, because closing is not a loss.
        """
        link, self._link = self._link, None
        if link is not None:
            link.state = _State.DOWN
            await link.session.close()

    # --- what the SDK reports

    def _on_frame(self, link: _Link, wire: bytes, is_last: bool) -> None:
        if link.state is not _State.UP:
            return
        req_id = message_pb2.Message.FromString(wire).server_response_animation.req_id
        self._on_segment(Segment(req_id=req_id, wire=wire, end=is_last))

    def _on_error(self, link: _Link, exc: Exception) -> None:
        # An error does not necessarily end the session: the server may report a failed request
        # and carry on. What ends it is the close, so remember the error and let _on_close decide.
        if link.state is not _State.UP:
            return
        link.error = exc
        logger.warning("driveningress session error: {}", _describe(exc))

    def _on_close(self, link: _Link) -> None:
        self._lose(link, "closed" if link.error is None else _describe(link.error))

    def _lose(self, link: _Link, reason: str) -> None:
        """Report that ``link`` ended, once; nothing if it is already DOWN."""
        if link.state is not _State.UP:
            return
        link.state = _State.DOWN
        try:
            self._on_down(reason)
        except Exception:  # noqa: BLE001
            # The callback is the caller's. Its bug must not reach the call that found the loss:
            # send() promises RuntimeError and interrupt() promises not to raise.
            logger.exception("on_down callback raised")


def _describe(exc: Exception) -> str:
    """Return a short reason for a log line or ``on_down``, such as ``creditsExhausted (4001)``."""
    if isinstance(exc, spatialreal.AvatarSDKError):
        if exc.close_code is None:
            return exc.code.value
        return f"{exc.code.value} ({exc.close_code})"
    return f"{type(exc).__name__}: {exc}"
