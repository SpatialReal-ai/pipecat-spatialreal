"""An in-process stand-in for the SpatialReal backend, for tests.

``FakeDriveningress`` serves the two things the ``spatialreal`` SDK talks to, on 127.0.0.1 and on
ports the OS picks: the console's session-token route (``console_url``) and the driveningress v2
WebSocket (``ingress_url``). It speaks the real protocol with the SDK's generated messages, so a
test drives the genuine SDK against it. It follows ``FakeBackend`` in the python-sdk repo's
``tests/test_session.py``.

What this one adds is an animation model. The real server answers audio with one
``ServerResponseAnimation`` per inference segment; the fake does the same under a rule that
depends on the audio and on nothing else, so a test sees the same segments on every run:

* Audio is counted per ``req_id`` in bytes of PCM16 mono, never in floating-point ms, so the same
  audio gives the same segments however it is cut into messages.
* The first segment of a request is full after ``first_segment_ms`` of audio and each later one
  after ``segment_ms`` (both rounded down to whole samples at the session's sample rate). A full
  segment is one message with ``end=False`` and ``animation_bytes`` bytes of animation, sent as
  soon as the audio that fills it has arrived. A single message that carries more than one
  segment's worth makes several.
* ``end=True`` closes the request with one message that has ``end=True``: it carries
  ``animation_bytes`` bytes if any audio is left over, and none if the last segment filled
  exactly or no audio was sent at all. The real server sends that message in both cases.
* ``ClientInterrupt`` drops the request's left-over audio without a flush, as the real server
  does.
"""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Self

import websockets
from aiohttp import web
from spatialreal.proto.generated import message_pb2

TOKEN_PATH = "/v1/auth/session-token"
SESSION_TOKEN = "tok"


@dataclass
class _Request:
    """What the server holds for one ``req_id``."""

    pending_bytes: int = 0  # audio received and not yet covered by a segment
    segments: int = 0  # full segments sent so far


class FakeDriveningress:
    """Console token route plus driveningress WebSocket, with the animation model above.

    Use it as an async context manager. The server keeps accepting connections for as long as it
    is entered, so a client that was dropped can connect again.

    Attributes:
        first_segment_ms: Audio that fills a request's first segment.
        segment_ms: Audio that fills every later segment.
        animation_bytes: Size of the animation payload in each message that has one.
        console_url: Origin of the token route, for ``console_endpoint_url``.
        ingress_url: Origin of the WebSocket, for ``ingress_endpoint_url``. It has no
            ``/v2/driveningress`` suffix because the SDK appends only ``/websocket``.
        configure: The ``ClientConfigureSession`` of the latest connection, or None before any.
        ws_headers: Request headers of the latest connection, names lower-cased.
        ws_path: Path and query of the latest connection's request.
        received: Every message the clients sent after their handshake, in arrival order and
            across connections. The ``ClientConfigureSession`` is in ``configure`` instead.
        token_requests: One ``(api_key, json_body)`` per call to the token route.
        token_gate: When set, the token route waits for this event before it answers.
        connections: Connections accepted so far.
        open_connections: Connections currently open.
    """

    def __init__(
        self,
        *,
        first_segment_ms: int = 200,
        segment_ms: int = 200,
        animation_bytes: int = 4800,
    ):
        """Set the animation model; nothing listens until the fake is entered.

        Args:
            first_segment_ms: Audio that fills a request's first segment.
            segment_ms: Audio that fills every later segment.
            animation_bytes: Size of the animation payload in each message that has one.
        """
        self.first_segment_ms = first_segment_ms
        self.segment_ms = segment_ms
        self.animation_bytes = animation_bytes
        self.console_url = ""
        self.ingress_url = ""
        self.configure: message_pb2.ClientConfigureSession | None = None
        self.ws_headers: dict[str, str] = {}
        self.ws_path = ""
        self.received: list[message_pb2.Message] = []
        self.token_requests: list[tuple[str, dict[str, Any]]] = []
        self.token_gate: asyncio.Event | None = None
        self.connections = 0
        self.open_connections = 0
        self._current: Any = None  # the latest connection's server-side socket
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> Self:
        """Start both servers."""
        async with AsyncExitStack() as stack:
            app = web.Application()
            app.router.add_post(TOKEN_PATH, self._token)
            runner = web.AppRunner(app)
            await runner.setup()
            stack.push_async_callback(runner.cleanup)
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            self.console_url = f"http://127.0.0.1:{runner.addresses[0][1]}"

            server = await websockets.serve(self._serve, "127.0.0.1", 0)

            async def stop_server() -> None:
                server.close()
                await server.wait_closed()

            stack.push_async_callback(stop_server)
            self.ingress_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            self._stack = stack.pop_all()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Stop both servers; connections still open are closed with 1001."""
        assert self._stack is not None
        await self._stack.aclose()

    # --- what a test can ask of it

    def received_audio_bytes(self, req_id: str) -> bytes:
        """Return the audio received for a request, concatenated in arrival order.

        Args:
            req_id: The request id the client sent the audio under.

        Returns:
            The ``audio`` of every ``ClientAudioInput`` with this ``req_id``.
        """
        return b"".join(
            m.client_audio_input.audio
            for m in self.received
            if m.type == message_pb2.MESSAGE_CLIENT_AUDIO_INPUT
            and m.client_audio_input.req_id == req_id
        )

    def received_audio_ms(self, req_id: str) -> float:
        """Return the length in ms of the audio received for a request.

        One division of the byte total, so a whole number of ms comes out exact.

        Args:
            req_id: The request id the client sent the audio under.

        Returns:
            The audio's duration at the latest connection's sample rate.
        """
        assert self.configure is not None, "no client has connected"
        return len(self.received_audio_bytes(req_id)) * 1000 / (2 * self.configure.sample_rate)

    async def close_with(self, code: int) -> None:
        """Close the latest connection from the server's side with a WebSocket close code.

        The server keeps accepting new connections.

        Args:
            code: The close code, for example 4001 for "credits exhausted".
        """
        assert self._current is not None, "no client has connected"
        await self._current.close(code=code, reason="closed by the test")

    async def send(self, message: message_pb2.Message) -> None:
        """Send a server message of the test's own making on the latest connection.

        Args:
            message: The message to send, for example a ``ServerError``.
        """
        assert self._current is not None, "no client has connected"
        await self._current.send(message.SerializeToString())

    # --- the console

    async def _token(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.token_requests.append((request.headers.get("X-Api-Key", ""), body))
        if self.token_gate is not None:
            await self.token_gate.wait()
        return web.json_response({"session_token": SESSION_TOKEN})

    # --- driveningress

    async def _serve(self, ws: Any) -> None:
        self.connections += 1
        self.open_connections += 1
        self._current = ws
        connection_id = f"conn-{self.connections}"
        # websockets 13 and later hand the handler a connection with .request; older ones keep
        # the request on the connection itself.
        request = getattr(ws, "request", None)
        headers = getattr(request, "headers", None) or getattr(ws, "request_headers", {})
        self.ws_headers = {name.lower(): value for name, value in headers.items()}
        self.ws_path = getattr(request, "path", None) or getattr(ws, "path", "")
        try:
            hello = message_pb2.Message.FromString(await ws.recv())
            assert hello.type == message_pb2.MESSAGE_CLIENT_CONFIGURE_SESSION
            self.configure = hello.client_configure_session
            confirm = message_pb2.Message(type=message_pb2.MESSAGE_SERVER_CONFIRM_SESSION)
            confirm.server_confirm_session.connection_id = connection_id
            await ws.send(confirm.SerializeToString())

            requests: dict[str, _Request] = {}
            async for raw in ws:
                message = message_pb2.Message.FromString(raw)
                self.received.append(message)
                if message.type == message_pb2.MESSAGE_CLIENT_AUDIO_INPUT:
                    await self._audio(ws, connection_id, requests, message.client_audio_input)
                elif message.type == message_pb2.MESSAGE_CLIENT_INTERRUPT:
                    requests.pop(message.client_interrupt.req_id, None)
        except websockets.exceptions.ConnectionClosed:
            pass  # the client went away, or the test closed the socket
        finally:
            self.open_connections -= 1

    async def _audio(
        self,
        ws: Any,
        connection_id: str,
        requests: dict[str, _Request],
        audio_input: message_pb2.ClientAudioInput,
    ) -> None:
        assert self.configure is not None
        req_id = audio_input.req_id
        request = requests.setdefault(req_id, _Request())
        request.pending_bytes += len(audio_input.audio)
        while True:
            ms = self.first_segment_ms if request.segments == 0 else self.segment_ms
            segment_bytes = 2 * max(1, ms * self.configure.sample_rate // 1000)
            if request.pending_bytes < segment_bytes:
                break
            request.pending_bytes -= segment_bytes
            request.segments += 1
            await self._animate(ws, connection_id, req_id, end=False, with_animation=True)
        if audio_input.end:
            del requests[req_id]
            await self._animate(
                ws, connection_id, req_id, end=True, with_animation=request.pending_bytes > 0
            )

    async def _animate(
        self, ws: Any, connection_id: str, req_id: str, *, end: bool, with_animation: bool
    ) -> None:
        message = message_pb2.Message(type=message_pb2.MESSAGE_SERVER_RESPONSE_ANIMATION)
        animation = message.server_response_animation
        animation.connection_id = connection_id
        animation.req_id = req_id
        animation.end = end
        if with_animation:
            animation.animation = b"\x01" * self.animation_bytes
        await ws.send(message.SerializeToString())
