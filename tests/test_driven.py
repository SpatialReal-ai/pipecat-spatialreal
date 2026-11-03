import asyncio
import dataclasses
import socket
import time

import pytest
import spatialreal
from conftest import make, pcm_ms, until
from fake_driveningress import SESSION_TOKEN
from spatialreal.proto.generated.message_pb2 import (
    AUDIO_FORMAT_PCM_S16LE,
    EGRESS_TYPE_UNSPECIFIED,
    MESSAGE_SERVER_ERROR,
    MESSAGE_SERVER_RESPONSE_ANIMATION,
    Message,
)

from pipecat_spatialreal.driven import DrivenIngress, Segment


async def test_connect_sends_configure(fake):
    d = DrivenIngress(
        api_key="k",
        app_id="a",
        avatar_id="av",
        sample_rate=24000,
        console_endpoint_url=fake.console_url,
        ingress_endpoint_url=fake.ingress_url,
        on_segment=lambda s: None,
        on_down=lambda r: None,
    )
    await d.connect()
    await asyncio.sleep(0.1)
    assert fake.configure.sample_rate == 24000 and fake.configure.egress_type == 0 and d.connected
    await d.close()


async def test_segments_arrive_whole_and_in_order(fake):
    got = []
    d = make(fake, on_segment=got.append)
    await d.connect()
    req = await d.send(pcm_ms(300, 24000))
    await d.send(b"", end=True)
    await asyncio.sleep(0.3)
    assert [s.req_id for s in got] == [req, req] and got[0].end is False and got[1].end is True
    assert Message.FromString(got[0].wire).server_response_animation.req_id == req


async def test_interrupt_targets_last_req(fake):
    d = make(fake)
    await d.connect()
    req = await d.send(pcm_ms(40, 24000))
    assert await d.interrupt() == req
    await asyncio.sleep(0.1)
    assert fake.received[-1].client_interrupt.req_id == req


async def test_server_close_marks_down(fake):
    downs = []
    d = make(fake, on_down=downs.append)
    await d.connect()
    await fake.close_with(4001)
    await asyncio.sleep(0.2)
    assert downs and not d.connected
    with pytest.raises(RuntimeError):
        await d.send(b"\x00\x00")


# --- Beyond the brief: contract details its four tests leave open. The processor leans on each of
# --- them: what a segment is, when the link counts as lost, and what a lost or closed link does
# --- to the calls that follow.


def closed_port() -> int:
    """A local port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def test_segments_are_the_servers_bytes_unchanged(fake):
    got = []
    d = make(fake, on_segment=got.append)
    await d.connect()
    req = await d.send(pcm_ms(700, 24000), end=True)  # three full segments, then the end
    await until(lambda: len(got) == 4)
    assert [s.wire for s in got] == fake.animation_frames
    assert [(s.req_id, s.end) for s in got] == [(req, False)] * 3 + [(req, True)]


def test_segment_is_immutable():
    segment = Segment(req_id="r", wire=b"w", end=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        segment.end = True


async def test_connect_authenticates_and_names_the_avatar(fake):
    d = make(fake)
    await d.connect()
    assert fake.token_requests[-1][0] == "k"
    assert fake.ws_headers["x-app-id"] == "a"
    assert fake.ws_headers["x-session-key"] == SESSION_TOKEN
    assert fake.ws_path == "/websocket?id=av"


async def test_the_session_asks_for_pcm_animation_over_the_websocket(fake):
    d = make(fake)
    await d.connect()
    assert fake.configure.audio_format == AUDIO_FORMAT_PCM_S16LE
    assert fake.configure.egress_type == EGRESS_TYPE_UNSPECIFIED
    assert not fake.configure.HasField("livekit_egress")
    assert not fake.configure.HasField("agora_egress")


@pytest.mark.parametrize(
    ("overrides", "seconds"), [({}, 86400), ({"token_ttl_s": 3600}, 3600)], ids=["default", "set"]
)
async def test_token_lives_for_token_ttl_s(fake, overrides, seconds):
    d = make(fake, **overrides)
    before = int(time.time())
    await d.connect()
    expire_at = fake.token_requests[-1][1]["expireAt"]
    assert before + seconds <= expire_at <= int(time.time()) + seconds


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"console_endpoint_url": f"http://127.0.0.1:{closed_port()}"}, spatialreal.AvatarSDKError),
        ({"api_key": ""}, ValueError),
    ],
    ids=["console unreachable", "no api key"],
)
async def test_a_failed_connect_leaves_nothing_connected(fake, overrides, error):
    downs = []
    d = make(fake, on_down=downs.append, **overrides)
    with pytest.raises(error):
        await d.connect()
    await asyncio.sleep(0.05)
    assert not d.connected and downs == [] and fake.connections == 0
    with pytest.raises(RuntimeError):
        await d.send(b"\x00\x00")


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (4001, "creditsExhausted (4001)"),
        (4002, "sessionDurationExceeded (4002)"),
        (4003, "concurrencyLimit (4003)"),
        (4010, "sessionTokenInvalid (4010)"),
        (4503, "upstreamError (4503)"),
    ],
)
async def test_a_close_code_is_reported_once_as_a_readable_reason(fake, code, reason):
    downs = []
    d = make(fake, on_down=downs.append)
    await d.connect()
    await fake.close_with(code)
    await until(lambda: downs)
    await asyncio.sleep(0.2)  # the SDK reports the error and then the close; one loss
    assert downs == [reason] and not d.connected


async def test_a_close_without_an_error_is_reported_as_closed(fake):
    downs = []
    d = make(fake, on_down=downs.append)
    await d.connect()
    await fake.close_with(1000)
    await until(lambda: downs)
    await asyncio.sleep(0.2)
    assert downs == ["closed"] and not d.connected


async def test_a_server_error_and_the_close_that_follows_it_are_one_loss(fake):
    """The server sends the error first and closes with the same code."""
    downs = []
    d = make(fake, on_down=downs.append)
    await d.connect()
    error = Message(type=MESSAGE_SERVER_ERROR)
    error.server_error.code = 4001
    error.server_error.message = "Insufficient credits."
    await fake.send(error)
    await fake.close_with(4001)
    await until(lambda: downs)
    await asyncio.sleep(0.2)
    assert downs == ["creditsExhausted (4001)"]


async def test_a_server_error_that_leaves_the_connection_open_is_not_a_loss(fake):
    got, downs = [], []
    d = make(fake, on_segment=got.append, on_down=downs.append)
    await d.connect()
    error = Message(type=MESSAGE_SERVER_ERROR)
    error.server_error.message = "inference failed"  # code 0: no close follows
    await fake.send(error)
    await asyncio.sleep(0.2)
    assert d.connected and downs == []
    await d.send(pcm_ms(200, 24000), end=True)  # still works
    await until(lambda: len(got) == 2)


@pytest.mark.parametrize(
    "failure",
    [
        spatialreal.AvatarSDKError(
            spatialreal.AvatarSDKErrorCode.connectionClosed, "boom", phase="websocket_send"
        ),
        ValueError("WebSocket connection is not established"),
    ],
    ids=["transport error", "session already closed"],
)
async def test_a_failed_send_is_a_loss(fake, monkeypatch, failure):
    downs = []
    d = make(fake, on_down=downs.append)
    await d.connect()

    attempts = []

    async def broken(audio, end=False):
        attempts.append(audio)
        raise failure

    monkeypatch.setattr(d._link.session, "send_audio", broken)
    with pytest.raises(RuntimeError, match="driveningress not connected") as raised:
        await d.send(pcm_ms(40, 24000))
    assert raised.value.__cause__ is failure
    assert not d.connected and len(downs) == 1
    with pytest.raises(RuntimeError, match="driveningress not connected"):
        await d.send(pcm_ms(40, 24000))
    assert len(attempts) == 1 and len(downs) == 1  # a known loss does not try the wire again
    # The socket is in fact still open. When the server later closes it, that is not a second loss.
    await fake.close_with(4001)
    await asyncio.sleep(0.2)
    assert len(downs) == 1


async def test_a_failed_interrupt_is_a_loss_and_not_an_error(fake, monkeypatch):
    downs = []
    d = make(fake, on_down=downs.append)
    await d.connect()
    await d.send(pcm_ms(40, 24000))

    async def broken():
        raise spatialreal.AvatarSDKError(
            spatialreal.AvatarSDKErrorCode.connectionClosed, "boom", phase="websocket_send"
        )

    monkeypatch.setattr(d._link.session, "interrupt", broken)
    assert await d.interrupt() is None
    assert not d.connected and len(downs) == 1


async def test_a_raising_on_down_cannot_change_what_send_and_interrupt_do(fake, monkeypatch):
    failure = spatialreal.AvatarSDKError(
        spatialreal.AvatarSDKErrorCode.connectionClosed, "boom", phase="websocket_send"
    )

    def on_down(reason):
        raise KeyError("a bug in the callback")

    async def broken_send(audio, end=False):
        raise failure

    async def broken_interrupt():
        raise failure

    d = make(fake, on_down=on_down)
    await d.connect()
    monkeypatch.setattr(d._link.session, "send_audio", broken_send)
    with pytest.raises(RuntimeError, match="driveningress not connected") as raised:
        await d.send(pcm_ms(40, 24000))
    assert raised.value.__cause__ is failure and not d.connected

    await d.connect()
    await d.send(pcm_ms(40, 24000))
    monkeypatch.setattr(d._link.session, "interrupt", broken_interrupt)
    assert await d.interrupt() is None and not d.connected


async def test_nothing_to_interrupt_before_audio_or_without_a_link(fake):
    d = make(fake)
    assert await d.interrupt() is None  # never connected
    await d.connect()
    assert await d.interrupt() is None  # connected, nothing sent
    await d.send(pcm_ms(40, 24000))
    await d.close()
    assert await d.interrupt() is None  # closed
    assert not [m for m in fake.received if m.WhichOneof("data") == "client_interrupt"]


async def test_nothing_to_interrupt_once_the_link_is_lost(fake, monkeypatch):
    downs, attempts = [], []
    d = make(fake, on_down=downs.append)
    await d.connect()
    await d.send(pcm_ms(40, 24000))
    await fake.close_with(4001)
    await until(lambda: downs)

    async def interrupt():
        attempts.append(1)

    monkeypatch.setattr(d._link.session, "interrupt", interrupt)
    assert await d.interrupt() is None
    assert attempts == []  # a known loss does not try the wire


async def test_close_is_quiet_and_idempotent(fake):
    downs = []
    d = make(fake, on_down=downs.append)
    await d.connect()
    await d.send(pcm_ms(40, 24000))
    await d.close()
    await d.close()
    await until(lambda: fake.open_connections == 0)
    await asyncio.sleep(0.1)
    assert downs == [] and not d.connected
    with pytest.raises(RuntimeError, match="driveningress not connected"):
        await d.send(b"\x00\x00")


async def test_close_without_connect_is_fine(fake):
    d = make(fake)
    await d.close()
    assert not d.connected


async def test_connect_again_after_a_drop(fake):
    got, downs = [], []
    d = make(fake, on_segment=got.append, on_down=downs.append)
    await d.connect()
    first = await d.send(pcm_ms(40, 24000))
    await fake.close_with(4001)
    await until(lambda: downs)

    await d.connect()
    assert d.connected
    assert await d.interrupt() is None  # the new session has sent nothing
    second = await d.send(pcm_ms(200, 24000), end=True)
    await until(lambda: len(got) == 2)
    assert second != first and {s.req_id for s in got} == {second}
    assert len(downs) == 1 and fake.connections == 2


async def test_connect_replaces_a_live_session(fake):
    got, downs = [], []
    d = make(fake, on_segment=got.append, on_down=downs.append)
    await d.connect()
    await d.connect()
    await until(lambda: fake.open_connections == 1)
    await asyncio.sleep(0.1)
    assert d.connected and downs == [] and fake.connections == 2
    req = await d.send(pcm_ms(200, 24000), end=True)
    await until(lambda: len(got) == 2)
    assert {s.req_id for s in got} == {req}


async def test_close_during_the_token_exchange_opens_no_socket(fake):
    downs = []
    d = make(fake, on_down=downs.append)
    fake.token_gate = asyncio.Event()
    connecting = asyncio.create_task(d.connect())
    await until(lambda: fake.token_requests)  # the request is at the console
    await d.close()
    fake.token_gate.set()
    with pytest.raises(RuntimeError, match="closed while connecting"):
        await connecting
    assert fake.connections == 0  # the session was dropped before it opened a socket (it bills)
    assert not d.connected and downs == []


async def test_close_during_the_handshake_fails_connect_and_leaves_no_socket(fake):
    fake.confirm_gate = asyncio.Event()
    d = make(fake)
    connecting = asyncio.create_task(d.connect())
    await until(lambda: fake.open_connections == 1)  # the socket is open, the handshake is not done
    await d.close()
    with pytest.raises(spatialreal.AvatarSDKError):  # the handshake failed under its closed socket
        await connecting
    await until(lambda: fake.open_connections == 0)
    assert not d.connected


async def test_close_during_the_socket_upgrade_closes_the_socket(fake, monkeypatch):
    d = make(fake)
    start = spatialreal.AvatarSession.start

    async def close_then_start(session):
        await d.close()  # a close() that lands while start() has not opened its socket yet
        return await start(session)

    monkeypatch.setattr(spatialreal.AvatarSession, "start", close_then_start)
    with pytest.raises(RuntimeError, match="closed while connecting"):
        await d.connect()
    await until(lambda: fake.connections == 1 and fake.open_connections == 0)
    assert not d.connected


async def test_cancelling_connect_closes_the_half_open_session(fake):
    fake.confirm_gate = asyncio.Event()
    d = make(fake)
    connecting = asyncio.create_task(d.connect())
    await until(lambda: fake.open_connections == 1)  # the socket is open, the handshake is not done
    connecting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await connecting
    await until(lambda: fake.open_connections == 0)
    assert not d.connected


async def test_no_segments_after_a_loss(fake, monkeypatch):
    got, downs = [], []
    d = make(fake, on_segment=got.append, on_down=downs.append)
    await d.connect()
    frame = Message(type=MESSAGE_SERVER_RESPONSE_ANIMATION)
    frame.server_response_animation.req_id = "r1"
    await fake.send(frame)
    await until(lambda: len(got) == 1)  # frames get through while the session is up

    async def broken(audio, end=False):
        raise ValueError("WebSocket connection is not established")

    # The send fails, so the session is lost, while the server, which knows nothing of it,
    # goes on talking on the socket that is still open.
    monkeypatch.setattr(d._link.session, "send_audio", broken)
    with pytest.raises(RuntimeError):
        await d.send(pcm_ms(40, 24000))
    await fake.send(frame)
    await asyncio.sleep(0.2)
    assert len(got) == 1 and len(downs) == 1
