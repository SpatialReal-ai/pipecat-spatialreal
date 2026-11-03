"""The fake backend is test infrastructure whose segment boundaries other tests rely on, so its
rules are pinned here with a bare WebSocket client and no SDK in between."""

import asyncio

import aiohttp
import pytest
import websockets
from conftest import pcm_ms
from fake_driveningress import SESSION_TOKEN, TOKEN_PATH, FakeDriveningress
from spatialreal.proto.generated.message_pb2 import (
    MESSAGE_CLIENT_AUDIO_INPUT,
    MESSAGE_CLIENT_CONFIGURE_SESSION,
    MESSAGE_CLIENT_INTERRUPT,
    MESSAGE_SERVER_CONFIRM_SESSION,
    MESSAGE_SERVER_RESPONSE_ANIMATION,
    Message,
)

RATE = 24000


def ws_url(fake: FakeDriveningress) -> str:
    return fake.ingress_url.replace("http://", "ws://") + "/websocket?id=av"


async def handshake(ws, sample_rate: int = RATE) -> Message:
    """Do what the SDK does on connect and return the server's confirmation."""
    hello = Message(type=MESSAGE_CLIENT_CONFIGURE_SESSION)
    hello.client_configure_session.sample_rate = sample_rate
    await ws.send(hello.SerializeToString())
    return Message.FromString(await asyncio.wait_for(ws.recv(), 2))


def audio(req_id: str, ms: int, *, end: bool = False, fill: bytes = b"", rate: int = RATE) -> bytes:
    msg = Message(type=MESSAGE_CLIENT_AUDIO_INPUT)
    msg.client_audio_input.req_id = req_id
    msg.client_audio_input.audio = fill or pcm_ms(ms, rate)
    msg.client_audio_input.end = end
    return msg.SerializeToString()


async def replies(ws, count: int) -> list[Message]:
    """Read exactly ``count`` server messages, then check that no more follows."""
    got = [Message.FromString(await asyncio.wait_for(ws.recv(), 2)) for _ in range(count)]
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(ws.recv(), 0.05)
    return got


def shape(messages: list[Message]) -> list[tuple[bool, int]]:
    """Reduce animation replies to ``(end, animation size)``."""
    assert all(m.type == MESSAGE_SERVER_RESPONSE_ANIMATION for m in messages)
    return [
        (m.server_response_animation.end, len(m.server_response_animation.animation))
        for m in messages
    ]


async def test_handshake_is_recorded_and_confirmed(fake):
    async with websockets.connect(ws_url(fake)) as ws:
        confirm = await handshake(ws, 16000)
        assert confirm.type == MESSAGE_SERVER_CONFIRM_SESSION
        assert confirm.server_confirm_session.connection_id == "conn-1"
    assert fake.configure.sample_rate == 16000
    assert fake.ws_path == "/websocket?id=av"
    assert fake.connections == 1
    await asyncio.sleep(0.05)
    assert fake.open_connections == 0


async def test_token_route_answers_and_records_the_request(fake):
    async with (
        aiohttp.ClientSession() as http,
        http.post(
            fake.console_url + TOKEN_PATH, json={"expireAt": 7}, headers={"X-Api-Key": "k"}
        ) as response,
    ):
        assert response.status == 200
        assert await response.json() == {"session_token": SESSION_TOKEN}
    assert fake.token_requests == [("k", {"expireAt": 7})]


async def test_segments_fill_at_the_first_size_then_the_regular_size():
    fake = FakeDriveningress(first_segment_ms=200, segment_ms=100, animation_bytes=16)
    async with fake, websockets.connect(ws_url(fake)) as ws:
        await handshake(ws)
        # 450 ms in one message: segments of 200, 100 and 100 ms, and 50 ms left over.
        await ws.send(audio("r1", 450))
        assert shape(await replies(ws, 3)) == [(False, 16)] * 3
        await ws.send(audio("r1", 0, end=True))
        assert shape(await replies(ws, 1)) == [(True, 16)]


async def test_segments_do_not_depend_on_how_the_audio_is_cut():
    """The same 450 ms as one message, as 45 pieces and as 7 ms pieces gives the same replies."""
    outcomes = []
    for cuts in ([450], [10] * 45, [7] * 64 + [2]):
        fake = FakeDriveningress(first_segment_ms=200, segment_ms=100, animation_bytes=4)
        async with fake, websockets.connect(ws_url(fake)) as ws:
            await handshake(ws)
            for ms in cuts:
                await ws.send(audio("r1", ms))
            await ws.send(audio("r1", 0, end=True))
            outcomes.append(shape(await replies(ws, 4)))
    assert outcomes == [[(False, 4)] * 3 + [(True, 4)]] * 3


async def test_segment_size_follows_the_sessions_sample_rate(fake):
    async with websockets.connect(ws_url(fake)) as ws:
        await handshake(ws, 16000)
        await ws.send(audio("r1", 200, rate=16000))  # 3200 samples fill the 200 ms segment
        assert shape(await replies(ws, 1)) == [(False, 4800)]


async def test_end_answers_even_when_nothing_is_left(fake):
    async with websockets.connect(ws_url(fake)) as ws:
        await handshake(ws)
        # Exactly one segment's worth: the segment goes out, then an empty end.
        await ws.send(audio("r1", 200, end=True))
        assert shape(await replies(ws, 2)) == [(False, 4800), (True, 0)]
        # No audio at all: still one empty end.
        await ws.send(audio("r2", 0, end=True))
        assert shape(await replies(ws, 1)) == [(True, 0)]


async def test_a_short_request_is_answered_on_end(fake):
    fake.first_segment_ms = 10_000
    async with websockets.connect(ws_url(fake)) as ws:
        await handshake(ws)
        await ws.send(audio("r1", 120))
        await ws.send(audio("r1", 0, end=True))
        assert shape(await replies(ws, 1)) == [(True, 4800)]


async def test_each_request_counts_for_itself(fake):
    async with websockets.connect(ws_url(fake)) as ws:
        await handshake(ws)
        await ws.send(audio("r1", 150))
        await ws.send(audio("r2", 150))  # neither request reaches 200 ms
        await ws.send(audio("r1", 50))  # r1 does
        got = await replies(ws, 1)
        assert got[0].server_response_animation.req_id == "r1"


async def test_interrupt_drops_what_was_left_over(fake):
    async with websockets.connect(ws_url(fake)) as ws:
        await handshake(ws)
        await ws.send(audio("r1", 150))
        interrupt = Message(type=MESSAGE_CLIENT_INTERRUPT)
        interrupt.client_interrupt.req_id = "r1"
        await ws.send(interrupt.SerializeToString())
        # Were the 150 ms kept, this end would flush them with an animation.
        await ws.send(audio("r1", 0, end=True))
        assert shape(await replies(ws, 1)) == [(True, 0)]


async def test_received_audio_is_concatenated_per_request(fake):
    async with websockets.connect(ws_url(fake)) as ws:
        await handshake(ws)
        await ws.send(audio("r1", 0, fill=b"\x01\x00" * 480))  # 20 ms
        await ws.send(audio("r2", 0, fill=b"\x07\x00" * 480))
        await ws.send(audio("r1", 0, fill=b"\x02\x00" * 480))
        await asyncio.sleep(0.1)
    assert fake.received_audio_bytes("r1") == b"\x01\x00" * 480 + b"\x02\x00" * 480
    assert fake.received_audio_ms("r1") == 40.0
    assert fake.received_audio_ms("nobody") == 0.0
    assert all(m.type == MESSAGE_CLIENT_AUDIO_INPUT for m in fake.received)  # no configure


async def test_close_with_closes_the_socket_and_keeps_serving(fake):
    async with websockets.connect(ws_url(fake)) as ws:
        await handshake(ws)
        await fake.close_with(4001)
        with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
            await asyncio.wait_for(ws.recv(), 2)
        assert closed.value.rcvd.code == 4001
    async with websockets.connect(ws_url(fake)) as ws:
        assert (await handshake(ws)).server_confirm_session.connection_id == "conn-2"
    assert fake.connections == 2


async def test_the_token_route_can_be_held_back(fake):
    fake.token_gate = asyncio.Event()

    async def post() -> int:
        async with (
            aiohttp.ClientSession() as http,
            http.post(fake.console_url + TOKEN_PATH, json={}) as response,
        ):
            return response.status

    call = asyncio.create_task(post())
    await asyncio.sleep(0.1)
    assert not call.done()
    fake.token_gate.set()
    assert await asyncio.wait_for(call, 2) == 200
