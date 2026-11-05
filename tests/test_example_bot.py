"""The example bot's wiring (``examples/bot.py``): what ``run_bot`` builds, and the pages it serves.

``run_bot`` runs here with a fake transport, whose input and output are bare processors, and with a
fake ``WorkerRunner`` that keeps the worker instead of running it, so nothing connects anywhere. The
services are the real ones, built with placeholder keys: building them opens no connection.
"""

import asyncio
import base64
import json
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

# The example needs the example extra: without it this module is skipped, not an error.
SKIP_REASON = "needs the example extra: uv sync --extra example"
pytest.importorskip("fastapi", reason=SKIP_REASON)
pytest.importorskip("dotenv", reason=SKIP_REASON)
pytest.importorskip("pipecat.runner.run", reason=SKIP_REASON, exc_type=ImportError)

import httpx
from example_loader import load_bot
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.types import DailyRunnerArguments, RunnerArguments
from pipecat.transports.base_transport import BaseTransport
from spatialreal.session import _session_token_endpoint

from pipecat_spatialreal import SpatialRealAvatarProcessor

with warnings.catch_warnings():
    # The module defines a TTS base class Pipecat deprecates, and warns as it does: not ours.
    warnings.simplefilter("ignore", DeprecationWarning)
    from pipecat.services.tts_service import TTSService

# Building the bot's real services loads Pipecat's models through an importlib API Python
# deprecates; those warnings are Pipecat's, not this package's.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:pipecat\\.")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
KEYS = {
    "SPATIALREAL_API_KEY": "sr-key",
    "SPATIALREAL_APP_ID": "app-1",
    "SPATIALREAL_AVATAR_ID": "avatar-1",
    "DEEPGRAM_API_KEY": "dg-key",
    "CARTESIA_API_KEY": "ca-key",
    "GEMINI_API_KEY": "gm-key",
}


class FakeTransport(BaseTransport):
    """A transport with an input and an output processor and nothing behind them.

    It keeps the event handlers the bot registers, so that a test can call them.
    """

    def __init__(self):
        super().__init__()
        self.handlers: dict[str, list] = {}
        self._input = FrameProcessor(name="transport-input")
        self._output = FrameProcessor(name="transport-output")

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    def add_event_handler(self, event_name: str, handler):
        self.handlers.setdefault(event_name, []).append(handler)


class FakeRunner:
    """Stands in for ``WorkerRunner``: keeps the workers, counts runs and cancels."""

    def __init__(self, **kwargs):
        self.workers = []
        self.runs = 0
        self.cancels = 0

    async def add_workers(self, *workers):
        self.workers.extend(workers)

    async def run(self):
        self.runs += 1

    async def cancel(self):
        self.cancels += 1


async def wire(monkeypatch, **env: str) -> SimpleNamespace:
    """Run ``run_bot`` on a fake transport with the keys set, and ``env`` on top of them.

    Returns:
        The bot module, the transport, the runner, the worker it was given, and the keyword
        arguments the avatar processor was built with.
    """
    bot = load_bot()
    for name in ("SPATIALREAL_CONSOLE_ENDPOINT", "SPATIALREAL_INGRESS_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {**KEYS, **env}.items():
        monkeypatch.setenv(name, value)
    runners: list[FakeRunner] = []
    avatar_args: list[dict] = []

    def make_runner(**kwargs) -> FakeRunner:
        runners.append(FakeRunner(**kwargs))
        return runners[-1]

    class RecordingAvatar(SpatialRealAvatarProcessor):
        def __init__(self, **kwargs):
            avatar_args.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(bot, "WorkerRunner", make_runner)
    monkeypatch.setattr(bot, "SpatialRealAvatarProcessor", RecordingAvatar)
    transport = FakeTransport()

    await bot.run_bot(transport, RunnerArguments())

    (runner,) = runners
    (worker,) = runner.workers
    (args,) = avatar_args
    return SimpleNamespace(bot=bot, transport=transport, runner=runner, worker=worker, avatar=args)


def downstream_of(processor: FrameProcessor) -> list[FrameProcessor]:
    """The processors a frame pushed downstream from ``processor`` passes, in order."""
    chain = []
    while processor is not None:
        chain.append(processor)
        processor = processor.next
    return chain


# --- run_bot


async def test_the_tts_audio_reaches_the_avatar_processor_before_the_transport_output(monkeypatch):
    wired = await wire(monkeypatch)

    chain = downstream_of(wired.transport.input())

    tts = [i for i, p in enumerate(chain) if isinstance(p, TTSService)]
    avatars = [i for i, p in enumerate(chain) if isinstance(p, SpatialRealAvatarProcessor)]
    assert len(tts) == 1 and len(avatars) == 1, [type(p).__name__ for p in chain]
    # The processor sees the TTS audio before the room plays it, and nothing comes in between.
    assert chain[tts[0] + 1] is chain[avatars[0]]
    assert chain[avatars[0] + 1] is wired.transport.output()
    assert chain.count(wired.transport.output()) == 1


async def test_the_session_ends_when_the_client_leaves(monkeypatch):
    wired = await wire(monkeypatch)
    assert set(wired.transport.handlers) == {"on_client_connected", "on_client_disconnected"}
    (on_client_disconnected,) = wired.transport.handlers["on_client_disconnected"]
    assert wired.runner.cancels == 0

    await on_client_disconnected(wired.transport, object())

    # The bot, and its billed SpatialReal connection, end with the client.
    assert wired.runner.cancels == 1


def greetings(wired, monkeypatch) -> list:
    """Every LLMRunFrame the bot queues on its worker from now on."""
    from pipecat.frames.frames import LLMRunFrame

    queued: list = []

    async def queue_frames(frames):
        queued.extend(f for f in frames if isinstance(f, LLMRunFrame))

    monkeypatch.setattr(wired.worker, "queue_frames", queue_frames)
    return queued


def client_ready_handlers(wired) -> list:
    """The example's own on_client_ready handlers (the worker adds one of its own, set_bot_ready)."""
    handlers = wired.worker.rtvi._event_handlers["on_client_ready"].handlers
    return [h for h in handlers if "run_bot" in h.__qualname__]


async def test_the_bot_greets_when_the_rtvi_client_is_ready_not_when_it_joins(monkeypatch):
    wired = await wire(monkeypatch)
    queued = greetings(wired, monkeypatch)
    (on_client_connected,) = wired.transport.handlers["on_client_connected"]

    await on_client_connected(wired.transport, object())
    assert queued == []  # joined, not ready: the browser may not receive the bot's track yet

    assert len(client_ready_handlers(wired)) == 1
    for handler in client_ready_handlers(wired):
        await handler(wired.worker.rtvi)
    assert len(queued) == 1

    await asyncio.sleep(0.01)
    for handler in client_ready_handlers(wired):
        await handler(wired.worker.rtvi)
    assert len(queued) == 1  # once per session


async def test_a_participant_without_an_rtvi_client_is_greeted_after_the_fallback(monkeypatch):
    wired = await wire(monkeypatch)
    monkeypatch.setattr(wired.bot, "GREETING_FALLBACK_S", 0.05)
    queued = greetings(wired, monkeypatch)
    (on_client_connected,) = wired.transport.handlers["on_client_connected"]

    await on_client_connected(wired.transport, object())
    await asyncio.sleep(0.2)

    assert len(queued) == 1


async def test_leaving_before_the_fallback_cancels_the_greeting(monkeypatch):
    wired = await wire(monkeypatch)
    monkeypatch.setattr(wired.bot, "GREETING_FALLBACK_S", 0.05)
    queued = greetings(wired, monkeypatch)
    (on_client_connected,) = wired.transport.handlers["on_client_connected"]
    (on_client_disconnected,) = wired.transport.handlers["on_client_disconnected"]

    await on_client_connected(wired.transport, object())
    await on_client_disconnected(wired.transport, object())
    await asyncio.sleep(0.2)

    assert queued == []


async def test_the_runner_runs_the_one_pipeline_worker(monkeypatch):
    wired = await wire(monkeypatch)

    assert isinstance(wired.worker, PipelineWorker)
    assert wired.runner.runs == 1


@pytest.mark.parametrize(
    "console",
    [None, "https://console.example", "https://console.example/v1/console/"],
    ids=["environment", "override", "legacy-override"],
)
async def test_the_processor_and_the_token_route_ask_the_same_console(monkeypatch, console):
    env = {} if console is None else {"SPATIALREAL_CONSOLE_ENDPOINT": console}
    ingress = "https://ingress.example/v2/driveningress"
    wired = await wire(monkeypatch, SPATIALREAL_INGRESS_ENDPOINT=ingress, **env)
    bot = wired.bot
    resolved = []

    async def resolve(environment):
        resolved.append(environment)
        return SimpleNamespace(api="https://api.of-the-environment.example", driveningress="")

    monkeypatch.setattr(bot.environments, "resolve", resolve)

    route_console = await bot.resolve_console_root()

    assert wired.avatar["environment"] == bot.ENVIRONMENT
    assert wired.avatar["ingress_endpoint_url"] == ingress
    if console is None:
        # Both leave it to the SDK's lookup of the same environment.
        assert wired.avatar["console_endpoint_url"] == ""
        assert resolved == [bot.ENVIRONMENT]
        assert route_console == "https://api.of-the-environment.example"
    else:
        # The processor's SDK session builds its token URL from the same value.
        assert wired.avatar["console_endpoint_url"] == console
        assert route_console + "/v1/auth/session-token" == _session_token_endpoint(console)


async def test_the_page_knows_the_bot_by_the_names_pipecats_runner_gives_it(monkeypatch):
    # The page hands the web SDK the audio of one participant, the bot, which it tells by the name
    # (Daily) or the identity (LiveKit) the development runner gives the bot. A runner that named it
    # otherwise would leave the avatar without a voice, so the names are checked against the runner.
    page = (EXAMPLES / "client" / "index.html").read_text()
    daily_name = re.search(r"const DAILY_BOT_NAME = '([^']+)'", page).group(1)
    identity = re.search(
        r"const identity = `([^`$]+)\$\{String\(started\.sessionId\)\.slice\(0, (\d+)\)\}`", page
    )
    prefix, length = identity.group(1), int(identity.group(2))

    # Daily: the participant the runner's bot joins as (pipecat.runner.utils.create_transport).
    import pipecat.transports.daily.transport as daily

    joined_as = []

    class RecordingDailyTransport:
        def __init__(self, room_url, token, bot_name, **kwargs):
            joined_as.append(bot_name)

    monkeypatch.setattr(daily, "DailyTransport", RecordingDailyTransport)
    from pipecat.runner.utils import create_transport

    runner_args = DailyRunnerArguments(room_url="https://example.daily.co/room", token="t")
    await create_transport(runner_args, load_bot().transport_params)
    assert joined_as == [daily_name]

    # LiveKit: the identity in the agent token POST /start mints for the bot's session.
    from pipecat.runner.livekit import generate_session_tokens

    session_id = "abcdef12-3456-4789-8abc-def012345678"
    agent_token, _ = generate_session_tokens(
        "room", session_id, "key", "a-secret-of-32-characters-or-so!"
    )
    claims = agent_token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4)))
    assert claims["sub"] == prefix + session_id[:length]


# --- the pages it serves


def _documented_pages() -> set[str]:
    """Every page of the bot's server that the example's documentation tells a reader to open."""
    paths = set()
    for doc in [EXAMPLES / "README.md", *EXAMPLES.glob("*/README.md")]:
        for url in re.findall(r"http://localhost:7860(/[^\s>)`'\"]*)", doc.read_text()):
            paths.add(urlsplit(url).path)
    paths |= set(re.findall(r"^\* ``(/[a-z]+/)``", load_bot().__doc__, re.MULTILINE))
    return paths


async def test_every_page_the_documentation_names_is_served():
    from pipecat.runner.run import app

    paths = _documented_pages()
    assert {"/avatar/", "/probe/"} <= paths, paths

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:7860") as client:
        for path in sorted(paths):
            reply = await client.get(path)
            assert reply.status_code == 200, f"{path}: {reply.status_code}"
            assert reply.headers["content-type"].startswith("text/html"), path
