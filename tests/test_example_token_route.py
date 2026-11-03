"""Tests for the example bot's SpatialReal token route (``examples/bot.py``).

The route mints the session token the browser gives the web SDK, so the API key must reach the
console and nowhere else. The console is a local aiohttp server here; nothing leaves the machine.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest
from aiohttp import web

# The example imports FastAPI and the Pipecat runner, which the example extra brings: without it
# (the dev extra alone, as CI installs) this module is skipped, not an error.
SKIP_REASON = "needs the example extra: uv sync --extra example"
fastapi = pytest.importorskip("fastapi", reason=SKIP_REASON)
pytest.importorskip("dotenv", reason=SKIP_REASON)
pytest.importorskip("pipecat.runner.run", reason=SKIP_REASON, exc_type=ImportError)

HTTPException = fastapi.HTTPException

BOT_PATH = Path(__file__).resolve().parent.parent / "examples" / "bot.py"
TOKEN_PATH = "/v1/auth/session-token"
API_KEY = "sr-secret-key"


@pytest.fixture(scope="module")
def bot():
    """The example module, loaded as a file the way ``python examples/bot.py`` loads it.

    Importing the Pipecat runner reads a ``.env`` into ``os.environ`` and sets ``ENV``; the
    environment is put back afterwards so that no other test sees a developer's keys. The
    deprecation warnings the example's third-party imports raise (the legacy websockets server in
    Pipecat's eval transport, a deprecated TTS base class) are not ours to fix and are dropped
    here, so the package's own warning filter stays as strict as it was.
    """
    saved = dict(os.environ)
    spec = importlib.util.spec_from_file_location("example_bot", BOT_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec.loader.exec_module(module)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return module


class FakeConsole:
    """A console whose token route answers as told and remembers what it was asked."""

    def __init__(self, status: int = 200, reply=None):
        self.status = status
        self.reply = {"session_token": "tok-1"} if reply is None else reply
        self.path = ""
        self.headers = {}
        self.body = {}

    async def token(self, request: web.Request) -> web.Response:
        self.path = request.path
        self.headers = dict(request.headers)
        self.body = await request.json()
        if isinstance(self.reply, str):
            return web.Response(status=self.status, text=self.reply)
        return web.json_response(self.reply, status=self.status)


@pytest.fixture
async def console_for(aiohttp_server):
    """Start a fake console that answers as told; returns ``(console, origin)``."""

    async def start(status: int = 200, reply=None) -> tuple[FakeConsole, str]:
        console = FakeConsole(status, reply)
        app = web.Application()
        app.router.add_post(TOKEN_PATH, console.token)
        server = await aiohttp_server(app)
        return console, str(server.make_url("")).rstrip("/")

    return start


@pytest.fixture
def spatialreal_env(monkeypatch):
    """Set the three SpatialReal settings the route needs; no console endpoint yet."""
    monkeypatch.setenv("SPATIALREAL_API_KEY", API_KEY)
    monkeypatch.setenv("SPATIALREAL_APP_ID", "app-1")
    monkeypatch.setenv("SPATIALREAL_AVATAR_ID", "avatar-1")
    monkeypatch.delenv("SPATIALREAL_CONSOLE_ENDPOINT", raising=False)


# --- mint_session_token


async def test_mint_session_token_asks_the_console_the_way_the_sdk_does(bot, console_for):
    console, origin = await console_for()
    before = int(time.time())

    reply = await bot.mint_session_token(origin, API_KEY, ttl_s=600)

    after = int(time.time())
    assert reply["session_token"] == "tok-1"
    assert console.path == "/v1/auth/session-token"
    assert console.headers["X-Api-Key"] == API_KEY
    assert console.headers["Content-Type"] == "application/json"
    assert set(console.body) == {"expireAt"}
    assert before + 600 <= console.body["expireAt"] <= after + 600


async def test_mint_session_token_defaults_to_one_hour(bot, console_for):
    console, origin = await console_for()
    before = int(time.time())

    await bot.mint_session_token(origin, API_KEY)

    assert before + 3600 <= console.body["expireAt"] <= int(time.time()) + 3600


async def test_mint_session_token_tolerates_a_trailing_slash(bot, console_for):
    console, origin = await console_for()

    await bot.mint_session_token(origin + "/", API_KEY)

    assert console.path == "/v1/auth/session-token"


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
async def test_mint_session_token_raises_on_a_non_200(bot, console_for, status):
    _, origin = await console_for(status=status, reply={"errors": [{"detail": "no"}]})

    with pytest.raises(bot.SessionTokenError, match=str(status)) as raised:
        await bot.mint_session_token(origin, API_KEY)

    assert API_KEY not in str(raised.value)


@pytest.mark.parametrize("reply", [{}, {"session_token": ""}, [], "not json"])
async def test_mint_session_token_raises_when_the_reply_has_no_token(bot, console_for, reply):
    _, origin = await console_for(reply=reply)

    with pytest.raises(bot.SessionTokenError, match="session_token"):
        await bot.mint_session_token(origin, API_KEY)


async def test_mint_session_token_raises_when_the_console_is_down(bot, unused_tcp_port):
    with pytest.raises(bot.SessionTokenError, match="cannot reach"):
        await bot.mint_session_token(f"http://127.0.0.1:{unused_tcp_port}", API_KEY)


# --- GET /spatialreal-token


def test_the_route_is_on_the_runner_app(bot):
    from pipecat.runner.run import app

    routes = [r for r in app.routes if getattr(r, "path", "") == "/spatialreal-token"]

    assert [r.methods for r in routes] == [{"GET"}]
    assert routes[0].endpoint is bot.spatialreal_token


async def test_the_route_returns_the_token_and_ids_but_not_the_key(
    bot, console_for, spatialreal_env, monkeypatch
):
    console, origin = await console_for()
    monkeypatch.setenv("SPATIALREAL_CONSOLE_ENDPOINT", origin)

    reply = await bot.spatialreal_token()

    assert reply == {"session_token": "tok-1", "avatar_id": "avatar-1", "app_id": "app-1"}
    assert API_KEY not in str(reply)
    assert console.headers["X-Api-Key"] == API_KEY


async def test_the_route_asks_production_unless_told_otherwise(bot, spatialreal_env, monkeypatch):
    asked = []

    async def fake_mint(console_url, api_key, ttl_s=3600):
        asked.append((console_url, api_key))
        return {"session_token": "tok-2"}

    monkeypatch.setattr(bot, "mint_session_token", fake_mint)

    await bot.spatialreal_token()

    assert asked == [("https://api.spatialreal.cloud", API_KEY)]


@pytest.mark.parametrize(
    "missing", ["SPATIALREAL_API_KEY", "SPATIALREAL_APP_ID", "SPATIALREAL_AVATAR_ID"]
)
async def test_the_route_names_the_setting_that_is_missing(
    bot, spatialreal_env, monkeypatch, missing
):
    monkeypatch.delenv(missing)

    with pytest.raises(HTTPException) as raised:
        await bot.spatialreal_token()

    assert raised.value.status_code == 500
    assert missing in raised.value.detail


async def test_the_route_answers_502_when_the_console_refuses(
    bot, console_for, spatialreal_env, monkeypatch
):
    _, origin = await console_for(status=401, reply={"errors": [{"detail": "bad key"}]})
    monkeypatch.setenv("SPATIALREAL_CONSOLE_ENDPOINT", origin)

    with pytest.raises(HTTPException) as raised:
        await bot.spatialreal_token()

    assert raised.value.status_code == 502
    assert "401" in raised.value.detail
    assert API_KEY not in raised.value.detail


# --- the pages


def test_the_pages_are_served_from_the_runner_app(bot):
    from pipecat.runner.run import app

    mounted = {route.path for route in app.routes if type(route).__name__ == "Mount"}

    assert {"/avatar", "/probe"} <= mounted


# --- the file as a program


def _run_help(script: Path) -> subprocess.CompletedProcess:
    """Run ``script --help`` the way a developer would, with no SpatialReal or service keys set."""
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_APP_ID"))}
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_the_bot_starts_and_prints_its_help_without_any_keys():
    done = _run_help(BOT_PATH)

    assert done.returncode == 0, done.stderr
    assert "--transport" in done.stdout


def test_a_copy_of_bot_py_alone_still_starts_and_says_which_pages_it_does_not_serve(tmp_path):
    alone = tmp_path / "bot.py"  # no client/ or probe/ beside it
    shutil.copy(BOT_PATH, alone)

    done = _run_help(alone)

    assert done.returncode == 0, done.stderr
    assert "--transport" in done.stdout
    assert "Not serving /avatar/" in done.stderr
    assert "Not serving /probe/" in done.stderr
