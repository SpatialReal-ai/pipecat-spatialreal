"""Tests for the example bot's SpatialReal token route (``examples/bot.py``).

The route mints the session token the browser gives the web SDK, so the API key must reach the
console and nowhere else, and the token must reach the example's own page and nothing else: the
runner lets every web origin read its routes, so the route itself refuses other origins. The
console is a local aiohttp server here; nothing leaves the machine.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from aiohttp import web

# The example imports FastAPI and the Pipecat runner, which the example extra brings: without it
# (the dev extra alone) this module is skipped, not an error.
SKIP_REASON = "needs the example extra: uv sync --extra example"
fastapi = pytest.importorskip("fastapi", reason=SKIP_REASON)
pytest.importorskip("dotenv", reason=SKIP_REASON)
pytest.importorskip("pipecat.runner.run", reason=SKIP_REASON, exc_type=ImportError)

import httpx
from example_loader import BOT_PATH, TELEMETRY_OFF, load_bot
from fastapi.middleware.cors import CORSMiddleware
from spatialreal import environments
from spatialreal.session import _session_token_endpoint

TOKEN_PATH = "/v1/auth/session-token"
API_KEY = "sr-secret-key"
# Where the runner serves the bot, and so the origin of the pages it serves.
BOT_ORIGIN = "http://localhost:7860"


@pytest.fixture(scope="module")
def bot():
    """The example module (see :func:`example_loader.load_bot`)."""
    return load_bot()


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


@pytest.fixture
def ask(bot):
    """``GET /spatialreal-token`` over HTTP, with the headers a browser (or curl) would send.

    The route sits behind the runner's default CORS policy, the one the bot runs with when
    ``PIPECAT_ALLOWED_ORIGINS`` is not set: every origin allowed, with credentials
    (``pipecat/runner/run.py``, ``_configure_server_app``). The server is ``base_url``, by default
    where the runner serves the bot.
    """

    async def get(headers: dict[str, str], base_url: str = BOT_ORIGIN) -> httpx.Response:
        app = fastapi.FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.add_api_route("/spatialreal-token", bot.spatialreal_token, methods=["GET"])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
            return await client.get("/spatialreal-token", headers=headers)

    return get


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


@pytest.mark.parametrize("suffix", ["/", "/v1/console", "/v1/console/"])
async def test_mint_session_token_takes_the_console_url_in_every_form_the_sdk_takes(
    bot, console_for, suffix
):
    console, origin = await console_for()

    await bot.mint_session_token(origin + suffix, API_KEY)

    assert console.path == "/v1/auth/session-token"


@pytest.mark.parametrize(
    "value",
    [
        "https://api.example.com",
        "https://api.example.com/",
        "https://api.example.com/v1/console",
        "https://api.example.com/v1/console/",
        "http://127.0.0.1:8090",
        "https://api.example.com/prefix",
    ],
)
def test_the_console_url_is_read_exactly_as_the_sdk_reads_it(bot, value):
    # The processor's sessions build their token URL with the SDK's own function; the route's must
    # come out the same for every value SPATIALREAL_CONSOLE_ENDPOINT can hold.
    assert bot.console_root(value) + TOKEN_PATH == _session_token_endpoint(value)


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


# --- GET /spatialreal-token: who gets a token


def test_the_route_is_on_the_runner_app(bot):
    from pipecat.runner.run import app

    routes = [r for r in app.routes if getattr(r, "path", "") == "/spatialreal-token"]

    assert [r.methods for r in routes] == [{"GET"}]
    assert routes[0].endpoint is bot.spatialreal_token


# What browsers send: Sec-Fetch-Site on every request, Origin on every read from another origin
# (and on none for a same-origin GET).
FROM_ANOTHER_ORIGIN = [
    pytest.param({"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}, id="site"),
    pytest.param({"Origin": "https://evil.example"}, id="origin-without-sec-fetch-site"),
    pytest.param({"Origin": "http://localhost:3000", "Sec-Fetch-Site": "same-site"}, id="port"),
    pytest.param({"Origin": "https://localhost:7860"}, id="scheme"),
    pytest.param({"Origin": "null"}, id="opaque-origin"),
    pytest.param({"Sec-Fetch-Site": "cross-site"}, id="no-cors-from-another-site"),
]


@pytest.mark.parametrize("headers", FROM_ANOTHER_ORIGIN)
async def test_a_page_of_another_origin_gets_no_token(
    ask, console_for, spatialreal_env, monkeypatch, headers
):
    console, origin = await console_for()
    monkeypatch.setenv("SPATIALREAL_CONSOLE_ENDPOINT", origin)

    reply = await ask(headers)

    assert reply.status_code == 403
    assert "tok-1" not in reply.text
    assert console.path == ""  # no token was minted for it


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({"Sec-Fetch-Site": "same-origin"}, id="page-served-here"),
        pytest.param(
            {"Sec-Fetch-Site": "same-origin", "Origin": BOT_ORIGIN}, id="page-served-here-origin"
        ),
        pytest.param({"Sec-Fetch-Site": "none"}, id="address-bar"),
        pytest.param({}, id="curl-or-a-server"),
    ],
)
async def test_the_page_served_here_and_callers_that_are_not_pages_get_a_token(
    ask, console_for, spatialreal_env, monkeypatch, headers
):
    _, origin = await console_for()
    monkeypatch.setenv("SPATIALREAL_CONSOLE_ENDPOINT", origin)

    reply = await ask(headers)

    assert reply.status_code == 200, reply.text
    assert reply.json()["session_token"] == "tok-1"


@pytest.mark.parametrize(
    "server, page",
    [
        ("https://bot.example", "https://bot.example"),
        ("https://bot.example", "https://bot.example:443"),
        ("https://bot.example:443", "https://bot.example"),
        ("http://bot.example", "http://bot.example:80"),
    ],
)
async def test_the_same_origin_is_recognised_however_its_port_is_written(
    ask, console_for, spatialreal_env, monkeypatch, server, page
):
    _, origin = await console_for()
    monkeypatch.setenv("SPATIALREAL_CONSOLE_ENDPOINT", origin)

    reply = await ask({"Origin": page, "Sec-Fetch-Site": "same-origin"}, base_url=server)

    assert reply.status_code == 200, reply.text


# --- GET /spatialreal-token: what the page gets


async def test_the_page_gets_the_token_and_ids_but_not_the_key(
    ask, console_for, spatialreal_env, monkeypatch
):
    console, origin = await console_for()
    # SPATIALREAL_CONSOLE_ENDPOINT is unset: the console is the one the SDK resolves for the
    # bot's environment, here the config service's answer for it, as the SDK caches it.
    monkeypatch.setattr(environments, "_cache", {"us-west": {"api": origin}})

    reply = await ask({"Sec-Fetch-Site": "same-origin"})

    assert reply.status_code == 200, reply.text
    assert reply.json() == {
        "session_token": "tok-1",
        "avatar_id": "avatar-1",
        "app_id": "app-1",
        "endpoints": {},  # production: the web SDK finds it by itself
    }
    assert API_KEY not in reply.text
    assert console.headers["X-Api-Key"] == API_KEY


@pytest.mark.parametrize("suffix", ["", "/", "/v1/console", "/v1/console/"])
async def test_another_deployment_is_asked_as_the_sdk_asks_it_and_named_to_the_page(
    ask, console_for, spatialreal_env, monkeypatch, suffix
):
    console, origin = await console_for()
    monkeypatch.setenv("SPATIALREAL_CONSOLE_ENDPOINT", origin + suffix)

    reply = await ask({"Sec-Fetch-Site": "same-origin"})

    assert reply.status_code == 200, reply.text
    assert console.path == "/v1/auth/session-token"
    # what the page passes to `new SpatialReal({ endpoints })`: a root with no trailing slash
    assert reply.json()["endpoints"] == {"api": origin}


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
async def test_without_a_console_endpoint_the_route_asks_the_console_the_sdk_resolves(
    bot, ask, spatialreal_env, monkeypatch, value
):
    if value is not None:
        monkeypatch.setenv("SPATIALREAL_CONSOLE_ENDPOINT", value)  # as a `.env` line `KEY=` gives
    monkeypatch.setattr(environments, "_cache", {})  # the config service did not answer
    asked = []

    async def fake_mint(console_url, api_key, ttl_s=3600):
        asked.append((console_url, api_key))
        return {"session_token": "tok-2"}

    monkeypatch.setattr(bot, "mint_session_token", fake_mint)

    reply = await ask({})

    assert reply.status_code == 200, reply.text
    # the SDK's built-in preset for the bot's environment
    assert asked == [("https://api.spatialreal.cloud", API_KEY)]
    assert reply.json()["endpoints"] == {}


@pytest.mark.parametrize(
    "missing", ["SPATIALREAL_API_KEY", "SPATIALREAL_APP_ID", "SPATIALREAL_AVATAR_ID"]
)
async def test_the_route_names_the_setting_that_is_missing(
    ask, spatialreal_env, monkeypatch, missing
):
    monkeypatch.delenv(missing)

    reply = await ask({})

    assert reply.status_code == 500
    assert missing in reply.json()["detail"]


async def test_the_route_answers_502_when_the_console_refuses(
    ask, console_for, spatialreal_env, monkeypatch
):
    _, origin = await console_for(status=401, reply={"errors": [{"detail": "bad key"}]})
    monkeypatch.setenv("SPATIALREAL_CONSOLE_ENDPOINT", origin)

    reply = await ask({})

    assert reply.status_code == 502
    assert "401" in reply.json()["detail"]
    assert API_KEY not in reply.text


# --- the pages


def test_the_pages_are_served_from_the_runner_app(bot):
    from pipecat.runner.run import app

    mounted = {route.path for route in app.routes if type(route).__name__ == "Mount"}

    assert "/avatar" in mounted


# --- the file as a program


def _run_help(script: Path) -> subprocess.CompletedProcess:
    """Run ``script --help`` the way a developer would, with no SpatialReal or service keys set."""
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_APP_ID"))}
    env.update(TELEMETRY_OFF)
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
    alone = tmp_path / "bot.py"  # no client/ beside it
    shutil.copy(BOT_PATH, alone)

    done = _run_help(alone)

    assert done.returncode == 0, done.stderr
    assert "--transport" in done.stdout
    assert "Not serving /avatar/" in done.stderr
