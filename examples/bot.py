"""A Pipecat voice bot with a SpatialReal avatar: the runner-based example.

The bot has the shape of Pipecat's own ``06-voice-agent`` example (Deepgram, OpenAI and Cartesia)
with one processor, ``SpatialRealAvatarProcessor``, added after the TTS. Run it with Pipecat's
development runner, on Daily or LiveKit::

    uv run python examples/bot.py -t daily      # or: -t livekit

Pipecat's development runner serves the bot on http://localhost:7860: ``POST /start`` starts a
session and returns the room the browser joins. This file adds three things to that server:

* ``GET /spatialreal-token``: a session token for the browser, minted with the SpatialReal API key
  (see :func:`mint_session_token`). The web SDK needs it to download the avatar, and the key
  itself must never reach the browser. Your own app serves such a route too; this one has no
  authentication, so it is for local use.
* ``/avatar/``: the example web page, ``examples/client`` (run it from here: the page and the
  route then share an origin).
* ``/probe/``: ``examples/probe/receiver.html``, the silent receiver that measures how the
  envelope and the bot's audio line up in a Daily room. A developer tool for this package.

Settings come from the environment or a ``.env`` file; ``env.example`` lists them. The runner
reads the transport's own (``DAILY_API_KEY``, ``LIVEKIT_URL`` and the like).
"""

import json
import os
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.run import app, main
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.workers.runner import WorkerRunner

from pipecat_spatialreal import SpatialRealAvatarProcessor

load_dotenv(override=True)

# Where session tokens come from unless SPATIALREAL_CONSOLE_ENDPOINT says otherwise.
DEFAULT_CONSOLE_ENDPOINT = "https://api.spatialreal.cloud"
SESSION_TOKEN_PATH = "/v1/auth/session-token"

# A Cartesia voice that Pipecat's own documentation uses; CARTESIA_VOICE_ID replaces it.
DEFAULT_VOICE_ID = "71a7ad14-091c-4e8e-a314-022ece01c121"

SYSTEM_INSTRUCTION = (
    "You are a friendly voice assistant. Your replies are spoken aloud: keep them to a few short "
    "sentences of plain text, with no lists, markdown or emoji."
)

# One entry per transport the runner can hand this bot, with both audio directions on. The Silero
# VAD sits on the user aggregator in run_bot().
transport_params = {
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "livekit": lambda: LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


class SessionTokenError(RuntimeError):
    """The console would not issue a session token, or could not be reached."""


async def mint_session_token(console_url: str, api_key: str, ttl_s: int = 3600) -> dict:
    """Exchange the SpatialReal API key for a short-lived session token.

    This is the request the ``spatialreal`` Python SDK makes when it opens a session. The web SDK
    takes the token as its ``credential``, so a browser never needs the API key.

    Args:
        console_url: Origin of the console API, for example ``https://api.spatialreal.cloud``.
        api_key: The SpatialReal API key. It is sent to the console and nowhere else.
        ttl_s: Seconds the token stays valid.

    Returns:
        The console's JSON reply; ``session_token`` is the token.

    Raises:
        SessionTokenError: If the console cannot be reached, answers with anything but 200, or
            leaves the token out of its reply.
    """
    url = console_url.rstrip("/") + SESSION_TOKEN_PATH
    payload = {"expireAt": int(time.time()) + ttl_s}
    headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as http,
            http.post(url, json=payload, headers=headers) as response,
        ):
            status, body = response.status, await response.text()
    except (aiohttp.ClientError, TimeoutError) as e:
        raise SessionTokenError(f"cannot reach the console at {console_url}: {e}") from e
    if status != 200:
        raise SessionTokenError(f"the console answered HTTP {status}: {body[:200]}")
    try:
        reply = json.loads(body)
    except ValueError:
        reply = None
    if not isinstance(reply, dict) or not reply.get("session_token"):
        raise SessionTokenError("the console's reply has no session_token")
    return reply


def _require_env(name: str) -> str:
    """Return the environment variable ``name``, or fail naming it.

    Args:
        name: The variable.

    Returns:
        Its value.

    Raises:
        RuntimeError: If it is unset or empty.
    """
    value = os.getenv(name, "")
    if not value:
        raise RuntimeError(f"{name} is not set (see env.example)")
    return value


@app.get("/spatialreal-token")
async def spatialreal_token() -> dict[str, str]:
    """Give the browser what the web SDK needs to show the avatar.

    Returns:
        ``session_token`` (the web SDK's ``credential``), ``avatar_id`` and ``app_id``. The API
        key stays here.

    Raises:
        HTTPException: 500 if a SpatialReal setting is missing, 502 if the console refuses.
    """
    try:
        api_key = _require_env("SPATIALREAL_API_KEY")
        app_id = _require_env("SPATIALREAL_APP_ID")
        avatar_id = _require_env("SPATIALREAL_AVATAR_ID")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    console_url = os.getenv("SPATIALREAL_CONSOLE_ENDPOINT") or DEFAULT_CONSOLE_ENDPOINT
    try:
        minted = await mint_session_token(console_url, api_key)
    except SessionTokenError as e:
        logger.error(f"Could not mint a SpatialReal session token: {e}")
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"session_token": minted["session_token"], "avatar_id": avatar_id, "app_id": app_id}


def _serve_page(route: str, directory: Path) -> None:
    """Serve a directory of this example at ``route``, if this copy of the file has it.

    The pages are optional: a copy of ``bot.py`` on its own still gives the bot and the token
    route, and the log says which pages are not served.

    Args:
        route: The URL path to serve the page at, for example ``/avatar``.
        directory: The directory beside this file that holds the page.
    """
    if directory.is_dir():
        app.mount(route, StaticFiles(directory=directory, html=True), name=route.strip("/"))
    else:
        logger.info(f"Not serving {route}/: {directory} does not exist")


# The pages, served by this process so that one command is enough. Register them before main().
_EXAMPLES = Path(__file__).resolve().parent
_serve_page("/avatar", _EXAMPLES / "client")
_serve_page("/probe", _EXAMPLES / "probe")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Run one session of the bot on ``transport`` until the client leaves.

    Args:
        transport: The transport the runner created for this session.
        runner_args: The runner's arguments for this session.
    """
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=_require_env("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=_require_env("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID") or DEFAULT_VOICE_ID
        ),
    )
    llm = OpenAILLMService(
        api_key=_require_env("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(system_instruction=SYSTEM_INSTRUCTION),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # What makes the avatar: this processor, right after the TTS and before the transport's
    # output. The bot's audio still plays through the transport; the animation reaches the
    # browser as RTVI server messages, which the web SDK lines up with that audio.
    avatar = SpatialRealAvatarProcessor(
        api_key=_require_env("SPATIALREAL_API_KEY"),
        app_id=_require_env("SPATIALREAL_APP_ID"),
        avatar_id=_require_env("SPATIALREAL_AVATAR_ID"),
        console_endpoint_url=os.getenv("SPATIALREAL_CONSOLE_ENDPOINT", ""),
        ingress_endpoint_url=os.getenv("SPATIALREAL_INGRESS_ENDPOINT", ""),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            avatar,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        # An unusable STT, LLM or TTS ends the session; the avatar processor never pushes errors.
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {"role": "developer", "content": "Say hello and briefly introduce yourself."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point the Pipecat runner calls for every session.

    Args:
        runner_args: The runner's arguments for this session; they decide the transport.
    """
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    main()
