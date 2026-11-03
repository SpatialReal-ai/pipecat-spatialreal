# Example: a Pipecat voice bot with a SpatialReal avatar

Written for Pipecat v1.15.0.

| Path | What it is |
|---|---|
| `bot.py` | The bot: Pipecat's `06-voice-agent` shape (Deepgram, Gemini, Cartesia) with `SpatialRealAvatarProcessor` after the TTS. One file, run by Pipecat's development runner on Daily or LiveKit. |
| `client/` | The web page: `pipecat-client-js` plus the SpatialReal web SDK. See [client/README.md](client/README.md). |
| `probe/` | A silent receiver that measures how the envelope and the bot's audio line up in a Daily room. A developer tool for this package. See [probe/README.md](probe/README.md). |

## Run it

You need keys for SpatialReal, Deepgram, Cartesia and Gemini (Google AI Studio), and for Daily or LiveKit.

```bash
uv sync --extra example
cp env.example .env        # fill it in: env.example lists every variable
uv run python examples/bot.py -t daily
```

Then open <http://localhost:7860/avatar/> and press Start. The page needs the web SDK's Pipecat
session, which is not on npm yet: copy it in first as [client/README.md](client/README.md) says.

`-t livekit` runs the same bot in a LiveKit room (set `LIVEKIT_URL`, `LIVEKIT_API_KEY` and
`LIVEKIT_API_SECRET`); open `http://localhost:7860/avatar/?transport=livekit`. Without `-t` the
runner offers every transport it has dependencies for. `transport_params` in `bot.py` also has the
`eval` entry that Pipecat's eval harness (which connects to the runner's `/ws`) looks for.

`uv run python examples/bot.py --help` lists the runner's options and needs no keys.

### Settings

All of them come from the environment or `.env`.

| Variable | Used for |
|---|---|
| `SPATIALREAL_API_KEY`, `SPATIALREAL_APP_ID`, `SPATIALREAL_AVATAR_ID` | The avatar. The API key stays on the server. |
| `SPATIALREAL_CONSOLE_ENDPOINT`, `SPATIALREAL_INGRESS_ENDPOINT` | Optional, set together: another SpatialReal deployment than production (`us-west`). The console endpoint is the console API root, such as `https://api.spatialreal.cloud` (a value ending in `/v1/console` is read the same way). The ingress endpoint is the full driveningress URL with its path, such as `https://driven.us-west.spatialreal.cloud/v2/driveningress`; a bare origin is accepted too. |
| `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `GEMINI_API_KEY` | Speech to text, text to speech and the language model (Gemini). `CARTESIA_VOICE_ID` picks another voice. |
| `PIPECAT_ALLOWED_ORIGINS` | The web origins allowed to read the runner's replies; `env.example` allows only the bot's own, `http://localhost:7860` (put yours there if you change the host or port). Read by Pipecat's runner, see below. |
| `DAILY_API_KEY` | `-t daily`: the runner creates a room per session. `DAILY_ROOM_URL` reuses one room instead. |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | `-t livekit`. |

## What `bot.py` adds to the runner

Pipecat's runner serves `POST /start`, which starts a bot session and returns the room the browser
joins. For Daily the request must ask for a room, `{"transport": "daily", "createDailyRoom":
true}`, unless `DAILY_ROOM_URL` is set; the example pages do. `bot.py` adds:

* `GET /spatialreal-token` returns `{"session_token", "avatar_id", "app_id", "endpoints"}`. The web
  SDK needs the session token to download the avatar. The route mints it with the API key, the way
  the `spatialreal` Python SDK does (`POST {console}/v1/auth/session-token`, header `X-Api-Key`, a
  token valid for one hour), from the console the processor uses, so the key never reaches the
  browser. `endpoints` is what the page passes to `new SpatialReal()`: empty for production, and
  `{"api": <console root>}` when `SPATIALREAL_CONSOLE_ENDPOINT` points the bot at another
  deployment, so that the page loads the avatar from the deployment that issued the token.
* `/avatar/` and `/probe/`: the two pages above, so that one process is enough.

The token route has no authentication, and it is same-origin only: it answers the pages this
server serves (and callers that are not web pages, such as `curl`) and refuses, with 403, a request
that another site or another origin made (`Sec-Fetch-Site: cross-site`, or an `Origin` that is not
the server's own). Without that, any page open in your browser could take session tokens billed to
you: the runner lets every web origin read its routes unless `PIPECAT_ALLOWED_ORIGINS` narrows it,
which `env.example` does. So open the pages from the bot's server; a copy served from elsewhere
gets no token. Your own app's token route needs a login, and the same origin rule.

The avatar needs RTVI, which Pipecat's `PipelineWorker` enables by default: the animation reaches
the browser as RTVI server messages, and the bot's audio keeps playing through the transport.

To measure how the avatar lines up with the bot's audio on a Daily room, use the probe, a developer
tool for this package: [probe/README.md](probe/README.md) says how.
