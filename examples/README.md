# Example: a Pipecat voice bot with a SpatialReal avatar

Tested with Pipecat v1.15.0.

| Path | What it is |
|---|---|
| `bot.py` | The bot: Pipecat's `06-voice-agent` shape (Deepgram, OpenAI, Cartesia) with `SpatialRealAvatarProcessor` after the TTS. One file, run by Pipecat's development runner on Daily or LiveKit. |
| `client/` | The web page: `pipecat-client-js` plus the SpatialReal web SDK. See [client/README.md](client/README.md). |
| `probe/` | A silent receiver that measures how the envelope and the bot's audio line up in a Daily room. A developer tool for this package. See [probe/README.md](probe/README.md). |

## Run it

You need keys for SpatialReal, Deepgram, Cartesia and OpenAI, and for Daily or LiveKit.

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
| `SPATIALREAL_CONSOLE_ENDPOINT`, `SPATIALREAL_INGRESS_ENDPOINT` | Optional: another SpatialReal deployment than production (`us-west`). |
| `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `OPENAI_API_KEY` | Speech to text, text to speech and the language model. `CARTESIA_VOICE_ID` picks another voice. |
| `DAILY_API_KEY` | `-t daily`: the runner creates a room per session. `DAILY_ROOM_URL` reuses one room instead. |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | `-t livekit`. |

## What `bot.py` adds to the runner

Pipecat's runner serves `POST /start`, which starts a bot session and returns the room the browser
joins. For Daily the request must ask for a room, `{"transport": "daily", "createDailyRoom":
true}`, unless `DAILY_ROOM_URL` is set; the example pages do. `bot.py` adds:

* `GET /spatialreal-token` returns `{"session_token", "avatar_id", "app_id"}`. The web SDK needs
  the session token to download the avatar. The route mints it with the API key, the way the
  `spatialreal` Python SDK does (`POST {console}/v1/auth/session-token`, header `X-Api-Key`, a token
  valid for one hour), so the key never reaches the browser. It has no authentication: your own app
  serves such a route behind its own login.
* `/avatar/` and `/probe/`: the two pages above, so that one process is enough.

The avatar needs RTVI, which Pipecat's `PipelineWorker` enables by default: the animation reaches
the browser as RTVI server messages, and the bot's audio keeps playing through the transport.

## What to expect from a good run

Measured with the [probe](probe/README.md) over ten consecutive turns on Daily:

* `turn_start` arrives at least 100 ms before the audible onset of its speech;
* every chunk of a turn's first segment has arrived before that onset;
* the onsets the processor announces and the onsets found in the audio the browser receives agree
  to within 40 ms (the probe's README says what it can compare);
* no `state` message with `STATUS_DEGRADED`;
* interrupting the bot mid-sentence gives `turn_abort` within 300 ms of the user starting to speak.

If the onsets differ by more than 40 ms on real speech (a soft attack or breath noise right at the
threshold), change `threshold` and, if needed, `min_silence_ms` until ten turns pass, then make
that the constructor default. Both ends must use the same rule: the probe takes the two values as
URL parameters, and the web SDK's detector must be changed with them.
