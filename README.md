# pipecat-spatialreal

Give a [Pipecat](https://github.com/pipecat-ai/pipecat) voice bot a real-time
[SpatialReal](https://www.spatialreal.ai/) 3D avatar. The avatar is rendered in the browser by the
SpatialReal web SDK and moves in sync with the bot's voice, so your pipeline never carries video.

Built and maintained by [SpatialReal](https://www.spatialreal.ai/).

[Demo video](https://youtube.com/shorts/4gmsrmBwV-s): a conversation with an interruption.

Tested with Pipecat v1.15.0, the SpatialReal Python SDK 0.3.0 and web SDK 1.0.0-beta.6, on Daily and
LiveKit.

## How it works

* `SpatialRealAvatarProcessor` sits right after your TTS service. It streams the bot's speech to
  SpatialReal, which returns facial animation.
* The animation goes to the browser as RTVI server messages. The bot's audio keeps playing through
  your transport, released in step with the animation.
* In the browser, the SpatialReal web SDK takes the bot's audio track and those messages, draws the
  avatar and plays the voice in sync with it.
* When the user interrupts, the avatar stops with the voice. If SpatialReal cannot be reached, the
  bot keeps talking without animation and the processor reconnects.

## Installation

```bash
uv add pipecat-spatialreal
# or
pip install pipecat-spatialreal
```

Python 3.11 or newer, Pipecat 1.15 or newer (below 2).

You need a SpatialReal API key, app ID and avatar ID: create an app and pick an avatar in
[SpatialReal Studio](https://app.spatialreal.ai).

## Usage

Put the processor right after the TTS service, before `transport.output()`:

```python
import os

from pipecat.pipeline.pipeline import Pipeline

from pipecat_spatialreal import SpatialRealAvatarProcessor

avatar = SpatialRealAvatarProcessor(
    api_key=os.environ["SPATIALREAL_API_KEY"],
    app_id=os.environ["SPATIALREAL_APP_ID"],
    avatar_id=os.environ["SPATIALREAL_AVATAR_ID"],
)

pipeline = Pipeline(
    [
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        avatar,  # right after the TTS, before the transport's output
        transport.output(),
        assistant_aggregator,
    ]
)
```

* **RTVI must be enabled.** It is Pipecat's default for `PipelineWorker`; the animation reaches the
  browser as RTVI server messages.
* **Sample rate.** The pipeline's `audio_out_sample_rate` must be one SpatialReal accepts: 8, 16,
  22.05, 24, 32, 44.1 or 48 kHz.
* **Keep the API key on the server.** The browser needs a short-lived session token instead, which
  your backend mints with the key. The example bot has a route that does it.

### Parameters

All keyword arguments. The first three are required.

| Parameter | Default | Meaning |
|---|---|---|
| `api_key` | | SpatialReal API key. |
| `app_id` | | SpatialReal app ID. |
| `avatar_id` | | The avatar to animate. The browser must show the same avatar. |
| `environment` | `"us-west"` | The SpatialReal deployment. Most users can leave it. |
| `console_endpoint_url` | `""` | Override the console API URL (session tokens). Empty uses `environment`. |
| `ingress_endpoint_url` | `""` | Override the animation service URL. Empty uses `environment`. |
| `lead_ms` | `200` | How far ahead of real time the audio is released to the transport. |
| `hold_cap_s` | `1.5` | Longest wait for a turn's first animation; after it, the turn plays without animation. |
| `output_chunk_ms` | `40` | Size of the audio chunks sent downstream. Must match your output transport's chunk size (`audio_out_10ms_chunks` × 10 ms). |
| `silence_chunk_ms` | `40` | Size of the silence chunks played while the TTS is behind. A multiple of `output_chunk_ms`. |
| `wire_timeout_s` | `2.0` | How long a write to SpatialReal may take before the connection counts as lost. |
| `reconnect_backoff_s` | `(1, 2, 4, 8)` | Seconds between reconnection attempts; the last value repeats. |
| `threshold`, `min_silence_ms` | `328`, `150` | Fixed by the web SDK. Leave them at the defaults. |
| `**kwargs` | | Passed to Pipecat's `FrameProcessor`. |

The constructor raises `ValueError` for an empty credential or a setting out of range. It connects
when the pipeline starts.

## The browser side

Install the SpatialReal web SDK (1.0.0-beta.6 or newer has the Pipecat session):

```bash
npm install @spatialreal/web-sdk@^1.0.0-beta.6
```

Create a Pipecat avatar session, then hand it the bot's audio track and every RTVI server message
from your `PipecatClient`:

```js
import { SpatialReal } from '@spatialreal/web-sdk'
import { PipecatClient } from '@pipecat-ai/client-js'

const session = await new SpatialReal({ appId }).createSession({
  avatarId, credential: sessionToken, container, pipecat: {},
})
await session.start()

const client = new PipecatClient({
  transport,
  enableMic: true,
  callbacks: {
    onTrackStarted: (track, participant) => {
      if (track.kind === 'audio' && isBot(participant)) session.setBotAudioTrack(track)
    },
    onServerMessage: (data) => session.handleServerMessage(data),
  },
})
```

**Do not play the bot's track yourself** (no `<audio>` element, no `PipecatClientAudio`): the SDK
plays it, and a second player doubles the voice and breaks the lip sync. Hand it only the bot's
track, not another participant's.

[`examples/client`](https://github.com/SpatialReal-ai/pipecat-spatialreal/tree/main/examples/client) is a complete page.

## Run the example

`examples/bot.py` is a complete voice bot (Deepgram, Gemini, Cartesia) with the avatar, run by
Pipecat's development runner on Daily or LiveKit:

```bash
uv sync --extra example
cp env.example .env        # fill in your keys
uv run python examples/bot.py -t daily
```

Then open <http://localhost:7860/avatar/> and press Start.
[`examples/README.md`](https://github.com/SpatialReal-ai/pipecat-spatialreal/blob/main/examples/README.md) has the LiveKit variant and every setting.

## Limitations

* **Transports.** Built for Daily and LiveKit. SmallWebRTC has not been tried.
* **TTS bots only.** Speech-to-speech services that produce audio without TTS frames are not
  supported yet.
* **The bot's track must carry only the voice.** Background audio mixed into it is not supported.
* **Stalls.** If the bot's host or the network stalls, the avatar can be briefly out of step with
  the voice and catches up within about 0.3 s after the audio resumes.
* **Transcripts run slightly ahead.** Word-timestamped text runs up to `lead_ms` ahead of the speech.
* **One connection per bot session.** Billing is per connection, not per viewer.
* **Plan session limits.** On a plan with a maximum session duration, SpatialReal closes the
  connection when it is reached. The processor reconnects at once; the turn being spoken plays on
  without animation.

## Development

```bash
uv sync --extra dev --extra example
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

The tests run against a fake SpatialReal service and need no network.
[CHANGELOG.md](https://github.com/SpatialReal-ai/pipecat-spatialreal/blob/main/CHANGELOG.md) lists what changed in each release.

## License

Apache License 2.0; see [LICENSE](https://github.com/SpatialReal-ai/pipecat-spatialreal/blob/main/LICENSE).
