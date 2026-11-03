# pipecat-spatialreal

A [Pipecat](https://github.com/pipecat-ai/pipecat) processor that gives a voice bot a
[SpatialReal](https://www.spatialreal.ai/) avatar. Put it after the TTS service: it streams the
bot's speech to SpatialReal for facial animation, sends the animation to the browser as RTVI server
messages, and releases the audio in step with it. The avatar is rendered in the browser by the
SpatialReal web SDK, so the pipeline never carries video frames.

Built and maintained by [SpatialReal](https://www.spatialreal.ai/).

Tested with Pipecat v1.15.0. That is the version the test suite and CI run against, with a fake
SpatialReal backend; the first live run on Daily and LiveKit rooms is still to be done. A weekly job
runs the tests against the newest Pipecat release the package allows (`pipecat-ai>=1.15.0,<2`) and
opens a pull request that moves this line when they pass on a newer one.

## How it works

* Each TTS context, from `TTSStartedFrame` to `TTSStoppedFrame`, is one turn: one request to
  SpatialReal's inference service and one turn of the avatar messages.
* The turn's audio goes to SpatialReal as it arrives, at TTS speed, and is held back. Once the first
  animation segment is back, the processor announces the turn to the browser (`turn_start`, then the
  keyframes, as RTVI server messages) and releases the audio downstream in real time, 200 ms ahead
  (`lead_ms`) so that the transport never runs dry.
* When the TTS falls behind playback, the processor plays silence and sends the same silence to
  SpatialReal, so that the audio inference saw and the audio in the room stay sample-identical.
* Pipecat's own frames keep their order: the bot speaks, is interrupted and has its context
  aggregated as before, at the delayed playout. An interruption ends the turn in the browser
  (`turn_abort`) and stops its inference.
* If SpatialReal cannot be reached, or answers too slowly (`hold_cap_s`), the audio still plays,
  unanimated. The browser is told (`STATUS_DEGRADED`) and the processor keeps reconnecting.
* The processor opens one SpatialReal connection for the whole bot session, when the pipeline
  starts. Billing is per connection, not per viewer.

## Installation

```bash
uv add pipecat-spatialreal
# or
pip install pipecat-spatialreal
```

Python 3.11 or newer. The package brings `pipecat-ai` (1.15 or newer, below 2), the `spatialreal`
Python SDK and protobuf.

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

The bot's audio keeps playing through the transport. The animation reaches the browser as RTVI
server messages, which the SpatialReal web SDK lines up with that audio, so RTVI must be enabled
(Pipecat's default). The audio goes downstream at the pipeline's `audio_out_sample_rate`, which must
be the transport's output rate too, and a rate SpatialReal accepts: 8, 16, 22.05, 24, 32, 44.1 or
48 kHz.

The API key stays on the server. The browser gets a short-lived session token from your own backend
(`examples/bot.py` has a route that mints one); see [the browser side](#the-browser-side).

### Parameters

All of them are keyword arguments. The first three are required and must not be empty.

| Parameter | Default | Meaning |
|---|---|---|
| `api_key` | | SpatialReal API key. |
| `app_id` | | SpatialReal app id. |
| `avatar_id` | | The avatar to animate. |
| `environment` | `"us-west"` | The SpatialReal deployment to use when the two URLs are not given. |
| `console_endpoint_url` | `""` | Origin of the console API that issues session tokens. Empty takes it from `environment`. |
| `ingress_endpoint_url` | `""` | Origin of SpatialReal's inference ingress. Empty takes it from `environment`. |
| `threshold` | `328` | Largest absolute sample value that still counts as silence when finding the audible onsets the browser anchors the animation to. The web SDK must use the same value. |
| `min_silence_ms` | `150` | Shortest silence, in ms, after which the next sound is a new onset. The web SDK must use the same value. A turn that does not follow the previous one back to back starts at least this long after the last audio before it, so that the web SDK can find its onset; one that would have started sooner waits until this long plus one `silence_chunk_ms` have passed. |
| `lead_ms` | `200` | How far ahead of real time the audio is released. |
| `hold_cap_s` | `1.5` | Longest time a turn's audio is held for its first animation; after it, the turn plays without animation. Also the longest wait, once a turn's audio has played, for the last segment of its animation. |
| `silence_chunk_ms` | `40` | Length of each chunk of silence played while the TTS is behind. |
| `reconnect_backoff_s` | `(1, 2, 4, 8)` | Seconds to wait before each new attempt to reach SpatialReal, in turn; the last value repeats. |
| `**kwargs` | | Passed to Pipecat's `FrameProcessor`. |

The constructor raises `ValueError` for an empty credential or a setting out of range. It connects
to nothing: the connection opens when the pipeline sets the processor up.

### Settings from the environment

The processor reads no environment variables: you pass the settings in. The example bot reads these
from the environment or a `.env` file (`env.example` lists them all):

| Variable | Used for |
|---|---|
| `SPATIALREAL_API_KEY`, `SPATIALREAL_APP_ID`, `SPATIALREAL_AVATAR_ID` | The avatar. |
| `SPATIALREAL_CONSOLE_ENDPOINT`, `SPATIALREAL_INGRESS_ENDPOINT` | Optional: another SpatialReal deployment than production (`us-west`). |
| `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `OPENAI_API_KEY` | The example's speech to text, text to speech and language model. `CARTESIA_VOICE_ID` picks another voice. |
| `DAILY_API_KEY` | `-t daily`. `DAILY_ROOM_URL` reuses one room instead of creating one per session. |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | `-t livekit`. |

## The browser side

The browser joins the bot's room with `pipecat-client-js`, creates a SpatialReal avatar session of
the Pipecat kind, and hands it the bot's audio track and the bot's RTVI server messages. The session
draws the avatar and plays the bot's voice in step with it; the page must not play the bot's track
itself.

That session kind is in [web-sdk#28](https://github.com/SpatialReal-ai/web-sdk/pull/28) and is not
released on npm yet. [`examples/client`](https://github.com/SpatialReal-ai/pipecat-spatialreal/tree/main/examples/client)
is a page that uses it, and its README says how to wire the session in until the release.

The messages between the processor and the browser are defined in
[`pipecat/v1/avatar.proto`](https://github.com/SpatialReal-ai/shared-proto/blob/main/pipecat/v1/avatar.proto)
in the shared contract repository. The generated code in `src/pipecat_spatialreal/proto/` comes from
it.

## Run the example

`examples/bot.py` is a complete voice bot (Deepgram, OpenAI, Cartesia) with the avatar processor
after the TTS, run by Pipecat's development runner on Daily or LiveKit:

```bash
uv sync --extra example
cp env.example .env        # fill it in
uv run python examples/bot.py -t daily
```

Then open <http://localhost:7860/avatar/> and press Start. The page needs the web SDK's Pipecat
session, which is not on npm yet.
[`examples/README.md`](https://github.com/SpatialReal-ai/pipecat-spatialreal/blob/main/examples/README.md)
has the steps, the LiveKit variant and the settings, and describes the probe, a developer tool that
measures how the avatar lines up with the bot's audio in a Daily room.

## Limitations

* **RTVI must be enabled.** It is Pipecat's default for `PipelineWorker`, but not for a bridged
  worker. Without it the processor logs an error: the audio still plays, and no animation reaches
  the browser.
* **Transports.** Built for Daily and LiveKit rooms. SmallWebRTC has not been tried.
* **TTS bots only.** A turn is a TTS context and its audio is `TTSAudioRawFrame`. Speech-to-speech
  services that produce audio without TTS frames are not supported in this release.
* **No background mixer on the bot's output track.** The browser anchors the animation to the
  audible onsets it finds in the track, and the processor finds them in the voice alone. Other audio
  mixed into the bot's track is not supported.
* **A TTS that ends a context late.** A service that closes a context only when its 3 s idle timeout
  runs out makes the processor play up to 3 s of silence at the end of the turn (sent to inference
  too) and send `turn_end` late.
* **A sound between two turns.** A sound that is not TTS, played between two turns, can in rare
  timing be taken by the browser for the next turn's first onset. The animation re-anchors at the
  next pause in the speech.
* **Transcripts run ahead.** Held frames keep the presentation time the TTS stamped on them, which is
  already past when they are released. Word-timestamped text and the assistant aggregator therefore
  run up to `lead_ms` ahead of the speech they report.
* **One connection per bot session.** The processor opens one SpatialReal connection, however many
  viewers watch.
* **Python 3.11 or newer.**

## Development

```bash
uv sync --extra dev --extra example
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

The tests run against a fake SpatialReal backend (`tests/fake_driveningress.py`) and need no
network. The generated code is produced by the Protobuf Codegen workflow from the tagged contract;
do not edit it by hand. [CHANGELOG.md](https://github.com/SpatialReal-ai/pipecat-spatialreal/blob/main/CHANGELOG.md)
lists what changed in each release.

## License

Apache License 2.0; see [LICENSE](https://github.com/SpatialReal-ai/pipecat-spatialreal/blob/main/LICENSE).
