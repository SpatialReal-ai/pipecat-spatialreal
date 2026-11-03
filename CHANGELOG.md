# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-03

First release. Tested with Pipecat v1.15.0 against a fake SpatialReal backend; the first live run
on Daily and LiveKit rooms is still to be done.

### Added

- `SpatialRealAvatarProcessor`: a Pipecat processor for the position after the TTS service. It
  streams the bot's speech to SpatialReal for facial animation, sends the animation to the browser
  as RTVI server messages, and releases the audio in step with it. It bridges a TTS stall with
  silence that inference sees too, aborts the turn on an interruption, and keeps the bot's audio
  playing, unanimated, when SpatialReal cannot be reached.
- The avatar envelope: builders for the `turn_start`, `turn_onsets`, `frames`, `turn_end`,
  `turn_abort` and `state` messages of `pipecat/v1/avatar.proto`, and the generated protobuf module
  they use.
- `OnsetTracker`: finds the audible onsets of a turn's audio by the rule the web SDK applies to the
  same audio.
- `DrivenIngress`: an asyncio wrapper around one SpatialReal inference session, which reports the
  animation segments and the loss of the session.
- An example bot (`examples/bot.py`) for Pipecat's development runner on Daily and LiveKit, with a
  route that gives the browser a session token and keeps the API key on the server.
- An example web page (`examples/client`) that shows the avatar with the web SDK's Pipecat session.
- A probe page (`examples/probe`) that joins a Daily room silently and measures how the avatar
  messages and the bot's audio line up.
- `wire_timeout_s` (2.0 s): a write to SpatialReal that does not return in time is a lost link.
  The audio plays on unanimated, the browser is told once, the socket is aborted and the processor
  reconnects. Nothing that handles frames or plays audio waits for the socket; every write goes
  through one writer task, in order. A session the server ends at the plan's ceiling (close code
  4002) is reconnected at once.
- `output_chunk_ms` (40 ms, which must equal the transport's `audio_out_10ms_chunks` × 10): audio
  is released in whole output chunks and every turn ends on a chunk boundary, padded with silence
  that inference sees and that counts into `pcm_ms`, so Pipecat's output transport pads and
  discards nothing and two turns back to back reach the room exactly as inference saw them. A
  turn that follows a turn without animation is not chained to it.
- `ingress_endpoint_url` is the full `https://<host>/v2/driveningress` URL; a bare origin is
  accepted and given that path.
- The example's token route answers its own origin only (403 for a request another site made),
  gets its token from the console the processor uses, and the example page hands the web SDK only
  the bot's audio track, can be stopped while the bot starts, and serves the probe at `/probe/`.
  The example bot uses Gemini (`GEMINI_API_KEY`) as its language model.
- CI: every job has a timeout and a hung test ends after 120 s; a second job tests the lowest
  versions `pyproject.toml` declares; a smoke test sends the avatar messages through RTVI and
  Pipecat's eval transport. The weekly compatibility job runs read-only and a separate job opens
  its README pull request; the codegen workflow keeps the App's key away from third-party code.
