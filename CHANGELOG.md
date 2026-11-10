# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

First release, not yet published (the date is set when it is). Tested with Pipecat v1.15.0, the
SpatialReal Python SDK 0.3.0 (0.2.2 and up are allowed) and web SDK 1.0.0-beta.6 against a fake
SpatialReal backend, and run live on Daily and LiveKit rooms, locally and on Pipecat Cloud,
between 2026-11-03 and 2026-11-09.

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
- A release workflow: publishing a GitHub Release builds the package, checks the tag against the
  built version, smoke-tests the wheel in a fresh environment and publishes to PyPI through
  trusted publishing.
- CI: every job has a timeout and a hung test ends after 120 s; a second job tests the lowest
  versions `pyproject.toml` declares; a smoke test sends the avatar messages through RTVI and
  Pipecat's eval transport. The weekly compatibility job runs read-only and a separate job opens
  its README pull request; the codegen workflow keeps the App's key away from third-party code.
- The audio envelope (`pipecat_spatialreal.fingerprint.Envelope`): the log-RMS of the audio the
  processor releases, over 10 ms windows every 5 ms, one byte per value (a keyframe is every eighth
  value). It covers every sample the room plays for a turn, stall silence and the padding of the
  last chunk included, and needs no numpy. The browser aligns the animation to the audio by
  correlating the track it receives against it, instead of anchoring on a loudness onset; the onsets
  stay in `turn_start` and `turn_onsets` as a prior only.
- `turn_envelope` messages (`pipecat/v1/avatar.proto`): the envelope in chunks of at most 2 400
  bytes, sent ahead of the audio they describe. `envelope.turn_envelope` builds them. While the TTS
  is behind, each chunk of stall silence is described by a message of its own (up to 25 a second at
  the default `silence_chunk_ms`).
- `gap_ms` in `turn_start`: the silence the processor itself released between the previous turn's
  last sample and this turn's first, so that the browser can join the two turns' envelopes. It is 0
  for a chained turn, the first turn, after an interruption and when the room ran dry.
- The inter-turn fill: after a turn's last sample the processor keeps the room fed with silence for
  up to one second, so that the gap is exact. It is plain audio, not bot speech, and is not sent to
  SpatialReal.
- The probe page (`examples/probe`) runs the alignment live on the tapped track, prints its
  alignment columns (`first lock ms`, `offset ms`, `p95 err ms`, `held / acc` and the four checks)
  and saves paired captures: the bot's track, the server's envelope and every message. Two clean
  captures, one on Daily and one on LiveKit, and a Daily run with spliced turns (chained, and with
  a gap) are fixtures in `tests/fixtures/sync/`.
- `hack/sync-sim/`: the simulation behind the alignment rule and its thresholds (`dsp.py` is the
  normative definition the web SDK ports), and `replay_capture.py`, which replays the rule over a
  paired capture and prints the design's validation minimum.
- `tests/test_sync_replay.py` replays the rule over the committed paired captures and asserts the
  minimum: no accepted alignment more than 20 ms off, every turn locked within 750 ms of speech, the
  onset prior within 10 ms of the truth before the lock on at least 95 % of turns, and a p95 error
  of at most 5 ms, all against the received content (400 ms windows aligned on their own); on the
  splice run, every join holds its `gap_ms` within one hop.

### Details

- The sdist leaves out `hack/` and the paired captures, so it stays at about 4 MB; the wheel is
  unchanged. The tests that need them skip without them.
- The probe page places a turn that follows silence the processor released (`gap_ms` > 0, the
  turn before it ended and was not aborted) at the previous turn's place plus its `pcm_ms` plus
  `gap_ms` until its first lock, as the web SDK does (an onset prior still takes over); it used the
  arrival of `turn_start` plus 200 ms, 78-182 ms off on the splice capture's gap turns that locked
  before their onset. `replay_capture.py` reports both priors and replays with the new one on
  `--r72-prior`.
- The envelope is computed as a turn's chunks join its hold, not when its release starts: a long run
  held whole (tens of ms of work: 12 s at 48 kHz measured 20-41 ms) no longer delays the turn's
  first audio after the decision to start it. The `turn_envelope` messages are the same bytes, sent
  at the same points.
- An interruption that aborts no announced turn (only the fill was playing) now also moves the
  processor's playout clock back to the moment of the interruption, as the transport drops the
  fill it held (up to 80 ms): the next turn gets its full lead and its `turn_end` goes when its
  audio has played. On any interruption the clock is never moved back past uninterruptible audio
  already released, which the transport keeps.
