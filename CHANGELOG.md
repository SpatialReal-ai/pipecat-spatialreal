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
