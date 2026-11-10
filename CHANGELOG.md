# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-11-09

### Changed

- A shorter README, written for users.
- The source distribution no longer includes development-only tools.

## [0.1.0] - 2026-11-09

First release. Tested with Pipecat v1.15.0, the SpatialReal Python SDK 0.3.0 and the SpatialReal
web SDK 1.0.0-beta.6, on Daily and LiveKit.

### Added

- `SpatialRealAvatarProcessor`: place it right after the TTS service. It streams the bot's speech
  to SpatialReal for facial animation, sends the animation to the browser as RTVI server messages,
  and releases the audio in step with it. The SpatialReal web SDK (1.0.0-beta.6 or newer) renders
  the avatar in the browser, so the pipeline carries no video.
- Audio and animation stay in sync: the web SDK lines the animation up with the audio the browser
  actually receives.
- Interruptions stop the avatar together with the voice. If SpatialReal cannot be reached, the bot
  keeps talking without animation and the processor reconnects.
- Works with Pipecat's Daily and LiveKit transports.
- An example bot (`examples/bot.py`) and web page (`examples/client`). The bot gives the browser a
  short-lived session token, so the API key stays on the server.
