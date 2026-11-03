# pipecat-spatialreal

A [Pipecat](https://github.com/pipecat-ai/pipecat) processor that gives a voice bot a SpatialReal
avatar. Placed after the TTS service, it streams the bot's speech to SpatialReal for facial
animation, forwards the animation to the browser as RTVI server messages, and releases the audio in
step with it. The avatar is rendered in the browser by the SpatialReal web SDK, so the pipeline
never carries video frames. This package is under development; usage documentation follows with
the first release.
