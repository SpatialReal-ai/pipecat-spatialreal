# The example web page

`index.html` is the browser side of the example: `pipecat-client-js` connects to the bot and
carries the microphone, the bot's audio track and the bot's RTVI messages, and the SpatialReal web
SDK draws the avatar and plays the bot's voice in step with it.

Press Start and the page:

1. gets a session token from `GET /spatialreal-token` (see `../bot.py`; the API key stays on the
   server) and creates a Pipecat avatar session in the `#avatar` element, with `new SpatialReal({
   appId, endpoints })`: the route's `endpoints` is empty for production and names the bot's
   SpatialReal deployment otherwise, so the avatar comes from the deployment that issued the token;
2. calls `session.start()`, which starts playback and opens no connection;
3. starts the bot with `POST /start` and connects with `client.connect()`, which returns once the
   bot is ready.

Stop works from the moment Start is pressed: it cancels what is still on its way and lets go of
what the run has made. Start gives up, and says so, when a request is not answered within 15 s or
the bot is not ready within 30 s: a bot that fails while it starts (a missing key, say) never gets
ready, and the bot's terminal says why.

The two pieces that make it work are in the client's callbacks:

```js
callbacks: {
  onTrackStarted: (track, participant) => {
    if (track.kind === 'audio' && !participant?.local && isBot(participant)) session.setBotAudioTrack(track)
  },
  onServerMessage: (data) => session.handleServerMessage(data), // false when it is not the avatar's
}
```

Hand the SDK the bot's audio track and no other: given another participant's track, it plays that
in place of the bot's voice and anchors the face to it. `isBot` tells the bot by the name Pipecat's
development runner gives it, `Pipecat Bot` on Daily, and on LiveKit by its identity,
`Pipecat Agent-` and the first 8 characters of the `sessionId` that `POST /start` returns. The page
expects one bot per room; your app tells its own bot by whatever name or identity it joins with.

The page logs the session's `turn` events (a bot turn starting, ending, or cut off by an
interruption) and `degraded` events (the avatar fell back to plain audio, with the reason).

**Never play the bot's track yourself.** Do not attach it to an `<audio>` element and do not use
`PipecatClientAudio`: the SDK plays it, and a second player doubles the voice and breaks the lip
sync. The page creates its audio context inside the click, before any request, because some
browsers unlock audio only there, and closes it when the run ends: the SDK plays through a context
it is handed but never closes it, and browsers cap how many stay open.

## The web SDK

The page imports the SpatialReal web SDK, `@spatialreal/web-sdk` 1.0.0-beta.6 (the first release
with the Pipecat session), from the jsDelivr CDN, so it needs no build step. In an app, install it
and import it through your bundler:

```bash
npm install @spatialreal/web-sdk@^1.0.0-beta.6
```

## Run it

With the bot running (`uv run python examples/bot.py -t daily`, see `../README.md`), open
<http://localhost:7860/avatar/>. The bot serves this page so that the page and the token route share
an origin: the route answers no other origin (see `../README.md`), so a copy of the page served
from elsewhere gets no token.

For a bot started with `-t livekit`, open `/avatar/?transport=livekit`. The same session code works
on both transports.

Test with speakers as well as headphones: how well echo cancellation copes with audio the SDK plays
through Web Audio is not verified yet.

The page loads `pipecat-client-js` 1.16.0, `@pipecat-ai/daily-transport` 1.7.2 and
`@pipecat-ai/livekit-transport` 1.1.2 from esm.sh so that it needs no build; in your own app install
them from npm (`npm install @pipecat-ai/client-js @pipecat-ai/daily-transport`).
