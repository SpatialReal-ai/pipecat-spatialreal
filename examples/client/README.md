# The example web page

`index.html` is the browser side of the example: `pipecat-client-js` connects to the bot and
carries the microphone, the bot's audio track and the bot's RTVI messages, and the SpatialReal web
SDK draws the avatar and plays the bot's voice in step with it.

Press Start and the page:

1. gets a session token from `GET /spatialreal-token` (see `../bot.py`; the API key stays on the
   server) and creates a Pipecat avatar session in the `#avatar` element;
2. calls `session.start()` inside the click, which starts playback and opens no connection;
3. starts the bot with `POST /start` and connects with `client.connect()`.

The two pieces that make it work are in the client's callbacks:

```js
callbacks: {
  onTrackStarted: (track, participant) => {
    if (!participant?.local && track.kind === 'audio') session.setBotAudioTrack(track)
  },
  onServerMessage: (data) => session.handleServerMessage(data), // false when it is not the avatar's
}
```

The page logs the session's `turn` events (a bot turn starting, ending, or cut off by an
interruption) and `degraded` events (the avatar fell back to plain audio, with the reason).

**Never play the bot's track yourself.** Do not attach it to an `<audio>` element and do not use
`PipecatClientAudio`: the SDK plays it, and a second player doubles the voice and breaks the lip
sync. The page creates its audio context inside the click, before any request, because some
browsers unlock audio only there, and closes it when the run ends: the SDK plays through a context
it is handed but never closes it, and browsers cap how many stay open.

## Get the web SDK's Pipecat session

The session kind is on the `feat/pipecat-session` branch of `SpatialReal-ai/web-sdk` and not on npm
yet. The page imports it from `vendor/spatialreal-web-sdk.js`, which git ignores. With that branch
checked out in `$WEB_SDK`, from the root of this repository:

```bash
mkdir -p examples/client/vendor
cp -R "$WEB_SDK"/dist/. examples/client/vendor/
echo "export * from './index.js'" > examples/client/vendor/spatialreal-web-sdk.js
```

(The SDK's build puts its entry in `index.js`; the last line gives it the name the page imports.)
Once the SDK is published, install it from npm instead and import `@spatialreal/web-sdk` through
your bundler.

## Run it

With the bot running (`uv run python examples/bot.py -t daily`, see `../README.md`), open
<http://localhost:7860/avatar/>: the bot serves this page, so the page and the token route share an
origin. Or serve the page anywhere and pass the bot's origin: `index.html?runner=http://localhost:7860`.

For a bot started with `-t livekit`, open `/avatar/?transport=livekit`. The same session code works
on both transports.

Test with speakers as well as headphones: how well echo cancellation copes with audio the SDK plays
through Web Audio is not verified yet.

The page loads `pipecat-client-js` 1.16.0, `@pipecat-ai/daily-transport` 1.7.2 and
`@pipecat-ai/livekit-transport` 1.1.2 from esm.sh so that it needs no build; in your own app install
them from npm (`npm install @pipecat-ai/client-js @pipecat-ai/daily-transport`).
