# The probe: a silent receiver for a Pipecat room

The processor announces the audible onsets of each turn (`turn_start.onsets_ms`) and sends the
animation ahead of the audio. The web SDK has to find the same onsets in the audio it receives and
anchor the animation there. `receiver.html` checks that on a real Daily room: it joins as a
listener, reads the avatar envelope from the room's app messages, taps the bot's audio track, and
prints how the two line up. It is a tool for developing this package, not part of what a customer
ships.

The page never plays the bot. The track goes into a muted AudioWorklet, held by a media element at
volume 0 (Chrome feeds a remote track to Web Audio only while an element holds it), and the graph
ends in a gain of 0. If the room later hands over another audio track for the bot, the first graph
is released and the new track tapped in its place, with a line on the timeline; `capture.wav`
carries on across the swap.

Onsets are found with the processor's own rule, `threshold` and `min_silence_ms`, on the same PCM16
quantisation as the web SDK.

## Run it

```bash
uv run python examples/bot.py -t daily
```

Open <http://localhost:7860/probe/receiver.html?mic=1> and press Start. The page asks the runner
for a room (`POST /start`), joins it and taps the bot. `mic=1` publishes your microphone, so you
talk to the bot from this tab. Because the page is silent, read the bot's side in the terminal; to
hear it, join the room (the status line shows its URL) from another device with headphones. The
bot ends its session when any participant leaves. Keep the probe tab in the foreground, and press
**Leave** when you are done: the capture stays on the page until you close it.

To join a room that already exists, pass it: `receiver.html?room=<room url>&token=<token>`. A room
and token come from `POST /start` with `{"transport": "daily", "createDailyRoom": true}`; they are
the `dailyRoom` and `dailyToken` of the reply.

| URL parameter | Meaning | Default |
|---|---|---|
| `room`, `token` | Join this Daily room instead of starting a session. | start one |
| `runner` | Origin of the runner that answers `POST /start`. | this origin |
| `mic` | `1` publishes the microphone. | off |
| `bot` | Part of the bot participant's name; only its audio is tapped. | `bot` (the runner names it "Pipecat Bot") |
| `threshold` | Largest PCM16 sample that still counts as silence. Must equal the processor's `threshold`. | 328 |
| `min_silence_ms` | Shortest silence that makes the next sound a new onset. Must equal the processor's `min_silence_ms`. | 150 |

## What it prints

One row per turn, derived again from the raw logs after every event, so a message that arrives
late only changes its own row.

| Column | Meaning | Passes when |
|---|---|---|
| `turn` | First 8 characters of `turn_id`. | |
| `lead ms` | The detected audible onset minus the arrival of `turn_start`. A `~` marks a chained turn with no onset of its own, placed from the previous turn (see below). | at least 100 |
| `segment 0` | Chunks arrived of `chunk_count`, the segment's bytes once joined in `chunk_index` order, and how long before the onset the last chunk arrived. | whole, and before the onset |
| `onsets_ms[0]` | The processor's first onset for the turn. | |
| `skew ms` | `lead` minus `onsets_ms[0]`: how much slower the audio path is than the message path. | steady from turn to turn |
| `onset Δ ms` | Tapped minus announced spacing: each later onset against the first, and for a chained turn (`prev_turn_id`) the first onset against where the previous turn's exact length puts it. | within 40 |
| `end` | `turn_end` (or `turn_abort`) time since `turn_start`, `pcm_ms` and the segment count. | |
| `abort Δ ms` | `turn_abort` minus Pipecat's `user-started-speaking` message. | at most 300 |
| `result` | `pass`, `fail` with the reasons, or `pending`. | |

Above the table a line counts the turns that pass or fail, the `state` messages that say
`STATUS_DEGRADED`, the contract problems and the audible onsets no turn accounts for. Below it,
**Contract problems** lists every message that breaks `pipecat/v1/avatar.proto`: a missing key,
1-based or out-of-range `chunk_index`, `chunk_count` below 1, `chunk_count` or `end` changing
inside a segment, a gap in `segment_seq`, `end` anywhere but the last segment, frames after
`turn_end` or `turn_abort`, an empty `turn_onsets`, a message over Daily's 4 KB limit, and the
like. **Timeline** puts every message, tapped onset and conversation marker in order.

### What the receiver cannot see

The first sample of a turn is not marked in the room's audio: between turns the track is silence,
and a turn that opens with silence of its own (`onsets_ms[0]` > 0) starts somewhere in it. So the
probe cannot subtract `onsets_ms[0]` from a measured position and compare. It compares what does
not depend on where the turn starts: the spacing of onsets inside a turn, and the first onset of a
chained turn, which the previous turn fixes. A turn with a single onset is checked on `lead`,
`segment 0` and `skew` only; `skew` is `lead` minus `onsets_ms[0]`, so an onset found early or late
shows as skew that moves between turns.

**A chained turn can have no onset to hear.** A turn with `prev_turn_id` follows the previous one
without a pause longer than `min_silence_ms`. The processor starts a fresh onset tracker for every
turn and counts a turn's leading silence from the turn's own first sample, so it always announces
`onsets_ms[0]`. The receiver's detector runs on across the join and needs `min_silence_ms` (150 ms)
of quiet before it finds a sound: when the end of the previous turn, any gap and the new turn's own
leading silence add up to less, there is nothing to find, and nothing is wrong. The probe does not
fail such a turn for it. It places the turn's first onset where the previous turn's exact length
puts it, which is what the web SDK does, marks `lead` with `~`, and checks what that position
allows: `lead`, segment 0 and the spacing of later onsets. It cannot confirm that the audio is
there, because there is no onset to confirm it with. When the join is quiet enough for an onset to
be found, that onset is compared with the chain's position instead (`onset Δ` shows `chain +N`,
within 40 ms). A chained turn whose predecessor was not measured cannot be placed, so a missing
onset is a failure there.

`abort Δ` counts from Pipecat's `user-started-speaking` message, which the bot sends once its VAD
confirms speech (`start_secs`, 0.2 s by default). It times the processor's reaction, not the sound
of the user's first syllable.

## The capture

**download capture.jsonl** saves one JSON object per line, one per avatar message, in arrival
order:

```json
{"t": 1730580000123, "tap": 481920, "data": {"tag": "spatialreal-avatar", "turn_start": {"turn_id": "..."}}}
```

* `t` is `Date.now()` when the message arrived.
* `tap` is how many samples of `capture.wav` had arrived by then, so a replay can put the message
  at `tap / sample rate` seconds into the audio.
* `data` is the `AvatarMessage`: the argument `onServerMessage` gets in `pipecat-client-js`, and
  what `session.handleServerMessage()` takes. Other RTVI messages are not recorded.

**download capture.wav** saves what the tap received: the bot's decoded track, mono PCM16, at the
rate of the browser's AudioContext (its header says which; 48 kHz on most machines), from the moment
the page started tapping.

## The live run

Needs the SpatialReal, Deepgram, Cartesia, OpenAI and Daily keys in `.env`.

1. `uv run python examples/bot.py -t daily`, then open the probe with `?mic=1` and press Start.
2. Talk to the bot for ten turns. Interrupt it mid-sentence once.
3. Expect every turn to pass, and the summary line to end in `state degraded: 0. contract problems:
   0. stray onsets: 0.` If the onsets disagree by more than 40 ms, adjust `threshold` (and
   `min_silence_ms`) in the processor and in the URL until ten turns pass, make that the constructor
   default, and record it in the README.
4. Download both capture files, move them into this directory, and commit them: they are the offline
   test material for the web SDK's Pipecat session.

LiveKit has no probe: run the bot with `-t livekit` and use the example page
(`/avatar/?transport=livekit`, see [../client/README.md](../client/README.md)).
