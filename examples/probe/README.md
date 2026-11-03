# The probe: a silent receiver for a Pipecat room

The processor announces the audible onsets of each turn (`turn_start.onsets_ms`) and sends the
animation ahead of the audio. The web SDK has to find the same onsets in the audio it receives and
anchor the animation there. The probe page, `index.html`, checks that on a real Daily room: it
joins as a listener, reads the avatar envelope from the room's app messages, taps the bot's audio
track, and prints how the two line up. It is a tool for developing this package, not part of what a
customer ships.

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

Open <http://localhost:7860/probe/?mic=1> and press Start. The page asks the runner
for a room (`POST /start`), joins it and taps the bot. `mic=1` publishes your microphone, so you
talk to the bot from this tab. Because the page is silent, read the bot's side in the terminal; to
hear it, join the room (the status line shows its URL) from another device with headphones. The
bot ends its session when any participant leaves. Keep the probe tab in the foreground, and press
**Leave** when you are done: the capture stays on the page until you close it.

To join a room that already exists, pass it: `/probe/?room=<room url>&token=<token>`. A room and
token come from `POST /start` with `{"transport": "daily", "createDailyRoom": true}`; they are the
`dailyRoom` and `dailyToken` of the reply.

| URL parameter | Meaning | Default |
|---|---|---|
| `room`, `token` | Join this Daily room instead of starting a session. | start one |
| `runner` | Origin of the runner that answers `POST /start`, for a copy of the page served elsewhere. The runner answers it only when its `PIPECAT_ALLOWED_ORIGINS` lists that page's origin (`env.example` lists only the bot's own). | this origin |
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
| `lead ms` | The detected audible onset minus the arrival of `turn_start`. A `~` marks a chained turn with no onset of its own, measured against the position the turn before it gives (see below). | at least 100 |
| `segment 0` | Chunks arrived of `chunk_count`, the segment's bytes once joined in `chunk_index` order, and how long before the onset the last chunk arrived (`~` again for a placed turn). | whole, and before the onset |
| `onsets_ms[0]` | The processor's first onset for the turn; `-` when it announced none. | |
| `skew ms` | `lead` minus `onsets_ms[0]`: how much slower the audio path is than the message path. `-` for a placed turn. | steady from turn to turn |
| `onset Δ ms` | Tapped minus announced spacing: each later onset against the first, and for a chained turn (`prev_turn_id`) the first onset against where the end of the turn before it puts it (`chain +N`). | within 40 |
| `end` | `turn_end` (or `turn_abort`) time since `turn_start`, `pcm_ms` and the segment count. | |
| `abort Δ ms` | `turn_abort` minus Pipecat's `user-started-speaking` message. Blind to a delay the two messages share (see *What `abort Δ` cannot see* below). | at most 300 |
| `result` | `pass`, `fail` with the reasons, `unverified` (a placed chained turn, see below) or `pending`. | |

Above the table a line counts the turns that pass, fail, are unverified or are pending, the `state`
messages that say `STATUS_DEGRADED`, the contract problems and the audible onsets no turn accounts
for; it is green only when every finished turn passed and nothing else is wrong. A run in which the
receiver missed a `turn_start` (it joined late, or a message was lost) therefore stays red to the
end: that turn shows `no turn_start`, its other messages are contract problems and its sound is a
stray onset. Reload the page and run again. Below the table, **Contract problems** lists every
message that breaks `pipecat/v1/avatar.proto`: a missing key, 1-based or out-of-range
`chunk_index`, `chunk_count` below 1, `chunk_count` or `end` changing inside a segment, a gap in
`segment_seq`, `end` anywhere but the last segment, frames after `turn_end` or `turn_abort`, an
empty `turn_onsets`, a message over Daily's 4 KB limit, and the like. **Timeline** puts every
message, tapped onset and conversation marker in order.

### What `lead` means

A chained turn is announced as soon as its animation is in hand, well ahead of its audio, so its
`lead` is large. An unchained turn is not: the processor sends its `turn_start` and first segments
in the step that releases the turn's first audio, never before, because a sound the room plays
between an earlier `turn_start` and the turn's own audio could be taken for the turn's onset. Its
`lead` is then only how far the message channel runs ahead of the audio channel: 120 to 160 ms
measured on a Daily room. That is not a margin the processor adds, and whether it always clears the
100 ms the table asks for is for the live run to confirm.

### What the receiver cannot see

The first sample of a turn is not marked in the room's audio: between turns the track is silence,
and a turn that opens with silence of its own (`onsets_ms[0]` > 0) starts somewhere in it. So the
probe cannot subtract `onsets_ms[0]` from a measured position and compare. It compares what does
not depend on where the turn starts: the spacing of onsets inside a turn and, for a chained turn,
its first onset against where the end of the turn before it puts it. A turn with a single onset is
checked on `lead`, `segment 0` and `skew` only; `skew` is `lead` minus `onsets_ms[0]`, so an onset
found early or late shows as skew that moves between turns. `onsets_ms` can also be empty: the
processor announces an onset only once it has found an audible sample, and a degraded turn may
announce none. A turn without announced onsets has nothing to compare, and `onsets_ms[0]` and
`skew` show `-`.

**An unchained turn always has an onset to hear.** The processor does not start a turn that has
animation and does not follow the previous one back to back until the room has been quiet for
`min_silence_ms`, so the receiver's detector finds that turn's first onset itself, and a missing one
is a failure. It holds such a turn only when it would start less than `min_silence_ms` (150 ms by
default) after the last sound the room was given: the turn then waits until that quiet has lasted
`min_silence_ms` plus one `silence_chunk_ms` as margin (150 + 40 ms), at most 190 ms of extra
silence. A turn that comes after a longer quiet starts as soon as it is ready. A degraded turn,
which plays without animation, is not held. An unchained turn's `turn_start` can arrive before the
previous turn's `turn_end`, which waits for that turn's last segment while the next turn needs only
the quiet; the probe pairs onsets by position and by time windows, not by message order. A sound
that belongs to no turn cannot be told from a turn's first onset: it is reported as a stray when no
turn is waiting for an onset, and otherwise taken for that turn's, whose own onset then shows up as
a stray. Mixing other audio into the bot's track is not supported.

**A chained turn can have no onset to hear.** A turn with `prev_turn_id` plays right after the
previous turn's last sample: no pause, nothing in between. (A turn that would start after any gap
is unchained.) The processor starts a fresh onset tracker for every turn and counts a turn's
leading silence from the turn's own first sample, while the receiver's detector runs on across the
join and needs `min_silence_ms` (150 ms) of quiet before it finds a sound. When the previous turn's
trailing quiet and the new turn's own leading silence add up to less, the receiver has no onset
there to find, and nothing is wrong. The probe does not fail such a
turn for it. It places the turn's first onset where the end of the previous turn puts it: the
previous turn's first onset minus its `onsets_ms[0]`, plus `prev_turn_pcm_ms`, plus the turn's own
`onsets_ms[0]`. That position is exact, with no slack, because there is no gap in between, and it
is the one the web SDK uses. The turn shows `~` on `lead` and on the `segment 0` margin (both are
measured against the placed position), an empty `skew`, and the spacing of its later onsets. Its
result is **unverified**, neither pass nor fail, because nothing in the audio confirms the
placement. When the join is quiet enough for the receiver to hear the turn's first onset, that
onset is compared with the chain's position instead (`onset Δ` shows `chain +N`, within 40 ms). A
heard onset within 200 ms (`CHAIN_TOL_MS`) of the chain's position is taken for the chained turn's
first onset, so that a miss shows as a chain error; one further away is a stray.

A chained turn that cannot be placed takes no onset: with no position to tell its onset by, the
time window alone could give it a later turn's. It cannot be placed when it announced no onsets,
when the turn it follows was never announced or announced none, or when that turn cannot be placed
either (it was not heard, or it is chained and has the same trouble). If it ended, it fails, and
the failure names which: `it announced no onsets`, `it follows aaaaaaaa, which was not heard`, and
the like. The onset such a turn has in the audio, if the receiver hears one, belongs to no turn as
far as the probe can tell, so it is treated like any other sound that belongs to no turn (see
above): a stray, or the first onset of a later unchained turn that is still waiting for one, whose
own onset then shows up as a stray. Read the chained turn's failure first.

`abort Δ` counts from Pipecat's `user-started-speaking` message, which the bot sends once its VAD
confirms speech (`start_secs`, 0.2 s by default). It times the processor's reaction, not the sound
of the user's first syllable.

### What `abort Δ` cannot see

`abort Δ` is the barge-in criterion (`turn_abort` within 300 ms of `user-started-speaking`), and it
cannot detect a delay that a burst of `frames` messages adds on the transport's send path. The
bot's output transport sends RTVI messages one at a time, awaiting each send, in the order they
were queued; the processor queues a segment's chunks all at once, and Pipecat queues
`user-started-speaking`, the interruption that stops the bot's audio, and then `turn_abort` behind
whatever is already waiting. A user who speaks while a segment's chunks are being sent therefore
delays both messages by the same burst: the bot's audio stops and `turn_abort` arrives that much
later, but the difference between the two messages stays small, and `abort Δ` passes whatever the
burst cost. What would show the delay is timing on the audio tracks themselves: the user's first
sound on the microphone track (`mic=1`) against the moment the bot's audio stops on the tapped track
and against the arrival of `turn_abort`. The probe measures none of these: that delay is not
measured.

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

Needs the SpatialReal, Deepgram, Cartesia, OpenAI and Daily keys in `.env`. It has not been done
yet: the values below are what the design expects, not measurements, and the first run decides
whether they hold.

1. `uv run python examples/bot.py -t daily`, then open the probe with `?mic=1` and press Start.
2. Talk to the bot for ten turns. Interrupt it mid-sentence once.
3. A good run shows, on every turn:
   * `turn_start` arriving at least 100 ms before the audible onset of its speech (for an unchained
     turn that is the message channel's lead over the audio channel, see above);
   * every chunk of the turn's first segment arrived before that onset;
   * the onsets the processor announces and the onsets found in the audio the browser receives
     within 40 ms of each other (the section above says what the probe can compare);
   * no `state` message with `STATUS_DEGRADED`;
   * after an interruption, `turn_abort` within 300 ms of Pipecat's `user-started-speaking` message,
     which arrives about 0.2 s (the VAD's `start_secs`) after the user's first sound. This cannot
     show a delay that both messages share, such as a burst of `frames` queued ahead of them on the
     transport (see *What `abort Δ` cannot see* above); that delay is not measured.

   The summary line should end in `state degraded: 0. contract problems: 0. stray onsets: 0.`
   The turns that count are the ones marked `pass`; an `unverified` turn is not one of the ten. If
   the onsets disagree by more than 40 ms, adjust `threshold` (and `min_silence_ms`) in the
   processor and in the URL until ten turns pass, make that the constructor default, and record it
   in the README.
4. Download both capture files, move them into this directory, and commit them: they are the offline
   test material for the web SDK's Pipecat session.

LiveKit has no probe: run the bot with `-t livekit` and use the example page
(`/avatar/?transport=livekit`, see [../client/README.md](../client/README.md)).
