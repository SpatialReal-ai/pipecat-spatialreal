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

The page also runs the envelope alignment on the tapped track, live, as the web SDK runs it: the
processor's `turn_envelope` (one byte per 5 ms of the audio it released) is correlated with the
envelope of the received track to find where each turn sits (design §3.2 of backend-ng's
`docs/superpowers/specs/2026-11-03-pipecat-sync-fingerprint-design.md`). The rule is in `sync.js`,
next to the page: a port of the normative oracle, `hack/sync-sim/dsp.py` (`envelopeBytes` and
`align` mirror `envelope_bytes` and `align`), held to the oracle's own numbers by
`tests/probe_check.mjs` (`tests/fixtures/sync/oracle.json`, written by `make_oracle_fixture.py`
next to it). One difference is deliberate: the oracle's 30 Hz high-pass on the received track is
zero phase over the whole signal, the page's is causal (first order), as a browser must run it; on
the fixture it moves the offset by 0.04 ms. The stall rule's tick-for-tick equivalence assumes the
envelope reaches the page ahead of the audio it describes (the processor's normal order): a chained
turn's prior can feed its predecessor's frames before the predecessor's own next tick does.

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
| `first lock ms` | Envelope alignment: how much of the turn's speech had been received at the first accepted tick, in ms, counted from where that lock puts `onsets_ms[0]` (the turn's first sample when it announced none). `none` while it has ticked without a lock. | see below |
| `offset ms` | The turn's first sample at the latest accepted tick minus the arrival of `turn_start`, on the tap's clock. It measures what `skew` measures with onsets: the audio path's delay over the message path's. | see below |
| `p95 err ms` | The 95th percentile, over the accepted ticks, of the distance between the accepted position and the *onset prior*: where the onset rule puts the turn's first sample (its first onset heard, or placed from the chain, minus `onsets_ms[0]`). `-` without an onset prior. | see below |
| `held / acc` | Ticks with speech in the window that were held (not accepted) and that were accepted. A `*` marks a reference spliced onto the turn before it. | see below |
| `peak / ratio / sharp / halves` | The four checks at the last tick with speech: correlation peak (≥ 0.95), peak over the second peak (≥ 1.2), sharpness (≥ 0.03) and the half windows' deviation in ms (≤ 10). A failed check is marked `!`, a peak at the edge of the search `(edge)`. | see below |
| `freezes` | The stall rule: how many times the turn's reference pointer froze, the time it stood still in all, and how many of those freezes a confirm closed at a new position (`1, 180 ms, 1 confirmed`); `N not` counts the rest (a full-window or newer-half lock closed them, the confirm found nothing in time, no confirm was possible, the 2 s cap, or the turn ended), `N open` a freeze in progress. `-` without one. | see below |
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
empty `turn_onsets`, a `turn_envelope` with a `hop_ms` or `window_ms` other than 5 and 10, an index
it covers twice, indices left uncovered at `turn_end` or one sent after `turn_end` or `turn_abort`,
a message over Daily's 4 KB limit, and the like. **Timeline** puts every
message, tapped onset and conversation marker in order.

### The alignment columns

The six alignment columns do not decide `pass` or `fail`: they are what the paired captures of the
sync-fingerprint plan's Task 7 (Chrome, Daily and LiveKit) record, and they become measurements
there, against the design's minimum (no false lock, a first lock within 300 ms of speech on 95 % of
turns, p95 error within 5 ms). The page reports them without judging them; the measurements are in
*Paired captures* below.

How the page runs the rule, per turn, from its `turn_start` on (design §3.2):

* **Received envelope.** The tapped track after a first-order 30 Hz high-pass, 10 ms windows every
  5 ms, dB of the mean square; only windows fully inside the audio count.
* **Reference.** The turn's `turn_envelope` values, placed by `first_index` (arrival order does not
  matter; a missing index is silence). It is spliced onto the previous turn's envelope as the web
  SDK splices it (design §3.3): a chained turn at `prev_turn_pcm_ms`; an unchained turn with
  `gap_ms > 0` (1000 included) at the previous turn's `turn_end.pcm_ms + gap_ms`, with silence in
  between; never on `gap_ms` 0 for an unchained turn, and never onto a turn that was aborted.
* **Cadence.** A tick every 50 ms of received audio until the turn's first lock, then every 200 ms;
  ticks stop once the last 1 s lies past the turn's end, or 1 s after `turn_abort`.
* **Prior.** Until the first lock: for a chained turn whose previous turn has locked, that turn's
  position (freezes and confirms included) plus `prev_turn_pcm_ms`, re-anchored on the onset prior
  only when that lies within 200 ms of it (rule 5, R74); else the onset prior when the onset rule
  has one; else, for an
  unchained turn with `gap_ms > 0` whose previous turn had ended (`turn_end`) and was not aborted
  when its `turn_start` arrived, the previous turn's place plus its `turn_end.pcm_ms` plus `gap_ms`,
  as for a chained turn (the processor released exactly that much silence; R72); else the arrival
  of `turn_start` plus 200 ms. After it, the turn's tracked position (the last accepted one, moved
  by the stall rule below). The search covers ± 300 ms around it. After 10 ticks with speech and no
  lock the search widens to ± 600 ms, re-centred on the onset prior (a chained turn's too: R76).
* **A tick** is `align` from `sync.js` on the last 3 s of received envelope: floor, speech gate,
  correlation of the last 1 s and its two halves, the four checks, parabolic refinement. A held
  tick changes nothing, unless the stall rule accepts the newer half (below).
* **The stall rule** (`hack/sync-sim/README.md`, "The stall rule"; `sync.js` `TurnTracker`, a port
  of `dsp.TurnTracker`). When the bot's host or the receiver stalls, Chrome plays a concealment fade
  and then exact zeros, and the rest of the turn arrives late by about the stall. After a turn's
  first lock, every received frame goes through its tracker before the tick: once 20 ms of frames
  are each at or below -60 dBFS while the turn's own reference at the pointer is above -50 dBFS,
  the pointer **freezes** (retroactively over those 20 ms) and stays put while the track stays
  silent, at most 2 s. Natural pauses and the processor's own silence are in the reference, so they
  never freeze it. When audio resumes, a small correlation **confirms** the new position (from
  200 ms of audio "nothing moved", a burst loss; from 300 to 1200 ms the confirm proper), a tick
  whose full window is not accepted may re-lock on the **newer half** window alone (two agreeing
  ticks, never on a half holding digital silence), and frozen time stays off the 2 s **loss
  clock** (sync-lost). Each freeze is a `frozen` event (`atMs`: where in the turn's audio the
  pointer stopped) and an `unfrozen` event when it ends (`frozenMs`; `confirmed` true only when the
  confirm closed it): the `freezes` column counts them. The page only records; the web SDK fades
  the face to idle during a freeze longer than about 300 ms.
* **The tap is ungained.** The rule's absolute levels (-60 dBFS silent, -50 dBFS reference speech,
  the confirm's 4 dB match, -95 dBFS digital silence) assume the received track at the level the
  processor measured: the page taps the decoded WebRTC track itself (a `MediaStreamAudioSourceNode`
  into the worklet; the keep-alive element's volume 0 and the muting gain after the worklet do not
  touch what it measures). A port that taps after a volume control or a gain node loses the confirm:
  its 4 dB last match fails, so no confirm opens and only the newer half re-locks, a turn later (at
  -6 dB received gain it confirmed 1 of 5 synthetic stalls, at -18 dB none).

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

**download capture.jsonl** saves one JSON object per line, one per RTVI message the page received
(the avatar messages, `turn_envelope` included, and the session messages such as
`user-started-speaking` and `bot-interrupted`), in arrival order:

```json
{"t": 1730580000.123, "tap": 481920, "message": {"label": "rtvi-ai", "type": "server-message", "data": {"tag": "spatialreal-avatar", "turn_start": {"turn_id": "..."}}}}
```

* `t` is the arrival time, epoch seconds.
* `tap` is how many samples of `capture.wav` had arrived by then, so a replay can put the message
  at `tap / sample rate` seconds into the audio.
* `message` is the whole RTVI message. For an avatar message its `data` is the `AvatarMessage`:
  the argument `onServerMessage` gets in `pipecat-client-js`, and what
  `session.handleServerMessage()` takes. This is the format `hack/sync-sim/sim.py --captures` and
  the web SDK's capture replay read, and the format of the capture committed next to this file.

**download capture.wav** saves what the tap received: the bot's decoded track, mono PCM16, at the
rate of the browser's AudioContext (its header says which; 48 kHz on most machines), from the moment
the page started tapping.

**download capture.sync.json** saves, per turn, what the oracle needs to run on the capture
directly: the reference envelope reassembled by index (`reference`, base64, one byte per 5 ms from
the turn's first sample, with `reference_missing` indices never received), the splice when there is
one (`splice`: `kind` `chained` or `gap`, the turn it joins, `join_ms`, `own_offset` — the index of
the turn's own index 0 — and the spliced `values`), the `turn_start` fields the splice depends on,
`turn_end.pcm_ms`, the tap positions of `turn_start`, `turn_end` and `turn_abort`, the onset prior,
and every tick the page ran (`pos`, the prior and what it was — `prior_kind`: `arrival`, `onset`,
`gap`, `chain` or `locked`; captures saved before the gap prior have no `prior_kind` — the onset
prior then, `onset_prior_pos`, null without one; and the search range, whether it was gated,
accepted, the position it accepted as `first_sample_pos` and the four checks). Positions are samples
of `capture.wav`. A turn's received envelope at a tick is the one of `capture.wav` up to `pos`.

Since the stall rule, each tick also records how it accepted (`via`: `full` or `half`), the newer
half's candidate (`half_pos`, `half_refused` when it held digital silence), where the tracker has
the turn after it (`tracked_pos`, freezes and confirms included), whether the pointer is frozen
(`frozen`), the `frozen` / `unfrozen` events since the previous tick (`events`), and the loss clock
(`loss_clock_ms`, `lost` once it reached 2 s). Per turn, `events` lists every event with the
received frame it fired on (`frame`, of 5 ms; `tap`, the sample position at which that frame was
complete): `{"kind": "frozen", "turnId", "atMs"}` and `{"kind": "unfrozen", "turnId", "frozenMs",
"confirmed"}`, the web SDK's shapes. `freezes` lists each freeze episode as `dsp.TurnTracker` keeps
it (`start` frame, `at_ms`, `frames` frozen, `resume`, `how` it closed — `confirm`, `full`, `half`,
`expired`, `no-match`, `superseded`, `end` — and for a confirm `confirm_l` and
`confirm_after_ms`; `open: true` for one in progress at the download), and `lost_at_frames` the
frames where sync-lost fired. The top level adds the rule's constants (`stall`).

## The live run

Needs the SpatialReal, Deepgram, Cartesia, Gemini and Daily keys in `.env`. The first runs were done
on 2026-11-03 with a scripted second participant, twice: once with a canned-reply stand-in for the
language model (the capture next to this file, `capture.jsonl` and `capture.wav`, is that Daily run)
and once with the example as shipped, Gemini included. With the shipped bot: on Daily, eleven turns
with a message lead of 174–367 ms over the detected onset, a first-segment latency of 460–642 ms
(median 527 ms) from the first TTS audio and 1.3–2.5 s from the LLM request (Gemini's 0.6–1.4 s to
its first token is in it), barge-ins aborted 0 and 1 ms after `user-started-speaking` (the two
messages arrive in one burst, so that delta measures nothing, see above), no degraded state; on
LiveKit, eleven turns, lead 236–445 ms, first-segment latency 411–563 ms from the first TTS audio,
barge-ins 1 and 3 ms. Each run had one onset outside the rule: on Daily a soft attack heard 40 ms
late and a click of 334 the processor saw as quiet, on LiveKit an announced onset whose pause had
shrunk to 144 ms in the received audio — pauses and attacks within a few ms or a few sample values
of the rule's limits, moved across them by the Opus decode. The values below are the design's
expectations; the runs above are the measurement.

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
4. Download the capture files, move them into this directory, and commit them: they are the offline
   test material for the web SDK's Pipecat session.

LiveKit has no probe: run the bot with `-t livekit` and use the example page
(`/avatar/?transport=livekit`, see [../client/README.md](../client/README.md)).

## Paired captures

`tests/fixtures/sync/daily-clean/` and `tests/fixtures/sync/livekit-clean/` are the first paired
captures (2026-11-03): the processor's per-turn envelope and messages next to the bot's track as a
real Chrome (156, headless) decoded it, in the format of *The capture* above. They were recorded
unattended against the SpatialReal test deployment with the example bot as shipped (Deepgram,
Gemini, Cartesia):

* **Speaker**: a scripted second participant (daily-python, or the LiveKit Python SDK) joins the room
  and plays ten spoken prompts, two of them as barge-ins 2.5 s into the bot's answer, four of them
  asking for a list with pauses ("Count slowly from one to five, with a pause between each number").
* **Receiver**: Chrome, driven by puppeteer. On Daily it is this page, joined with `?room=&token=`.
  On LiveKit it is a copy of this page with the Daily calls replaced by `livekit-client` 2.22.4 (the
  same tap, recorder, `sync.js` tracker, analysis and downloads), kept outside the repository.
* **No impaired run.** The plan asked for a second run per transport at 3 % loss / 100 ms jitter;
  Network Link Conditioner is not installed on the recording machine and `pf`/dummynet needs root,
  so both captures are clean networks only. The impaired thresholds rest on the spike's simulation.

Each run gave 12 turns (2 aborted by barge-in, 0 `STATUS_DEGRADED`, no splice: every turn followed an
idle longer than 1 s, so `gap_ms` was 0 throughout). The fixtures are trimmed to whole turns, 9 of
Daily's and 8 of LiveKit's, so that each directory stays under 10 MB: each kept turn keeps the audio
from 3.5 s before its `turn_start` to 1.5 s after its last tick, cut on a 40 ms grid (a multiple of
the 5 ms hop and of the 128-sample worklet block), and every tap position is rebased
(`capture.sync.json` says what was kept under `trimmed`). On the kept turns the replay gives the same
numbers on the trimmed and on the full recording.

`hack/sync-sim/replay_capture.py DIR` replays the rule on a capture with the oracle (`dsp.align`,
the page's cadence, prior and widening) and measures it against the whole-turn alignment (the turn's
reference correlated once with everything received of it). On the full recordings (24 turns; the
fixtures are subsets):

| | Daily | LiveKit |
|---|---|---|
| accepted ticks / false locks (> 20 ms off) | 211 / **0** | 200 / **0** |
| first lock within 300 ms of speech | **9 / 12 (75 %)** | **10 / 12 (83 %)** |
| first lock, ms of speech: p50 / p95 / max | 125 / 572 / 685 | 107 / 592 / 718 |
| error against the whole-turn alignment: p50 / p95 / max | 0.62 / 5.00 / 6.94 ms | 0.25 / 1.24 / 1.99 ms |
| error against the phrase-by-phrase alignment: p50 / p95 | 0.22 / 1.82 ms | 0.16 / 1.07 ms |
| received audio moving inside a turn (largest) | 9.8 ms | 5.1 ms |
| \|onset prior - truth\| before the first lock: p50 / max | 0.9 / 6.7 ms | 0.9 / 5.1 ms |
| the page's live ticks (causal high-pass): accepted / false / p95 | 213 / 0 / 5.03 ms | 200 / 0 / 1.27 ms |

Against the design's minimum as §4.1 first wrote it (backend-ng
`2026-11-03-pipecat-sync-fingerprint-design.md`): **no false lock on either transport; the
first-lock minimum (300 ms on 95 % of turns) fails on both**; the p95 error is 5.00 ms on Daily (at
the limit; 5.02 with the browser's causal high-pass) and 1.24 ms on LiveKit. The late first locks
are turns whose first second is mostly the silence before the turn: the candidate is right (peak
0.99 or more, within 7 ms of the whole-turn alignment) but its sharpness is under 0.03 or its peak
ratio under 1.2. Until the lock the frames follow the onset prior, which was within 7 ms of the
truth on every turn here. The Daily p95 comes from the two greetings at the start of the call, where
the received audio moved by up to 10 ms inside the turn (the jitter buffer settling); against each
phrase's own alignment the p95 is 1.8 ms. Lowering the thresholds would meet the first-lock minimum
on these captures, but every relaxation tried re-opens false locks on the spike's impaired
simulation (sharpness 0.025: one at 21.5 ms; ratio 1.1 with sharpness 0.03: six, up to 130 ms; ratio
1.1 with sharpness 0.02: seven). Source of those counts: the spike's impaired simulation,
`hack/sync-sim/sim.py` replayed with each of those thresholds (its `--thr`, see
`hack/sync-sim/README.md`, "Rerunning"). The result pickles are not committed and `thrsearch.out`
lists only the 500-combination grid on the frozen features, so the counts are quoted from that run,
not recomputed here. These clean captures cannot show a false lock at any threshold (no tick of
theirs is more than 7 ms off at the fixed first prior, so `thrsearch.py` on them finds zero for all
500 combinations). The thresholds were not changed.

On the fixtures themselves (the trimmed subsets, which keep the slow turns and so weigh them more):

| | `daily-clean` (9 turns) | `livekit-clean` (8 turns) |
|---|---|---|
| false locks | 0 of 165 accepted | 0 of 133 accepted |
| first lock within 300 ms of speech | 6 / 9 | 6 / 8 |
| first lock, latest (replay / live page) | 685 / 679 ms | 718 / 732 ms |
| p95 error, whole-turn truth | 6.07 ms | 1.28 ms |
| p95 error, phrase-by-phrase truth | 2.32 ms | 0.84 ms |
| \|onset prior - truth\| before the first lock, max | 6.7 ms | 5.1 ms |

**The criteria as restated by the owner after these captures** (they replace the two that failed as
written; zero false locks stays): the error is measured against the **phrase-by-phrase** alignment,
p95 at most 5 ms (R63), because the whole-turn alignment charges the rule for audio that really moved
inside the turn; and the first lock is **every turn locks within 750 ms of speech, and until it
locks the onset prior is within 10 ms of the truth on at least 95 % of turns** (R62). Both fixtures,
and both full recordings, meet all three: no false lock, every turn locked by 732 ms (live) / 718 ms
(replay), the prior within 6.7 ms on every turn, and a phrase-truth p95 of 2.32 and 0.84 ms (1.82 and
1.07 ms on the full recordings). `replay_capture.py` prints these restated criteria (the 300 ms
count stays beside them for reference), and `tests/test_sync_replay.py` asserts them on the
fixtures. Since the splice run below, the error truth is the window content rather than the phrases
(*The splice run*).

**Pauses and late onsets (design §8, item 2).** 26 pauses of 0.3 s or more were received. The audio
after each pause sat where the audio before it did, within -4.8 to +5.1 ms (each phrase aligned on
its own); no pause moved it by tens of milliseconds, and no lead-in was lost (no reference frame of
the first 150 ms after a pause 10 dB over the floor while the received track stayed at it). The onset
rule still missed on LiveKit: +57.8 ms on one onset (after a 260 ms quiet), +12 to +18 ms on three
more and -11 ms on one, with the phrase not moved (-2.8 to +0.4 ms). The +57.8 ms one is a click at
the start of the phrase whose decoded peak is 327 where the server's was over 328: the threshold
missed it by one sample value and found the voiced onset 58 ms later. On these captures, then, a late onset after a
pause is the onset threshold meeting a soft lead-in, not a delay change and not a lead-in removed by
DTX.

### The splice run (2026-11-04)

`tests/fixtures/sync/daily-splice/` is 20 whole turns of a third run (Daily, Chrome 156) made so that
turns are spliced (design §3.3): a scratch copy of the example bot gave each sentence its own
Cartesia context, so a three-sentence answer plays as three chained turns, and followed each answer
with a short question 100-400 ms after it, a turn with `gap_ms` > 0. The full run had 55 turns: 25
chained, 16 with a gap of 200-800 ms, 14 plain. The fixture keeps turns 9-28 (10 chained, 6 with a
gap), trimmed like the clean ones; the replay gives the same numbers on the trimmed and the full
recording.

**The joins.** On every one of the 41 joins the received track holds exactly the gap the splice
assumes: the received distance between a turn's last sample and the next one's first, minus
`gap_ms`, is at most 2.1 ms (p95 1.1 ms), measured from the 400 ms windows nearest the join on both
sides. `tests/test_sync_replay.py` asserts it within one hop on the fixture.

**The truth.** The same run shows Chrome's jitter buffer moving the received audio inside a turn by
5-60 ms, in steps: a turn whose last 200 ms arrived 51 ms later than the rest of it, a turn whose
first 240 ms arrived 65 ms ahead of the rest. The rule follows those steps, so a phrase truth (one
position per phrase of up to 5 s) charged it for them: 41 "false locks" on the first replay, none
of them a lock away from the audio it was looking at. `replay_capture.py` now measures every
accepted tick against the window content: 400 ms windows of the reference, each aligned on its own,
overlap-weighted over the tick's 1 s window (its docstring has the definition). It also ticks a turn
the way the page does, from the turn's first `turn_envelope` with the reference as it had arrived,
with the prior the page had; with the causal high-pass its ticks are the page's own, tick for tick,
on 63 of the 64 turns of the three captures (1 723 ticks). On the other one tick holds on a ratio of
1.1998 where the page accepted on 1.2003 (the 16-bit rounding of `capture.wav`), and that turn's
later ticks shift with it.

| | `daily-clean` | `livekit-clean` | splice run (55 turns) | `daily-splice` (20) |
|---|---|---|---|---|
| accepted ticks / false locks (window truth) | 167 / **0** | 134 / **0** | 720 / **0** | 204 / **0** |
| first lock, max ms of speech (live page) | 679 (679) | 732 (732) | **1065** (1065) | 156 (156) |
| onset prior within 10 ms before the lock | 9 / 9 | 8 / 8 | 53 / 55 (96 %) | 20 / 20 |
| p95 error, window truth | 1.24 ms | 0.83 ms | 0.86 ms | 0.60 ms |

(Zero-phase high-pass; with the causal one the p95 moves by at most 0.05 ms and the splice run has
719 accepted ticks, still none false.) The one
criterion the splice run fails is the first lock of one turn, kept as
`tests/fixtures/sync/daily-splice-failures/` and marked `xfail` there: its lead-in and first
syllable reached the browser 65 ms ahead of the rest of it. The page's onset rule fired on them, so
the prior was 68 ms off the rest of the turn; the rule's first two ticks found the early copy (peak
0.998) but held on the peak ratio (1.08, 1.06), and the next 20 held because their 1 s window
straddled the step, until it lay past it: first lock after 1065 ms of speech. That is the re-lock
delay design §3.2 accepts after a jitter-buffer jump, here at the start of a turn.

### The stalled run (2026-11-04)

`tests/fixtures/sync/daily-stall/` is 8 whole turns (9-12 and 42-45 of 54) of a Daily run in which
the bot's host stalled twice, trimmed like the others (42 s kept of 242 s; the replay gives the same
numbers per turn on the trimmed and the full recording). At 15:52:14.6 fixture turn 0 arrived 240 ms
late from a point on and its chained follow-up (turn 1) 891 ms late; at 15:54:38-46 turn 5 (chained)
arrived 1657 ms late after 1.66 s of concealment and zeros, and Chrome then played the rest of it
and turn 6 faster, in -20 to -45 ms steps. `daily-stall/oracle-ticks.json` holds the oracle's ticks
on it (`tests/fixtures/sync/make_stall_fixture.py`).

With the browser's causal high-pass, on the dense local truth (`replay_capture.py --gaps`):

| | today's rule | stall rule |
|---|---|---|
| re-align after the 240 ms stall (turn 0) | 1125 ms | **280 ms** |
| re-align after the 891 ms stall (turn 1) | never | **275 ms** |
| re-align after the 1.66 s stall (turn 5) | never | **290 ms** |
| face ahead of the sound during the 1.66 s gap | 1655 ms | 90 ms |
| ms with the face > 20 ms off after the first lock, + 2 s per sync-lost turn, all turns | 9345 | 7690 (no turn worse) |
| accepts (full, half, confirm) / false outside or across a gap | 64 / 0 | 72 / 0 |
| freezes (confirmed) / newer-half accepts / sync-lost turns | - / - / 2 | 7 (4) / 5 / 2 |

Against the window-content truth segmented at each re-align: 78 accepted ticks, none false. The two
sync-lost turns are turns 5 and 6, which Chrome played faster after the 1.66 s stall: neither rule
follows that. With the zero-phase high-pass turn 1's freeze is not confirmed and that stall is not
re-aligned either (the others: 280 and 340 ms). `tests/test_sync_replay.py` asserts the re-align
times (within 50 ms), no turn worse than today's, zero false locks, and today's long hold on the
same capture.

The page's own tracker (`sync.js`) reproduces the oracle on this capture tick for tick (405 ticks;
positions, accept/hold and how, the newer half, the loss clock, the tracked position and every
`frozen` / `unfrozen` event), both on the oracle's received envelope and on its own from
`capture.wav`, and so it does on the synthetic cases of `tests/fixtures/sync/stall.json` (a stall,
a burst loss, two periodic references, the 2 s cap, three chained turns; 595 ticks under the stall
rule and today's), `tests/probe_check.mjs`. The one value not compared is the candidate of a held
tick whose correlation curve is flat to rounding (peak about 1e-17: the reference all floor in the
search range), where the order of summation picks the lag.
