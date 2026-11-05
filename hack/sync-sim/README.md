# sync-sim: the normative definition of the envelope fingerprint

The plugin sends the web SDK a **fingerprint of the audio it releases**: a log-RMS envelope, one
byte per 5 ms (`turn_envelope`, see `docs/superpowers/specs/2026-11-03-pipecat-sync-fingerprint-design.md`
in backend-ng). The SDK computes the same curve from the audio track it receives and lines the two
up on *content* instead of arrival time. This directory is the code that decides what "the same
curve", "lined up" and "trustworthy" mean. It is the spike's code, kept as the oracle:

* `dsp.py` is **normative**. The web SDK ports it verbatim; the package's `Envelope`
  (`src/pipecat_spatialreal/fingerprint.py`) is pinned to it bit for bit by
  `tests/test_fingerprint_oracle.py`.
* `sim.py`, `codec.py`, `thrsearch.py`, `analyze.py`, `make_speech.sh` are the simulation that
  chose the thresholds. They are evidence, not part of the contract. `sim.py` calls the same
  `dsp` building blocks (`ncc_curve`, `floor_db`, `peak_features`, `features`, `passes`) that
  `dsp.align` composes, and `tests/test_fingerprint_oracle.py::test_align_equals_the_simulation_path`
  holds the two compositions together: on one tick `align` and the simulation's
  `tick_curves`/`features` must agree to 1e-9, so `RESULTS.md` keeps describing `align`.
* `replay_capture.py` replays the rule on a real paired capture (the probe's `capture.wav` +
  `capture.sync.json` + `capture.jsonl`, e.g. `tests/fixtures/sync/daily-clean`) the way the probe
  page runs it, and measures it against the window content (400 ms reference windows aligned on
  their own, so the truth follows the jitter buffer's steps inside a turn), with the splice joins
  and the pauses and onsets inside each turn; the numbers are in `examples/probe/README.md`,
  *Paired captures* and *The splice run*.
* `thrsearch_stall.py` / `thrsearch_stall.out` choose the stall rule's constants (below) on the
  stall simulation (`sim.py --stall`) and the paired captures.
* `RESULTS.md` is the simulation's output with the frozen thresholds (1 s window): error,
  false locks, time to first lock, re-lock after a gap, and the onset heuristic on the same
  channel for comparison. `thrsearch.out` is the threshold grid's output. Both are regenerated
  by the commands under "Rerunning" (the real-turn rows need the capture directories, which are
  not committed).

Dev only. The package does not import numpy; `numpy` is in the `dev` extra so CI runs the oracle
test. `pip install -r requirements.txt` for running the scripts by hand.

## The two functions

```python
dsp.envelope_bytes(pcm_int16: np.ndarray, sample_rate: int) -> np.ndarray   # uint8
dsp.align(received_env, reference_env, prior_index, params=AlignParams()) -> AlignResult
```

`envelope_bytes`: windows of 10 ms every 5 ms (`ms * sample_rate // 1000` samples), only windows
fully inside the audio (1 s gives 199 values), `10*log10(mean_square / 32768**2 + 1e-10)` dB,
clamped to [-96, 0] and mapped by `floor((db + 96) / 96 * 255 + 0.5)` (round half up). The
package's quantiser is the same expression operation for operation; the spike's own quantiser
(`np.round((db + 96) * 255 / 96)`, half to even, other operation order) was replaced by it.

`align` is one alignment tick:

* `received_env`: the received track's envelope so far, in dB (float), one value per hop,
  fully-inside windows only. The rule uses its last 1 s (200 values).
* `reference_env`: the turn's envelope as sent (`uint8`), starting at reference index 0. Outside
  its range the reference counts as silence (-96 dB).
* `prior_index`: where reference index 0 is expected on `received_env`'s index axis, in hops
  (fractional allowed); the search is +/- 300 ms around it.
* result: `AlignResult(offset_ms, checks)`. `offset_ms` is the parabolic-interpolated position of
  reference index 0 on the received index axis, in ms (received value `j` corresponds to
  reference value `j - offset_ms / 5`), or `None` -- *hold*: the caller changes nothing.
  `checks` (`AlignChecks`) carries the speech gate, the measured peak, ratio, sharpness and
  half-window deviation, the candidate offset even when rejected, and one flag per check.

## Fixed terms (design 3.2)

The thresholds were chosen with these definitions; redefining any of them means re-running
`thrsearch.py`.

* **Floor.** The 5th percentile of the last 3 s of the *received* envelope plus 6 dB, never
  below -70 dBFS. It clips both the received window and the reference before correlating.
* **Speech gate.** The clipped received window needs a standard deviation of at least 2 dB, else
  the tick does nothing. The two halves are gated the same way; a half without speech is skipped.
* **Correlation.** Mean-removed Pearson correlation (normalised cross-correlation) of the last
  1 s of received envelope against the reference at every 5 ms lag within +/- 300 ms of the
  prior.
* **Second-highest peak.** The maximum of the curve outside the main lobe: from the highest
  peak, walk down both sides while the curve does not increase; everything beyond is "outside".
  The ratio is peak / that maximum (infinite when it is <= 0.001).
* **Half-window check.** The same correlation on the older and the newer 0.5 s of the same
  window, from the same tick. Each half is searched within +/- 40 ms of the full-window
  (interpolated) peak; it must peak within 10 ms of it, and not at the edge of that range.
* **Sharpness.** The interpolated peak value minus the larger of the curve values at the
  interpolated peak +/- 20 ms (linear interpolation of the curve).
* **Parabolic interpolation.** Over the three lags around the peak; the estimate stays on the
  grid at the range's ends.
* **Thresholds.** Peak >= 0.95, peak / second peak >= 1.2, sharpness >= 0.03, half-window
  deviation <= 10 ms, and the peak not at the edge of the search range. All must hold.
* **Cadence** (not in `dsp.align`, which is one tick): a tick every 50 ms until the turn's first
  lock, every 200 ms afterwards.

Known difference to the browser: the simulation's receiver high-pass (30 Hz, `dsp.band_mask_filter`
with `dc_block`) is **zero phase**, applied in the FFT domain over the whole signal. The browser's
is **causal** (a first-order IIR at 30 Hz). Its effect has been checked and is negligible: it
moves the oracle fixture's offset by 0.04 ms; on the paired Chrome captures (`replay_capture.py`
with and without `--causal`) the Daily recording's p95 error against the whole-turn truth is
5.00 ms zero phase and 5.02 ms causal, and the `daily-clean` fixture's against the phrase truth
2.32 and 2.31 ms (against the window-content truth, with the page's cadence: 1.24 and 1.27 ms).
`tests/test_sync_replay.py` replays the captures both ways.

## The stall rule (design 3.2, owner 2026-11-04)

When the bot's host or the receiver stalls, the browser's jitter buffer gets no packets: it plays
a concealment fade (60-130 ms) and then exact digital zeros, and the rest of the turn arrives late
by about the stall. Today's rule coasts at the old offset until its 1 s window clears the jump
(about 1 s with the face moving and no sound), and reaches sync-lost after an insertion beyond its
+/- 300 ms search. The stall rule is per-turn state on top of `align` (which is unchanged):
`dsp.TurnTracker`, run by `replay_capture.replay_turn_tracked` in the page's tick loop. Its
constants are `dsp.StallParams` (`dsp.STALL`; `dsp.TODAY` switches every part off and reproduces
today's ticks exactly). After the turn's first lock:

1. **Freeze.** Once 20 ms of consecutive received frames are each at or below -60 dBFS while the
   turn's own reference at the (still advancing) pointer is above -50 dBFS on every one of them,
   the pointer stops, retroactively over those 20 ms, and stays stopped while the received stays
   silent. Natural pauses and the processor's own silence are in the reference, so they never
   trigger it. While a confirm is pending, only digital silence (<= -95 dBFS) counts as silent.
   **Cap:** after 2 s frozen the pointer advances again and `unfrozen` (`confirmed: false`) fires
   at that frame, so a consumer never shows a freeze longer than 2 s; no new freeze starts until
   received audio resumes, and on the resume the confirm still opens (its accept then moves the
   pointer without a second `unfrozen`).
2. **Confirm.** When audio resumes, every 50 ms from 300 ms to 1200 ms of it, the received audio
   since the resume is correlated with the own reference (both clipped at -70 dBFS) at positions
   in [pointer before the freeze - 150 ms, pointer before the freeze + everything received since
   the last frame that matched the reference before the freeze + 20 ms]. The received audio since
   the resume must pass the 2 dB speech gate (standard deviation after clipping), else that attempt
   does nothing. The last matching frame: the newest frame within 400 ms before the run with 3
   consecutive frames over -60 dBFS within 4 dB of the reference at the pointer then (no such frame:
   no confirm). Accepted at an interior peak >= 0.95 with peak / second-highest peak >= 1.2 and
   sharpness >= 0.03 (curve at the peak +/- 20 ms): `align`'s thresholds. Until it accepts, the
   normal rule continues. A freeze that starts while a confirm is pending keeps that confirm's
   bracket (its pointer before the freeze and last match), so the confirm can still choose
   "nothing moved" (a burst loss) or "everything since moved" (a second stall).
   - **No-shift check.** At 200 and 250 ms of resumed audio, and at the confirm's first attempt
     (300 ms) -- not after -- the same correlation tests "nothing moved" alone: a peak >= 0.95
     within 10 ms of the pointer before the freeze, while the second-highest peak in the range
     stays below 0.95 (the ambiguity guard that stands in for the ratio), is accepted without the
     ratio and sharpness checks.
   - Evaluated and not adopted (`revert_unconfirmed`, off): when the confirm's time runs out,
     put the pointer back where it would be had it never frozen. With the 1200 ms window it gains
     nothing measurable; with an 800 ms window it undoes real stalls (sim sync-lost 2 -> 21).
3. **Newer-half re-lock.** A tick whose full window passes the speech gate but is not accepted
   searches the newer 0.5 s alone within +/- 300 ms of the pointer (peak, ratio, sharpness,
   interior as `align`); it is accepted when the previous candidate agrees within 10 ms, and
   refused when the newer half holds a run of 40 ms of digital silence. The previous candidate is
   the last speech tick's: a tick that fails the speech gate keeps it, a speech tick without a
   (non-refused) candidate and any accept clear it -- the two agreeing ticks are consecutive
   *speech* ticks (200 ms apart unless a silent tick lies between).
4. **Loss clock.** Frozen time does not count toward the 2 s loss clock (sync-lost).
5. **Chained prior** (R74, R76). A chained turn whose predecessor is locked takes the predecessor's
   position (freezes and confirms included) + `prev_turn_pcm_ms` as its prior; before its own
   first lock an onset prior re-anchors it only within +/- 200 ms of that position
   (`dsp.chained_prior`), as the web SDK does. Once the turn has held 10 speech ticks and widens
   to +/- 600 ms, the wide search centres on the onset prior when one exists (R76): a stall in the
   predecessor's trailing silence (which it cannot freeze on) puts the chain off by the stall, and
   past 200 ms the turn would otherwise never lock.

Order per tick, as the replay runs it: every received frame up to the tick's last complete frame
(`count - 1`) goes through `TurnTracker.frame` first (freeze, resume, confirm attempts, cap), then
the tick (`TurnTracker.tick`: `align`, newer half, loss clock, lock).

The probe page ports the tracker and this tick loop (`examples/probe/sync.js`: `TurnTracker`,
`SyncTracker`); `tests/probe_check.mjs` holds it to them tick for tick on the oracle's output for
synthetic stalls and for the committed stalled capture (`tests/fixtures/sync/stall.json`,
`daily-stall/oracle-ticks.json`, written by `tests/fixtures/sync/make_stall_fixture.py`). A capture
the page saves since then records the onset prior on every tick (`onset_prior_pos`), which
`replay_capture.page_prior` takes as it is: the prior's kind alone (`chain`) no longer says it.

Events, in the web SDK's shapes: `frozen` (`turnId`, `atMs`: the turn's own audio position where
the pointer stopped) when a freeze starts, `unfrozen` (`turnId`, `frozenMs`, `confirmed`) when
its episode closes (the confirm accepted: true; a full or newer-half lock, the confirm's time ran
out, no confirm possible, a new freeze, the turn's end, or the 2 s cap: false). Presentation (a
freeze longer than about 300 ms fades the face to idle) is the SDK's, not the oracle's.

**Levels.** The absolute levels (-60 dBFS silent, -50 dBFS reference speech, the 4 dB last-match
tolerance, -95 dBFS digital silence) assume the received track has the reference's level: the
SDK must tap the decoded track before any volume or gain stage. Measured on synthetic turns
(`_speechlike`, 5 seeds, a 300 ms stall): at -6 and -12 dB received gain the confirm accepted 1
of 5 stalls instead of 5 of 5 (the 4 dB last match fails, so no confirm opens; the newer half
re-locked every turn later), at -18 dB none; no false freeze or false accept at any of these gains.

The constants are the threshold search's (`thrsearch_stall.out`): from the spike's (40 ms,
-70 / -40 dB, confirm 300-800 ms) it moved the trigger to 20 ms, the levels to -60 / -50 dB and
the confirm's end to 1200 ms, and added the no-shift check from 200 ms (it cuts the burst-loss
cost and changes nothing on the stall conditions or the captures). A value is adopted
only when its grid neighbours pass with zero false accepts too. **Several sit exactly one grid
step from a value that passes a false accept**, so moving any of them is a re-run of the search,
not a tweak: the confirm's start (300 ms; 200 ms passes false confirms, 250 ms is next to it), its
upper margin (+20 ms; +60 ms does, +40 ms is next to it), the no-shift check's start (200 ms;
100 ms does, 150 ms is next to it), the freeze
cap (2 s; 500 ms does, 1 s is next to it) and the newer half's digital-silence refusal (40 ms;
switching it off does, 80 ms is next to that). The confirm's acceptance thresholds were swept but
kept at `align`'s (a ratio of 1.11 already passed a false confirm on the same runs).

**What the search can and cannot say.** It tunes and reports on the same data: the simulation's
sources are the committed fixtures' bot turns, and the stalled captures (run1, run2, run3, run3b)
are in it too. There is no held-out set. The one-grid-step margins above were found on that
data, and a constant left at the spike's value means only that no alternative on its grid beat it
by 1 % there, not that it is optimal elsewhere.

`replay_capture.py` replays with the rule by default (`--no-stall-rule` for today's, `--gaps` for
re-align times on the dense local truth). Its window-content truth aligns each 400 ms window only
+/- 150 ms around an anchor; a turn whose accepted positions jump by more than 100 ms (a re-align
after a stall) gets windows around each segment's position too
(`segmented_content_windows`), so its "false locks" line counts real ones (run2 turn 43's
pre-stall accepts were reported 1.6 s off before).

## Rerunning

```bash
pip install -r requirements.txt            # numpy
./make_speech.sh                           # clean/*.wav: 2 voices x 20 sentences, macOS `say` + ffmpeg
python sim.py OUT.pkl --captures CAPTURES --runs daily,daily-shipped,livekit,livekit2,livekit-shipped \
    --thr 0.95,1.2,0.03,0.01 0.85,1.2,0.05,0.02 0.8,1.1,0.02,0.02
python thrsearch.py OUT.pkl                # the threshold grid (500 combinations)
python analyze.py OUT.pkl 0.95,1.2,0.03,0.01 0.85,1.2,0.05,0.02 0.8,1.1,0.02,0.02 > RESULTS.md
python sim.py --stall OUT.json             # the stall rule vs today's on the stall conditions
python thrsearch_stall.py --captures DIR... --cache CACHE.pkl > thrsearch_stall.out
python replay_capture.py CAPTURE_DIR --causal --gaps
```

* `sim.py --stall` and `thrsearch_stall.py` take their sources from the committed fixtures (26 real
  bot turns x 13 conditions x 3 seeds); `thrsearch_stall.py` adds the fixtures and every
  `--captures` directory (the stalled Chrome captures, not committed) and takes about 10 minutes on
  10 cores.

* `sim.py` also needs `ffmpeg` with libopus on `PATH` and libopus for ctypes (`OPUS_LIB` points to
  it when the system lookup fails). `--quick` runs a subset; `--jobs N` sets the worker count
  (default 10); `--captures DIR` adds real bot turns from capture directories
  (`DIR/<run>/capture.wav` + `capture.jsonl`, the probe's recording format -- not committed);
  `SYNC_SIM_WIN=0.5` runs the 0.5 s window variant.
* `sim.py` replays **every** `--thr` set for **every** fingerprint (F1, F2, F3), so any set can
  be tabulated later. The three sets given here are the frozen thresholds of F1, F2, F3; for
  `analyze.py` the three sets are **positional**: the first is read as F1's, the second as F2's,
  the third as F3's, and each must have been passed to `sim.py --thr`.
  **F1 is the one that ships** (5 ms hop, 10 ms window, one band); F2 (10 ms hop) and F3 (two
  bands) were rejected, see the design. `thrsearch.py` reads the per-tick features that `sim.py`
  stores in the pickle.
* A full run (40 TTS references + 56 real turns, 2 bitrates, 10 conditions) takes about 7 minutes
  on 10 cores.
