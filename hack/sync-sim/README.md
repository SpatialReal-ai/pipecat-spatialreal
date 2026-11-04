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
  `capture.sync.json` + `capture.jsonl`, e.g. `tests/fixtures/sync/daily-clean`) and measures it
  against the turn's whole-turn alignment, with the pauses and onsets inside each turn; the numbers
  of the first captures are in `examples/probe/README.md`, *Paired captures*.
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
2.32 and 2.31 ms. `tests/test_sync_replay.py` replays the captures both ways.

## Rerunning

```bash
pip install -r requirements.txt            # numpy
./make_speech.sh                           # clean/*.wav: 2 voices x 20 sentences, macOS `say` + ffmpeg
python sim.py OUT.pkl --captures CAPTURES --runs daily,daily-shipped,livekit,livekit2,livekit-shipped \
    --thr 0.95,1.2,0.03,0.01 0.85,1.2,0.05,0.02 0.8,1.1,0.02,0.02
python thrsearch.py OUT.pkl                # the threshold grid (500 combinations)
python analyze.py OUT.pkl 0.95,1.2,0.03,0.01 0.85,1.2,0.05,0.02 0.8,1.1,0.02,0.02 > RESULTS.md
```

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
