"""Replay the alignment rule on a paired capture and measure it against the received content.

usage: python replay_capture.py CAPTURE_DIR [--causal] [--thr C,R,S,TOL_MS] [--features OUT.pkl] [--json OUT.json]

A paired capture is what the probe page saves (``examples/probe/README.md``, "The capture"):
``capture.wav`` (the bot's track as the browser decoded it), ``capture.sync.json`` (per turn the
reference envelope reassembled by index, the splice, the tap positions of ``turn_start`` /
``turn_end`` / ``turn_abort``, the onset prior and the page's own ticks) and ``capture.jsonl``
(every RTVI message with its tap position).

What it does, per turn:

* **The rule** (design 3.2), replayed with :func:`dsp.align` the way the page runs it
  (``examples/probe/sync.js``, ``SyncTracker``): the first tick is the first 128-sample block after
  the turn's first ``turn_envelope`` arrived (a chained turn's ``turn_start`` can arrive seconds
  before its audio; its envelope arrives with it); a tick every 50 ms of received audio until the
  first lock, every 200 ms after it; each tick correlates the reference as it had arrived by then,
  spliced onto the previous turn's envelope as far as that had arrived (:func:`splice_at`). The
  prior before the first lock is what the page had: the onset prior once its analysis set it (R57:
  then used unconditionally), else ``turn_start`` + 200 ms (:func:`page_prior` reads it from the
  page's own ticks, since the analysis runs on a wall-clock timer); after the first lock the last
  accepted position; after 10 held ticks with speech before a lock the search widens to +/- 600 ms
  around the onset prior; the ticks stop once the 1 s window lies past the turn's end (``turn_end``
  known by then) or 1 s after ``turn_abort``. With the causal high-pass the replay's ticks are the
  page's tick for tick on the committed captures (same positions, same accept/hold; a tick that
  sits within 0.05 % of a threshold can go the other way, since ``capture.wav`` holds the track
  rounded to 16 bits and the page had Web Audio floats). The received envelope is the oracle's:
  the whole track high-passed at 30 Hz with the zero-phase filter (:func:`dsp.envelope_db`,
  ``dc_block``); ``--causal`` uses the browser's first-order IIR instead (sync.js
  ``ReceivedEnvelope``).
* **The truth** the criteria use is the **window content** (:func:`content_windows`,
  :func:`content_truth`). Chrome's jitter buffer moves the received audio inside a turn: in the
  splice run (55 turns on Daily, 2026-11-04) by 5-60 ms, in steps (a turn's tail 51 ms later than
  the rest of it, a turn's first 240 ms 65 ms ahead of the rest). The rule tracks those moves, so
  any single position per turn or per phrase charges it for them, and a phrase truth has nothing to
  say about a tick whose window holds the previous turn's audio. So the turn's final correlated
  reference (spliced as the page spliced it) is cut into 400 ms windows every 50 ms; each that is
  mostly speech (half its frames over -45 dBFS) is aligned on its own against the received envelope
  at every 5 ms lag over +/- 150 ms around the whole-turn alignment, parabolic peak, kept at peak
  >= 0.9. A tick's truth is the overlap-weighted position of the windows whose received span
  overlaps its 1 s window. A tick that no window overlaps (only seen on a turn's last syllable after
  a step, where the 400 ms windows straddle the step) is scored against its own 1 s window aligned
  on the whole reference over +/- 600 ms (:func:`own_window_truth`), and counted (``fallback``). A
  false lock is an accepted tick more than 20 ms off this truth. The prior's truth is the turn's
  first window in its own part. The whole-turn alignment (the turn's own reference correlated once
  against everything received of it, +/- 600 ms) is kept as a column, and so is the phrase truth
  below.
* **Joins**: for a spliced turn, δ is the received distance between the previous turn's last and
  this turn's first sample minus ``gap_ms`` (:func:`join_delta`, from the content windows nearest
  the join on both sides).
* **Per tick**: ``first lock`` is the speech received at the first accepted tick, in ms from where
  that lock puts ``onsets_ms[0]`` (the page's definition); negative on a spliced turn that locks on
  the previous turn's audio before its own.
* **Phrase truth** (the criteria's error truth until the window content replaced it): the turn's
  phrases (split at every quiet run of 150 ms or more, the onset rule's silence; quiet = a
  reference frame under -45 dBFS, about the onset rule's |pcm16| 328) aligned one by one.
  ``drift`` is the range of their alignments. ``err phrase`` measures every accepted tick against
  the overlap-weighted alignment of the phrases inside its 1 s window. A phrase runs up to 5 s, so
  a step inside it is averaged into the phrase's one position, and a window that lies in the
  previous turn of a splice falls back to this turn's whole-turn alignment.
* **Pauses** (design 8, item 2): every pause of 0.3 s or more inside the reference splits the turn
  into phrases; each phrase is aligned on its own (its span +/- up to
  150 ms of the pauses around it, +/- 200 ms of lag around the truth). ``shift`` is the phrase after
  the pause minus the phrase before it, in ms: non-zero means the received audio after the pause
  really moved (a jitter-buffer delay change). ``lead-in`` counts the reference's first 150 ms after
  the pause that are 10 dB over the floor while the received track sits within 3 dB of it (a
  lead-in the sender's DTX could have eaten). ``onset`` is the onset rule (|pcm16| > 328 after
  150 ms of quiet) on the received track against the announced onset placed by the truth, and by
  the phrase's own alignment.
* **Onsets**: the same onset comparison for every announced onset after the first, whatever the
  pause before it, with the shift of its phrase (150 ms split) against the phrase before. An onset
  that is late while its phrase did not move is the onset rule's error on the waveform, not a
  delay change.

``--thr`` replays the rule with other thresholds (peak, ratio, sharpness, half-window tolerance in
ms) for a proposal; the design's values stay the default and ``dsp.py`` is not changed. The output
is plain text tables; ``--json`` writes the numbers, ``--features`` writes the per-tick
features of every 50 ms tick (fixed first prior, as ``sim.py`` stage A) in the pickle format
``thrsearch.py`` reads (F1 only; F2 and F3 empty).
"""

import argparse
import base64
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsp

HOP_MS = dsp.HOP_MS
BLOCK = 128  # the AudioWorklet's block: a live tick runs at the end of a block
BEFORE_LOCK_MS, AFTER_LOCK_MS, PRIOR_LEAD_MS = 50, 200, 200
WIDEN_AFTER, WIDE_SEARCH_MS = 10, 600
FALSE_LOCK_MS = 20.0
TRUTH_SEARCH_MS = 600
PAUSE_DB, PAUSE_MIN_MS, PHRASE_GAP_MS = -45.0, 300, 150
PHRASE_CONTEXT_MS, PHRASE_SEARCH_MS = 150, 200
LEADIN_MS = 150
ONSET_THR, ONSET_SIL_S = 328, 0.15
# The restated minimum (design 4.1, R62/R63): the criteria tests/test_sync_replay.py asserts.
FIRST_LOCK_MAX_MS = 750  # every turn locks within this much speech
PRIOR_TOL_MS, PRIOR_SHARE = 10.0, 0.95  # until the lock the onset prior is this close, this often
P95_MAX_MS = 5.0  # p95 of the accepted ticks' error against the window-content truth
SPEECH_TURN_MS = 300  # a turn counts for the first-lock criteria when this much of it was received
# The window-content truth (the criteria's error truth): 400 ms reference windows every 50 ms, each
# mostly speech (half its frames over PAUSE_DB), aligned on their own over +/- 150 ms of lag around
# the whole-turn alignment at every 5 ms lag with a parabolic peak, kept at peak >= 0.9.
CONTENT_WIN_MS, CONTENT_STEP_MS, CONTENT_SEARCH_MS = 400, 50, 150
CONTENT_SPEECH_SHARE, CONTENT_MIN_PEAK = 0.5, 0.9


# ---------------------------------------------------------------- inputs


def load(capture_dir):
    """The track, the per-turn record, the announced onsets and every ``turn_envelope`` arrival.

    ``arrivals`` maps a turn id to its envelope messages as ``(tap, first_index, bytes)`` in
    arrival order: what the page had of the turn's reference at any tap position.
    """
    x, fs = dsp.read_wav(os.path.join(capture_dir, "capture.wav"))
    with open(os.path.join(capture_dir, "capture.sync.json")) as fh:
        sync = json.load(fh)
    onsets, arrivals = {}, {}
    with open(os.path.join(capture_dir, "capture.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            d = r["message"].get("data")
            if not isinstance(d, dict):
                continue
            for k in ("turn_start", "turn_onsets"):
                if k in d and isinstance(d[k], dict):
                    onsets.setdefault(d[k]["turn_id"], []).extend(d[k].get("onsets_ms") or [])
            e = d.get("turn_envelope")
            if isinstance(e, dict):
                arrivals.setdefault(e["turn_id"], []).append(
                    (r["tap"], e["first_index"], b64(e.get("values")))
                )
    sync["arrivals"] = arrivals
    return x, fs, sync, onsets


def causal_high_pass(x, fs, hz=30.0, block=4096):
    """sync.js's first-order high-pass, y[n] = a (y[n-1] + x[n] - x[n-1]), vectorised per block."""
    rc = 1 / (2 * np.pi * hz)
    a = rc / (rc + 1 / fs)
    dx = np.diff(np.concatenate([[0.0], x]))
    y = np.empty_like(x)
    k = np.arange(block)
    up = a ** (k + 1)  # a^(n+1)
    down = a ** (-k)  # a^(-k)
    prev = 0.0
    for s in range(0, len(x), block):
        d = dx[s : s + block]
        n = len(d)
        acc = np.cumsum(d * down[:n])
        y[s : s + n] = up[:n] * (prev + acc)
        prev = y[s + n - 1]
        # the recursion carries prev = y[s+n-1]; y[n] = a^(n+1) prev + sum_k a^(n-k+1) d_k
    return y


def received_envelope(x, fs, causal):
    hop = round(HOP_MS / 1000 * fs)
    if causal:
        return dsp._window_db(causal_high_pass(x, fs), hop, 2 * hop), hop
    return dsp.envelope_db(x, fs, HOP_MS / 1000, dc_block=True), hop


def b64(s):
    return np.frombuffer(base64.b64decode(s), np.uint8) if s else np.zeros(0, np.uint8)


# ---------------------------------------------------------------- the rule, replayed


def frames_at(pos, hop):
    """Complete envelope frames once ``pos`` samples have arrived (fully-inside windows)."""
    return max(0, pos // hop - 1)


def reassemble(chunks, before):
    """The page's reassembly (sync.js ``reassembleEnvelope``) of the chunks that arrived before
    tap ``before``: a message arriving at tap T is handled after the block ending at T, so the
    first tick that sees it is the next block's."""
    parts = [(at, v) for tap, at, v in chunks if tap < before and len(v)]
    n = max((at + len(v) for at, v in parts), default=0)
    out = np.zeros(n, np.uint8)
    for at, v in parts:
        out[at : at + len(v)] = v
    return out


def splice_at(t, own, by_id, arrivals, pos):
    """The reference the page correlated at tap ``pos``: ``own`` spliced onto the previous turn's
    envelope as it had arrived by then (sync.js ``reference``/``splicePlan``), or ``own``.

    The replay splices only a turn the page's final record shows spliced, onto the turn it names,
    at the join it names; a tick sees the splice once what it needs has arrived (the previous
    turn's envelope; for a gap splice its ``turn_end``) and not after that turn's ``turn_abort``.
    """
    sp = t.get("splice")
    prev = by_id.get(sp["from"]) if sp else None
    if prev is None:
        return own, 0
    if prev.get("abort_tap") is not None and prev["abort_tap"] < pos:
        return own, 0
    if sp["kind"] == "gap" and (prev.get("end_tap") is None or prev["end_tap"] >= pos):
        return own, 0
    pv = reassemble(arrivals.get(prev["turn_id"], []), pos)
    off = max(0, round(sp["join_ms"] / HOP_MS))
    ref = np.zeros(off + len(own), np.uint8)
    ref[: min(len(pv), off)] = pv[: min(len(pv), off)]
    ref[off:] = own
    return ref, off


def page_prior(t, fs, hop):
    """The onset prior the page had at a tap position, as a function ``pos -> samples or None``.

    The page sets it from its analysis pass, which runs up to 100 ms after an onset is heard and
    can move it (a chained turn placed from its predecessor's end, then heard): wall-clock timing
    the capture does not hold. Its ticks do: before the first lock each records the prior it used
    (``prior_pos``), which is the onset prior when the page had one and ``turn_start`` + 200 ms
    when it had none. The replay follows that record tick by tick. Past the page's last tick before
    its first lock (the replay locking later than the page did) it uses the turn's final onset
    prior, from 100 ms after the onset it rests on was received (no earlier than that last tick).
    """
    default = t["start_tap"] + round(PRIOR_LEAD_MS * fs / 1000)
    steps = []
    for k in t.get("ticks") or []:
        steps.append((k["pos"], None if abs(k["prior_pos"] - default) < 0.5 else k["prior_pos"]))
        if k["accepted"]:
            break
    final = t.get("onset_prior_pos")
    o = t.get("onsets_ms") or []
    final_from = None
    if final is not None:
        heard = final + (o[0] * fs / 1000 if o else 0) + round(0.1 * fs)
        final_from = max(heard, steps[-1][0] + 1) if steps else heard
    last_pos = steps[-1][0] if steps else -1

    def at(pos):
        if pos <= last_pos:
            v = None
            for p, val in steps:
                if p > pos:
                    break
                v = val
            return v
        return (
            final
            if final_from is not None and pos >= final_from
            else (steps[-1][1] if steps else None)
        )

    return at


def replay_turn(t, env, hop, fs, n_samples, params=dsp.DEFAULTS, by_id=None, arrivals=None):
    """The page's SyncTracker.tick loop for one turn, with dsp.align. Returns the ticks.

    As the page runs it (examples/probe/sync.js): the first tick is the first block after the
    turn's first ``turn_envelope`` arrived (and not before ``turn_start``); every tick correlates the
    reference as it had arrived by then, spliced as the page spliced it (:func:`splice_at`); the
    prior before the first lock is the one the page had (:func:`page_prior`).
    """
    by_id = by_id or {}
    arrivals = arrivals or {}
    chunks = arrivals.get(t["turn_id"], [])
    if not chunks:
        return []
    ms = lambda v: round(v * fs / 1000)
    prior_at = page_prior(t, fs, hop)
    locked, last_l, held_no_lock = False, None, 0
    pos = max(t["start_tap"], min(c[0] for c in chunks) + 1)
    ticks = []
    hist = round(params.floor_history_ms / HOP_MS)
    while pos <= n_samples:
        pos = -(-pos // BLOCK) * BLOCK
        own = reassemble(chunks, pos)
        if not len(own):
            pos += BLOCK
            continue
        onset_prior = prior_at(pos)
        if locked:
            place = last_l
        elif onset_prior is not None:
            place = onset_prior / hop
        else:
            place = (t["start_tap"] + ms(PRIOR_LEAD_MS)) / hop
        window_start = pos / hop - params.window_ms / HOP_MS
        if t.get("abort_tap") is not None and pos > t["abort_tap"] + ms(params.window_ms):
            break
        # finished: the whole window lies past the turn's end (once turn_end has arrived)
        ended = (
            t.get("end_tap") is not None and t["end_tap"] < pos and t.get("end_pcm_ms") is not None
        )
        if ended and window_start > place + t["end_pcm_ms"] / HOP_MS:
            break
        ref, own_off = splice_at(t, own, by_id, arrivals, pos)
        wide = not locked and held_no_lock >= WIDEN_AFTER
        centre = onset_prior / hop if wide and onset_prior is not None else place
        search = WIDE_SEARCH_MS if wide else params.search_ms
        count = frames_at(pos, hop)
        base = max(0, count - hist)
        p = dsp.AlignParams(**{**params.__dict__, "search_ms": search})
        r = dsp.align(env[base:count], ref, centre - own_off - base, p)
        if not r.checks.window_ok:
            pos += ms(BEFORE_LOCK_MS)
            continue
        accepted = r.offset_ms is not None
        l_acc = r.offset_ms / HOP_MS + own_off + base if accepted else None
        c = r.checks
        ticks.append(
            {
                "pos": pos,
                "speech": c.speech,
                "accepted": accepted,
                "l": l_acc,  # received frame of the turn's own index 0
                "candidate_l": None
                if c.candidate_ms is None
                else c.candidate_ms / HOP_MS + own_off + base,
                "prior_l": centre,
                "wide": wide,
                "spliced": own_off > 0,
                "checks": (c.peak, c.ratio, c.sharpness, c.half_deviation_ms, c.edge),
                "fails": [
                    n
                    for n, ok in zip(
                        ("peak", "ratio", "sharpness", "halves", "edge"),
                        (c.peak_ok, c.ratio_ok, c.sharpness_ok, c.halves_ok, c.interior_ok),
                    )
                    if not ok
                ]
                if c.speech
                else [],
            }
        )
        if accepted:
            locked, last_l = True, l_acc
        elif c.speech and not locked:
            held_no_lock += 1
        pos += ms(AFTER_LOCK_MS if locked else BEFORE_LOCK_MS)
    return ticks


# ---------------------------------------------------------------- truth and pauses


def correlate_span(env, ref_db, l0, a, b, search_ms, floor):
    """Align reference frames [a, b) around received frame l0 + a (lags +/- search_ms).

    Returns (l, peak) with ``l`` the received frame of the reference's index 0, or (None, peak).
    """
    l0r = round(l0)
    j0, j1 = l0r + a, l0r + b
    if j0 < 0 or j1 > len(env) or b - a < 20:
        return None, None
    nlag = round(search_ms / HOP_MS)
    lags = np.arange(-nlag - 2, nlag + 3)
    pad = nlag + 10 + (b - a)
    ref_p = dsp.pad_reference(ref_db, pad)
    c = dsp.ncc_curve(env[j0:j1], ref_p, pad + a, lags, floor)
    if c is None:
        return None, None
    lags_ms = lags * HOP_MS
    pk, _, _, est_i, edge = dsp.peak_features(lags_ms, c, -search_ms, search_ms)
    if edge:
        return None, float(pk)
    return l0r + est_i / HOP_MS, float(pk)


def received_end_frame(t, hop, n_env):
    """Last received frame (exclusive) the turn can be correlated on: up to turn_abort if aborted."""
    if t.get("abort_tap") is not None:
        return min(n_env, frames_at(t["abort_tap"], hop))
    return n_env


def truth_for(t, env, hop, l0):
    own = b64(t["reference"])
    ref_db = dsp.dequantise(own)
    end = received_end_frame(t, hop, len(env))
    l = l0
    pk = None
    for _ in range(3):
        b = min(len(own), int(end - round(l)))
        floor = dsp.floor_db(env[max(0, int(l) - 600) : max(int(l) + b, int(l) + 1)])
        nl, pk = correlate_span(env, ref_db, l, 0, b, TRUTH_SEARCH_MS, floor)
        if nl is None:
            return None, pk, floor
        if abs(nl - l) < 0.01:
            l = nl
            break
        l = nl
    return l, pk, floor


def content_windows(t, env, hop, l_truth, end):
    """The window-content truth's measurements: the turn's final correlated reference (spliced
    when the page spliced it) cut into 400 ms windows every 50 ms; each window that is mostly
    speech is aligned on its own against the received envelope.

    Returns ``(windows, own_off)``: ``windows`` is ``[(s, l, peak)]`` with ``s`` the window's first
    reference frame and ``l`` the received frame of the *turn's own* index 0 that window puts it at,
    for the windows that align (peak >= 0.9, not at the edge of the search).
    """
    own = b64(t["reference"])
    if t.get("splice"):
        ref_q, own_off = b64(t["splice"]["values"]), t["splice"]["own_offset"]
    else:
        ref_q, own_off = own, 0
    ref = dsp.dequantise(ref_q)
    l0 = l_truth - own_off  # received frame of the correlated reference's index 0
    floor = dsp.floor_db(env[max(0, int(l0) - 600) : max(int(l0) + len(ref), int(l0) + 1)])
    win, step = round(CONTENT_WIN_MS / HOP_MS), round(CONTENT_STEP_MS / HOP_MS)
    out = []
    for s in range(0, len(ref) - win + 1, step):
        if (ref[s : s + win] > PAUSE_DB).mean() < CONTENT_SPEECH_SHARE:
            continue
        if l0 + s + win > end:
            break
        l, pk = correlate_span(env, ref, l0, s, s + win, CONTENT_SEARCH_MS, floor)
        if l is None or pk is None or pk < CONTENT_MIN_PEAK:
            continue
        out.append((s, l + own_off, pk))
    return out, own_off


def content_truth(windows, own_off, count, window):
    """A tick's window-content truth: the overlap-weighted position (received frame of the turn's own
    index 0) of the content windows whose received span overlaps the tick's received window
    ``[count - window, count)``; None when no content window overlaps it."""
    win = round(CONTENT_WIN_MS / HOP_MS)
    w = acc = 0.0
    for s, l, _ in windows:
        lo = l - own_off + s  # where this window's content lies on the received axis
        ov = min(lo + win, count) - max(lo, count - window)
        if ov > 0:
            w += ov
            acc += ov * l
    return acc / w if w else None


def join_delta(t, windows, own_off):
    """δ of a spliced turn's join, ms: the received distance between the previous turn's last
    sample and this turn's first, minus ``gap_ms`` (0 when the received track holds exactly the
    gap the splice assumes; a chained turn's gap is 0).

    Measured locally: the last content window wholly inside the previous turn and the first
    wholly inside this one each place this turn's index 0 on their own; δ is the second minus the
    first. Both windows lie within a few hundred ms of the join, so a jitter-buffer step elsewhere
    in either turn does not enter it. Returns ``(delta_ms, a_end_ms, b_start_ms)`` (how far before
    and after the join the two windows end and start), or None without a window on both sides.
    """
    sp = t.get("splice")
    if not sp:
        return None
    win = round(CONTENT_WIN_MS / HOP_MS)
    prev_end = round(
        (sp["join_ms"] - (t.get("gap_ms") or 0)) / HOP_MS
    )  # the previous turn's length
    a = [(s, l) for s, l, _ in windows if s + win <= min(prev_end, own_off)]
    b = [(s, l) for s, l, _ in windows if s >= own_off]
    if not a or not b:
        return None
    (sa, la), (sb, lb) = a[-1], b[0]
    return (lb - la) * HOP_MS, (own_off - sa - win) * HOP_MS, (sb - own_off) * HOP_MS


def own_window_truth(t, env, hop, l_truth, count, window):
    """The fallback for a tick no content window overlaps: the tick's own received window
    ``[count - window, count)`` aligned on the turn's final correlated reference over +/- 600 ms of
    lag around the whole-turn alignment, every 5 ms lag, parabolic peak (received frame of the
    turn's own index 0, or None). Same content as the tick, so it only catches a lock on a peak
    that is not the window's best match over the wider search with the whole reference."""
    if t.get("splice"):
        ref_q, own_off = b64(t["splice"]["values"]), t["splice"]["own_offset"]
    else:
        ref_q, own_off = b64(t["reference"]), 0
    ref = dsp.dequantise(ref_q)
    l0 = l_truth - own_off
    floor = dsp.floor_db(env[max(0, count - 600) : count])
    a = count - window - round(l0)
    l, _ = correlate_span(env, ref, l0, a, a + window, TRUTH_SEARCH_MS, floor)
    return None if l is None else l + own_off


def quiet_runs(own_db, min_ms):
    """Runs [i, j) of frames under PAUSE_DB of at least ``min_ms``, with speech on both sides, and
    the first and last speech frame (exclusive end)."""
    quiet = own_db < PAUSE_DB
    speech_idx = np.flatnonzero(~quiet)
    if not len(speech_idx):
        return [], 0, 0
    first, last = int(speech_idx[0]), int(speech_idx[-1])
    out, i = [], first
    while i <= last:
        if quiet[i]:
            j = i
            while j <= last and quiet[j]:
                j += 1
            if (j - i) * HOP_MS >= min_ms:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out, first, last + 1


def align_phrases(own_db, runs, first, last, env, l_truth, end, floor):
    """Align every phrase between the quiet ``runs`` on its own; returns [(a, b, l, peak)]."""
    bounds = [first] + [q for r in runs for q in r] + [last]
    phrases = [(bounds[2 * k], bounds[2 * k + 1]) for k in range(len(runs) + 1)]
    ctx = round(PHRASE_CONTEXT_MS / HOP_MS)
    out = []
    for k, (a, b) in enumerate(phrases):
        left = min(ctx, (a - phrases[k - 1][1]) // 2) if k else min(ctx, a)
        right = min(ctx, (phrases[k + 1][0] - b) // 2) if k < len(phrases) - 1 else ctx
        aa, bb = a - left, min(b + right, len(own_db))
        if l_truth + bb > end:
            out.append((a, b, None, None))
            continue
        out.append((a, b, *correlate_span(env, own_db, l_truth, aa, bb, PHRASE_SEARCH_MS, floor)))
    return out


def local_truth(phrases, l_truth, count, window):
    """Overlap-weighted alignment of the phrases inside the received window [count - window, count)."""
    lo, hi = count - window - l_truth, count - l_truth
    w = l = 0.0
    for a, b, la, _ in phrases:
        ov = min(b, hi) - max(a, lo)
        if la is not None and ov > 0:
            w += ov
            l += ov * la
    return l / w if w else l_truth


def onsets_received(x, fs):
    idx = np.flatnonzero(np.abs(x) > ONSET_THR)
    if not len(idx):
        return idx
    gaps = np.diff(np.concatenate([[-(10**12)], idx]))
    return idx[gaps > ONSET_SIL_S * fs]


def pause_report(t, env, hop, fs, l_truth, floor, onsets_x, onsets_ms):
    own_db = dsp.dequantise(b64(t["reference"]))
    ps, first, last = quiet_runs(own_db, PAUSE_MIN_MS)
    if not ps:
        return []
    end = received_end_frame(t, hop, len(env))
    aligned = align_phrases(own_db, ps, first, last, env, l_truth, end, floor)
    out = []
    lead = round(LEADIN_MS / HOP_MS)
    for k, (pa, pb) in enumerate(ps):
        (_, _, lb, pkb), (_, _, la, pka) = aligned[k], aligned[k + 1]
        row = {
            "pause_ms": (pb - pa) * HOP_MS,
            "at_ms": pa * HOP_MS,
            "before_l": lb,
            "after_l": la,
            "before_peak": pkb,
            "after_peak": pka,
            "shift_ms": None if lb is None or la is None else (la - lb) * HOP_MS,
            "received": l_truth + pb <= end,
        }
        if row["received"]:
            j = round(l_truth) + pb
            r_seg = env[j : j + lead]
            s_seg = own_db[pb : pb + lead]
            n = min(len(r_seg), len(s_seg))
            lost = (s_seg[:n] > floor + 10) & (r_seg[:n] <= floor + 3)
            row["leadin_lost_ms"] = int(lost.sum()) * HOP_MS
            jp = round(l_truth) + pa
            row["pause_rx_db"] = (
                float(np.median(env[jp : round(l_truth) + pb])) if pb > pa else None
            )
            row["floor_db"] = float(floor)
            # the announced onset that opens the phrase after the pause
            cand = [o for o in onsets_ms if abs(o - pb * HOP_MS) <= 60]
            if cand:
                o = min(cand, key=lambda v: abs(v - pb * HOP_MS))
                pred = l_truth * hop + o * fs / 1000
                near = onsets_x[(onsets_x >= pred - 0.06 * fs) & (onsets_x <= pred + 0.2 * fs)]
                row["announced_onset_ms"] = o
                row["onset_delta_ms"] = None if not len(near) else (near[0] - pred) / fs * 1000
                if la is not None and len(near):
                    row["onset_delta_local_ms"] = (near[0] - (la * hop + o * fs / 1000)) / fs * 1000
        out.append(row)
    return out


# ---------------------------------------------------------------- features for thrsearch


def stage_a_features(t, env, hop, fs, l_truth, params=dsp.DEFAULTS):
    """sim.py stage A on a real turn: every 50 ms tick with the turn's first prior fixed."""
    own = b64(t["reference"])
    if t.get("splice"):
        ref, own_off = b64(t["splice"]["values"]), t["splice"]["own_offset"]
    else:
        ref, own_off = own, 0
    prior = (
        t["onset_prior_pos"] / hop
        if t.get("onset_prior_pos") is not None
        else (t["start_tap"] + round(PRIOR_LEAD_MS * fs / 1000)) / hop
    )
    end = received_end_frame(t, hop, len(env))
    stop = min(end, l_truth + len(own) + params.window_ms / HOP_MS)
    loose = dsp.AlignParams(min_peak=-9, min_ratio=0, min_sharpness=-9, half_tol_ms=np.inf)
    rows = []
    pos = t["start_tap"]
    hist = round(params.floor_history_ms / HOP_MS)
    while frames_at(pos, hop) <= stop:
        count = frames_at(pos, hop)
        base = max(0, count - hist)
        r = dsp.align(env[base:count], ref, prior - own_off - base, loose)
        c = r.checks
        if c.speech:
            est = c.candidate_ms / HOP_MS + own_off + base
            err = abs(est - l_truth) * HOP_MS / 1000
            rows.append(
                [
                    c.peak,
                    c.ratio,
                    c.sharpness,
                    c.half_deviation_ms / 1000,
                    est * HOP_MS / 1000,
                    est * HOP_MS / 1000,
                    float(c.edge),
                    err,
                    0.0,
                ]
            )
        pos += round(BEFORE_LOCK_MS * fs / 1000)
    return rows


# ---------------------------------------------------------------- report


def minimum(results):
    """The restated design 4.1 numbers of one capture, from ``main``'s per-turn results.

    The error truth is the window-content truth (``window_errors_ms``, see the module docstring):
    the jitter buffer moves the audio inside a turn and the rule tracks it. The prior is measured
    against the same truth at the turn's first content window. The first-lock and prior criteria
    count the turns with at least 300 ms of received speech that the rule saw. The phrase truth
    stays beside it (``phrase_*``).
    """
    seen = [r for r in results if r["accepted"] or r["held"]]
    speech = [r for r in seen if (r["rx_ms"] or 0) >= SPEECH_TURN_MS]
    window = [e for r in results for e in r["window_errors_ms"]]
    local = [e for r in results for e in r["local_errors_ms"]]
    fl = [r["first_lock_ms"] for r in speech]
    locked = [v for v in fl if v is not None]
    prior = [
        abs(r["prior_error_window_ms"]) for r in speech if r["prior_error_window_ms"] is not None
    ]
    return {
        "turns": len(seen),
        "speech_turns": len(speech),
        "ticks": sum(r["accepted"] for r in results),
        "scored": len(window),
        "fallback": sum(len(r["window_fallback"]) for r in results),
        "false_locks": sum(e > FALSE_LOCK_MS for e in window),
        "locked_in_time": sum(v is not None and v <= FIRST_LOCK_MAX_MS for v in fl),
        "within_300": sum(v is not None and v <= 300 for v in fl),
        "never_locked": len(fl) - len(locked),
        "first_lock_p50": pct(locked, 50),
        "first_lock_p95": pct(locked, 95),
        "first_lock_max": max(locked) if locked else None,
        "prior_turns": len(prior),
        "prior_ok": sum(e <= PRIOR_TOL_MS for e in prior),
        "window_p50": pct(window, 50),
        "window_p95": pct(window, 95),
        "window_max": max(window) if window else None,
        "phrase_false_locks": sum(e > FALSE_LOCK_MS for e in local),
        "phrase_p95": pct(local, 95),
        "phrase_max": max(local) if local else None,
    }


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def fmt(v, nd=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def main(argv=None):
    """Replay one capture directory, print the tables, and return the per-turn numbers."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture_dir")
    ap.add_argument("--causal", action="store_true", help="the browser's causal 30 Hz high-pass")
    ap.add_argument("--thr", help="replay with these thresholds: peak,ratio,sharpness,half_tol_ms")
    ap.add_argument("--features", help="write sim.py stage-A features for thrsearch.py")
    ap.add_argument("--json", help="write the per-turn numbers as JSON")
    args = ap.parse_args(argv)

    params = dsp.DEFAULTS
    if args.thr:
        c, r, sh, tol = (float(v) for v in args.thr.split(","))
        params = dsp.AlignParams(min_peak=c, min_ratio=r, min_sharpness=sh, half_tol_ms=tol)
    x, fs, sync, announced = load(args.capture_dir)
    env, hop = received_envelope(x, fs, args.causal)
    onsets_x = onsets_received(x, fs)
    print(f"# {args.capture_dir}")
    print(
        f"{len(x) / fs:.1f} s at {fs} Hz, {len(sync['turns'])} turns; received high-pass "
        f"{'causal first-order (browser)' if args.causal else 'zero-phase (oracle)'}; thresholds "
        f"{params.min_peak}/{params.min_ratio}/{params.min_sharpness}/{params.half_tol_ms} ms"
        f"{'' if params == dsp.DEFAULTS else ' (NOT the design defaults)'}\n"
    )
    hdr = (
        "| turn | kind | ref ms | rx ms | truth peak | drift ms | held/acc | first lock ms | offset ms "
        "| prior err ms (window / whole) | err window p95 | err window max | false (window) | fallback "
        "| err phrase p95 | false (phrase) | err whole p95 | page first lock ms | page acc | page err p95 | page false |"
    )
    print(hdr)
    print("|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|")
    results, all_err, all_local, all_page_err, feats = [], [], [], [], []
    by_id = {t["turn_id"]: t for t in sync["turns"]}
    for n_turn, t in enumerate(sync["turns"]):
        own = b64(t["reference"])
        tid = f"{n_turn:2d} {t['turn_id'][8:14]}"  # index and the id's hhmmss
        if not len(own):
            print(f"| {tid} | no reference |" + " |" * (hdr.count("|") - 3))
            continue
        ticks = replay_turn(t, env, hop, fs, len(x), params, by_id, sync["arrivals"])
        acc = [k for k in ticks if k["accepted"]]
        held = [k for k in ticks if k["speech"] and not k["accepted"]]
        # truth: start from the last accepted position, else the onset prior, else turn_start + 200 ms
        if acc:
            l0 = acc[-1]["l"]
        elif t.get("onset_prior_pos") is not None:
            l0 = t["onset_prior_pos"] / hop
        else:
            l0 = (t["start_tap"] + round(PRIOR_LEAD_MS * fs / 1000)) / hop
        l_truth, truth_pk, floor = truth_for(t, env, hop, l0)
        o_ms = sorted(announced.get(t["turn_id"], t.get("onsets_ms") or []))
        o0 = o_ms[0] if o_ms else 0.0
        errs = [abs(k["l"] - l_truth) * HOP_MS for k in acc] if l_truth is not None else []
        false = sum(e > FALSE_LOCK_MS for e in errs)
        # what places the frames before the first lock: the onset prior (R57), else turn_start + 200 ms
        prior_l = (
            t["onset_prior_pos"]
            if t.get("onset_prior_pos") is not None
            else t["start_tap"] + round(PRIOR_LEAD_MS * fs / 1000)
        ) / hop
        prior_err = None if l_truth is None else (prior_l - l_truth) * HOP_MS
        phrases, drift, local, onset_rows = [], None, [], []
        window, fallback, prior_err_w, content, join = [], [], None, [], None
        if l_truth is not None:
            content, own_off = content_windows(
                t, env, hop, l_truth, received_end_frame(t, hop, len(env))
            )
            m_win = round(dsp.DEFAULTS.window_ms / HOP_MS)
            for k in acc:
                count_k = frames_at(k["pos"], hop)
                truth_k = content_truth(content, own_off, count_k, m_win)
                if truth_k is None:
                    # no content window inside it: the tick's own window over the whole reference
                    truth_k = own_window_truth(t, env, hop, l_truth, count_k, m_win)
                    fallback.append(k["pos"])
                if truth_k is not None:
                    window.append(abs(k["l"] - truth_k) * HOP_MS)
            join = join_delta(t, content, own_off)
            # the prior's truth: where the turn's own first content window puts it
            own_first = [l for s_, l, _ in content if s_ >= own_off]
            if own_first:
                prior_err_w = (prior_l - own_first[0]) * HOP_MS
            own_db = dsp.dequantise(own)
            runs, first_f, last_f = quiet_runs(own_db, PHRASE_GAP_MS)
            end_f = received_end_frame(t, hop, len(env))
            phrases = align_phrases(own_db, runs, first_f, last_f, env, l_truth, end_f, floor)
            ls = [p[2] for p in phrases if p[2] is not None]
            drift = (max(ls) - min(ls)) * HOP_MS if ls else None
            win = round(dsp.DEFAULTS.window_ms / HOP_MS)
            local = [
                abs(k["l"] - local_truth(phrases, l_truth, frames_at(k["pos"], hop), win)) * HOP_MS
                for k in acc
            ]
            onset_rows = []
            for o in o_ms[1:]:
                fo = o / HOP_MS
                # the phrase it opens: the one containing it, or the next one starting within 60 ms
                idx = [i for i, ph in enumerate(phrases) if ph[0] - 12 <= fo < ph[1]]
                if not idx:
                    continue
                i = idx[0]
                la = phrases[i][2]
                lb = phrases[i - 1][2] if i > 0 else None
                pred = l_truth * hop + o * fs / 1000
                if pred > end_f * hop:
                    continue
                near = onsets_x[(onsets_x >= pred - 0.06 * fs) & (onsets_x <= pred + 0.2 * fs)]
                quiet_ms = (phrases[i][0] - phrases[i - 1][1]) * HOP_MS if i > 0 else None
                onset_rows.append(
                    {
                        "onset_ms": o,
                        "quiet_before_ms": quiet_ms,
                        "phrase_shift_ms": None if la is None or lb is None else (la - lb) * HOP_MS,
                        "delta_ms": None if not len(near) else (near[0] - pred) / fs * 1000,
                        "delta_local_ms": None
                        if la is None or not len(near)
                        else (near[0] - (la * hop + o * fs / 1000)) / fs * 1000,
                    }
                )
        first_lock = None
        if acc:
            k = acc[0]
            first_lock = (k["pos"] - (k["l"] * hop + o0 * fs / 1000)) / fs * 1000
        offset = None if not acc else (acc[-1]["l"] * hop - t["start_tap"]) / fs * 1000
        page_acc = [k for k in t["ticks"] if k["accepted"]]
        # the page's own first lock, same definition (its ticks are in capture.sync.json)
        page_first_lock = None
        if page_acc:
            k = page_acc[0]
            page_first_lock = (k["pos"] - (k["first_sample_pos"] + o0 * fs / 1000)) / fs * 1000
        page_err = (
            [abs(k["first_sample_pos"] / hop - l_truth) * HOP_MS for k in page_acc]
            if l_truth is not None
            else []
        )
        rx_frames = received_end_frame(t, hop, len(env)) - (
            round(l_truth) if l_truth is not None else 0
        )
        rx_ms = min(len(own), rx_frames) * HOP_MS if l_truth is not None else None
        kind = []
        if t.get("splice"):
            kind.append(t["splice"]["kind"])
        if t.get("abort_tap") is not None:
            kind.append("aborted")
        pauses = (
            pause_report(t, env, hop, fs, l_truth, floor, onsets_x, o_ms)
            if l_truth is not None
            else []
        )
        res = {
            "turn_id": t["turn_id"],
            "label": tid,
            "kind": kind,
            "ref_ms": len(own) * HOP_MS,
            "rx_ms": rx_ms,
            "truth_l": l_truth,
            "truth_peak": truth_pk,
            "held": len(held),
            "accepted": len(acc),
            "first_lock_ms": first_lock,
            "offset_ms": offset,
            "errors_ms": errs,
            "false_locks": false,
            "prior_error_ms": prior_err,
            "drift_ms": drift,
            "phrase_l": [(a, b, la, pk) for a, b, la, pk in phrases],
            "local_errors_ms": local,
            "window_errors_ms": window,
            "window_fallback": fallback,
            "window_false_locks": sum(e > FALSE_LOCK_MS for e in window),
            "prior_error_window_ms": prior_err_w,
            "content_windows": len(content),
            "gap_ms": t.get("gap_ms") or 0,
            "join_delta_ms": None if join is None else join[0],
            "join_windows_ms": None if join is None else join[1:],
            "ticks": [
                {k: v for k, v in tk.items() if k != "checks"}
                | {"checks": [float(c) if c is not None else None for c in tk["checks"]]}
                for tk in ticks
            ],
            "onsets": onset_rows,
            "page_first_lock_ms": page_first_lock,
            "page_accepted": len(page_acc),
            "page_errors_ms": page_err,
            "page_false": sum(e > FALSE_LOCK_MS for e in page_err),
            "held_reasons": {},
            "pauses": pauses,
        }
        for k in held:
            for f in k["fails"]:
                res["held_reasons"][f] = res["held_reasons"].get(f, 0) + 1
        results.append(res)
        all_err += errs
        all_local += local
        all_page_err += page_err
        print(
            f"| {tid} | {','.join(kind) or '-'} | {res['ref_ms']} | {fmt(rx_ms, 0)} | {fmt(truth_pk, 3)} "
            f"| {fmt(drift, 1)} | {len(held)}/{len(acc)} | {fmt(first_lock, 0)} | {fmt(offset, 1)} "
            f"| {fmt(prior_err_w, 1)} / {fmt(prior_err, 1)} | {fmt(pct(window, 95), 2)} | {fmt(max(window) if window else None, 2)} "
            f"| {res['window_false_locks']} | {len(fallback)} | {fmt(pct(local, 95), 2)} | {sum(e > FALSE_LOCK_MS for e in local)} "
            f"| {fmt(pct(errs, 95), 2)} | {fmt(page_first_lock, 0)} | {len(page_acc)} "
            f"| {fmt(pct(page_err, 95), 2)} | {res['page_false']} |"
        )
        if args.features and l_truth is not None:
            feats += stage_a_features(t, env, hop, fs, l_truth)

    print("\nheld ticks by failed check (a tick can fail several):")
    for r in results:
        if r["held_reasons"]:
            print(f"  {r['turn_id']}: {r['held_reasons']}")

    print("\n## Joins of the spliced turns\n")
    print(
        "| turn | kind | gap ms | δ ms (received distance - gap_ms) | previous turn's window ends ms "
        "before the join | this turn's window starts ms after it |"
    )
    print("|---|---|---|---|---|---|")
    for r in results:
        if r["kind"] and r["kind"][0] in ("chained", "gap"):
            w = r["join_windows_ms"] or (None, None)
            print(
                f"| {r['label']} | {r['kind'][0]} | {r['gap_ms']:.0f} | {fmt(r['join_delta_ms'], 2)} "
                f"| {fmt(w[0], 0)} | {fmt(w[1], 0)} |"
            )

    print("\n## Pauses of 0.3 s or more inside a turn\n")
    print(
        "| turn | at ms | pause ms | phrase before / after peak | shift ms | lead-in lost ms "
        "| pause rx dB / floor | onset Δ ms (truth) | onset Δ ms (phrase) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    n_p = 0
    for r in results:
        for p in r["pauses"]:
            if not p["received"]:
                continue
            n_p += 1
            print(
                f"| {r['label']} | {p['at_ms']} | {p['pause_ms']} | {fmt(p['before_peak'], 3)} / {fmt(p['after_peak'], 3)} "
                f"| {fmt(p['shift_ms'], 2)} | {p.get('leadin_lost_ms', '-')} "
                f"| {fmt(p.get('pause_rx_db'), 1)} / {fmt(p.get('floor_db'), 1)} "
                f"| {fmt(p.get('onset_delta_ms'), 1)} | {fmt(p.get('onset_delta_local_ms'), 1)} |"
            )
    print(f"\n{n_p} received pauses")

    print("\n## Every later announced onset\n")
    print(
        "| turn | onset ms | quiet before ms | phrase shift ms | onset Δ ms (truth) | onset Δ ms (phrase) |"
    )
    print("|---|---|---|---|---|---|")
    for r in results:
        for o in r["onsets"]:
            print(
                f"| {r['label']} | {o['onset_ms']:.0f} | {fmt(o['quiet_before_ms'], 0)} | {fmt(o['phrase_shift_ms'], 2)} "
                f"| {fmt(o['delta_ms'], 1)} | {fmt(o['delta_local_ms'], 1)} |"
            )

    m = minimum(results)
    print("\n## Design 4.1 minimum (restated, R62/R63; error truth: window content)\n")
    print(
        f"turns with speech ticks: {m['turns']}; with >= 300 ms of received speech: "
        f"{m['speech_turns']}; accepted ticks: {m['ticks']} ({m['scored']} scored; {m['fallback']} "
        f"had no content window inside their 1 s window and are scored on their own window)"
    )
    print(
        f"false locks (accepted > {FALSE_LOCK_MS:.0f} ms off the window-content truth): {m['false_locks']}"
        f"  -> {'PASS' if m['false_locks'] == 0 else 'FAIL'}"
    )
    print(
        f"first lock within {FIRST_LOCK_MAX_MS} ms of speech: {m['locked_in_time']}/{m['speech_turns']}"
        f" (every turn), never locked {m['never_locked']}  -> "
        f"{'PASS' if m['locked_in_time'] == m['speech_turns'] else 'FAIL'}"
    )
    print(
        f"  first lock ms: p50 {fmt(m['first_lock_p50'], 0)}, p95 {fmt(m['first_lock_p95'], 0)}, "
        f"max {fmt(m['first_lock_max'], 0)}; within 300 ms (the original criterion): "
        f"{m['within_300']}/{m['speech_turns']}"
    )
    print(
        f"until the first lock the onset prior is within {PRIOR_TOL_MS:.0f} ms of the truth on "
        f"{m['prior_ok']}/{m['prior_turns']} turns = {m['prior_ok'] / max(1, m['prior_turns']):.1%}"
        f" (>= {PRIOR_SHARE:.0%})  -> {'PASS' if m['prior_ok'] >= PRIOR_SHARE * m['prior_turns'] else 'FAIL'}"
    )
    print(
        f"error against the window-content truth: p50 {fmt(m['window_p50'], 2)} ms, p95 {fmt(m['window_p95'], 2)} ms"
        f" (<= {P95_MAX_MS:.0f}), max {fmt(m['window_max'], 2)} ms  -> "
        f"{'PASS' if m['window_p95'] <= P95_MAX_MS else 'FAIL'}"
    )
    print(
        f"  against the phrase truth: false {m['phrase_false_locks']}, p95 {fmt(m['phrase_p95'], 2)} ms, "
        f"max {fmt(m['phrase_max'], 2)} ms; against the whole-turn truth: p50 {fmt(pct(all_err, 50), 2)} ms, "
        f"p95 {fmt(pct(all_err, 95), 2)} ms"
    )
    print(
        f"page's own ticks (causal high-pass, live): accepted {len(all_page_err)}, "
        f"false {sum(e > FALSE_LOCK_MS for e in all_page_err)}, err p95 {fmt(pct(all_page_err, 95), 2)} ms "
        f"(whole-turn truth)"
    )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1, default=float)
    if args.features:
        empty = np.zeros((0, 9), np.float32)
        F = np.array(feats, np.float32) if feats else empty
        with open(args.features, "wb") as fh:
            pickle.dump([{"feat": {"F1": F, "F2": empty, "F3": empty}}], fh)
        print(f"\nwrote {len(F)} stage-A ticks to {args.features}")
    return results


if __name__ == "__main__":
    main()
