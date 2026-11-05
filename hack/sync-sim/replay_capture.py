"""Replay the alignment rule on a paired capture and measure it against the received content.

usage: python replay_capture.py CAPTURE_DIR [--causal] [--r72-prior] [--thr C,R,S,TOL_MS] [--features OUT.pkl] [--json OUT.json]
                                [--no-stall-rule] [--gaps]

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
  page's own ticks, since the analysis runs on a wall-clock timer, so a capture replays with the
  prior it was recorded with); after the first lock the last
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
  overlaps its 1 s window. A turn whose accepted positions jump by more than 100 ms (a re-align after
  a stall) has its windows aligned around each segment's position as well
  (:func:`segmented_content_windows`), so the windows on both sides of the stall are found. A tick that no window overlaps (only seen on a turn's last syllable after
  a step, where the 400 ms windows straddle the step) is scored against its own 1 s window aligned
  on the whole reference over +/- 600 ms (:func:`own_window_truth`), and counted (``fallback``). A
  false lock is an accepted tick more than 20 ms off this truth. The prior's truth is the turn's
  first window in its own part. The whole-turn alignment (the turn's own reference correlated once
  against everything received of it, +/- 600 ms) is kept as a column, and so is the phrase truth
  below.
* **Joins**: for a spliced turn, δ is the received distance between the previous turn's last and
  this turn's first sample minus ``gap_ms`` (:func:`join_delta`, from the content windows nearest
  the join on both sides).
* **The prior before the lock (R72)**: since R72 the page places an unchained turn with
  ``gap_ms`` > 0, whose previous turn had ended and was not aborted when its ``turn_start``
  arrived, at the previous turn's place + its ``turn_end.pcm_ms`` + ``gap_ms`` until an onset prior
  or a lock replaces it (:func:`gap_join`). The page records each tick's prior kind
  (``prior_kind``); a gap prior is replayed from the previous turn's replayed place, not from the
  recorded value. Captures recorded before R72 have no ``prior_kind``, hold the arrival prior in
  their ticks and replay with it. For every such turn the replay also reports what each prior
  would have placed at its first lock against the truth (``prelock_prior_error_ms``, the prior it
  replayed with; ``r72_prior_error_ms``, the R72 prior, None where an onset prior was in effect,
  which R72 leaves first). ``--r72-prior`` replays those turns with the R72 prior instead.
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

**The stall rule** (``dsp.TurnTracker``, README "The stall rule") is replayed by default
(:func:`replay_turn_tracked`, :func:`replay_all_tracked`); ``--no-stall-rule`` replays today's rule
(``dsp.TODAY``, the page's tick for tick). Per turn the output lists the freezes (where in the
turn's audio the pointer stopped, when, for how long, how the episode closed and after how much
audio a confirm accepted), the newer-half accepts, sync-lost and the loss clock at the last tick.
``--gaps`` scores the rule and today's on the dense local truth (300 ms reference windows every
25 ms, +/- 2 s, :func:`dense_track`): per gap the shift, the re-align time, how far ahead the face
ran during it and the largest error after it, and the accepts more than 20 ms off outside or
across a gap.

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
import itertools

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
# A jump of the accepted position by more than this splits a turn for the content windows.
REALIGN_MS = 100.0


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


def recorded_kind(k, default):
    """What a pre-lock tick's prior was: its ``prior_kind`` (``onset``, ``gap`` or ``arrival``), or,
    in a capture saved before the page recorded it, ``arrival`` when ``prior_pos`` is
    ``turn_start`` + 200 ms (``default``) and ``onset`` otherwise (those pages had no gap prior)."""
    kind = k.get("prior_kind")
    if kind is not None:
        return kind
    return "arrival" if abs(k["prior_pos"] - default) < 0.5 else "onset"


def recorded_gap_prior(t, fs):
    """Whether the page placed ``t`` with the R72 gap prior before its first lock."""
    default = t["start_tap"] + round(PRIOR_LEAD_MS * fs / 1000)
    for k in t.get("ticks") or []:
        if recorded_kind(k, default) == "gap":
            return True
        if k["accepted"]:
            break
    return False


def page_prior(t, fs, hop):
    """The onset prior the page had at a tap position, as a function ``pos -> samples or None``.

    The page sets it from its analysis pass, which runs up to 100 ms after an onset is heard and
    can move it (a chained turn placed from its predecessor's end, then heard): wall-clock timing
    the capture does not hold. Its ticks do: before the first lock each records the prior it used
    (``prior_pos``) and what it was (``prior_kind``, :func:`recorded_kind`). The replay follows
    that record tick by tick, taking the recorded value of an onset prior only: an arrival prior
    is ``turn_start`` + 200 ms, and a gap prior (R72) follows the previous turn's replayed place,
    which the caller computes (None here for both). Past the page's last tick before its first
    lock (the replay locking later than the page did) it uses the turn's final onset prior, from
    100 ms after the onset it rests on was received (no earlier than that last tick).
    """
    default = t["start_tap"] + round(PRIOR_LEAD_MS * fs / 1000)
    steps = []
    for k in t.get("ticks") or []:
        onset = recorded_kind(k, default) == "onset"
        steps.append((k["pos"], k["prior_pos"] if onset else None))
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


def gap_join(t, prev):
    """R72 (sync.js ``SyncTracker.message``): ``(previous turn id, join ms)`` when the page places
    unchained ``t`` from the previous turn's end until its first lock, else None.

    ``t`` has ``gap_ms`` > 0 and the turn announced before it (``prev``) had its ``turn_end`` and
    no ``turn_abort`` when ``t``'s ``turn_start`` arrived; the join is that ``turn_end.pcm_ms`` +
    ``gap_ms``, as for the splice.
    """
    if t.get("prev_turn_id") or not (t.get("gap_ms") or 0) > 0 or prev is None:
        return None
    end_tap, abort_tap = prev.get("end_tap"), prev.get("abort_tap")
    if end_tap is None or end_tap > t["start_tap"] or prev.get("end_pcm_ms") is None:
        return None
    if abort_tap is not None and abort_tap <= t["start_tap"]:
        return None
    return prev["turn_id"], prev["end_pcm_ms"] + t["gap_ms"]


def place_at(placed, turn_id, pos):
    """Where turn ``turn_id`` was placed at tap ``pos`` (received frames of its index 0): its last
    accepted tick at or before ``pos``, else its prior then. ``placed`` maps an id to the turn's
    replayed ticks and its pre-lock prior (``pos -> frames``)."""
    ticks, prior = placed[turn_id]
    last = None
    for k in ticks:
        if k["pos"] > pos:
            break
        if k["accepted"]:
            last = k["l"]
    return last if last is not None else prior(pos)


def replay_turn(
    t, env, hop, fs, n_samples, params=dsp.DEFAULTS, by_id=None, arrivals=None, gap_prior=None
):
    """The page's SyncTracker.tick loop for one turn, with dsp.align. Returns the ticks.

    As the page runs it (examples/probe/sync.js): the first tick is the first block after the
    turn's first ``turn_envelope`` arrived (and not before ``turn_start``); every tick correlates the
    reference as it had arrived by then, spliced as the page spliced it (:func:`splice_at`); the
    prior before the first lock is the one the page had (:func:`page_prior`). ``gap_prior``
    (``pos -> frames``, the R72 prior) replaces the arrival prior where no onset prior is in effect.
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
        elif gap_prior is not None:
            place = gap_prior(pos)
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


def replay_turn_tracked(
    t,
    env,
    hop,
    fs,
    n_samples,
    params=dsp.DEFAULTS,
    by_id=None,
    arrivals=None,
    gap_prior=None,
    stall=dsp.STALL,
    chain_prior=None,
):
    """:func:`replay_turn` with the per-turn state of :class:`dsp.TurnTracker`: the stall rule
    (freeze, confirm, newer-half re-lock, frozen time off the loss clock) and the 2 s loss clock.

    Same tick positions, reference, splice and priors as :func:`replay_turn`; with ``dsp.TODAY``
    its ticks are :func:`replay_turn`'s (``tests/test_stall_rule.py`` holds them equal).
    ``chain_prior`` (``pos -> frames or None``, rule 5) is the predecessor's locked position +
    ``prev_turn_pcm_ms`` for a chained turn, None while the predecessor has not locked; it is used
    only with ``stall.chain_prior``, through :func:`dsp.chained_prior` (an onset prior re-anchors
    within 200 ms of it, else it is ignored).

    Returns None for a turn without an envelope, else a dict: ``ticks`` (as :func:`replay_turn`'s,
    plus ``via``: ``full`` / ``half``), ``disp`` (received frame -> the position the face uses at
    that frame), ``history`` (``[(pos, l)]``: where the tracker had the turn once locked, for
    :func:`tracked_place_at`), ``events``, ``freezes``, ``lost_at`` (frames) and ``tracker``.
    """
    by_id = by_id or {}
    arrivals = arrivals or {}
    chunks = arrivals.get(t["turn_id"], [])
    if not chunks:
        return None
    ms = lambda v: round(v * fs / 1000)
    prior_at = page_prior(t, fs, hop)
    tr = dsp.TurnTracker(t["turn_id"], params, stall)
    pos = max(t["start_tap"], min(c[0] for c in chunks) + 1)
    ticks, disp, history = [], {}, []
    done, last_tick_pos = None, None
    aborted_by = None if t.get("abort_tap") is None else frames_at(t["abort_tap"], hop)
    use_chain = chain_prior is not None and stall.chain_prior
    while pos <= n_samples:
        pos = -(-pos // BLOCK) * BLOCK
        own = reassemble(chunks, pos)
        if not len(own):
            pos += BLOCK
            continue
        onset_prior = prior_at(pos)
        chain_l = chain_prior(pos) if use_chain else None
        if chain_l is not None:
            prior_l, _ = dsp.chained_prior(
                chain_l, None if onset_prior is None else onset_prior / hop, stall, HOP_MS
            )
            # R76: once the turn widens (10 held ticks), the +/- 600 ms search centres on the
            # onset prior when there is one, wherever the chain put the turn
            wide_at = onset_prior / hop if onset_prior is not None else prior_l
        elif onset_prior is not None:
            prior_l = wide_at = onset_prior / hop
        elif gap_prior is not None:
            prior_l, wide_at = gap_prior(pos), None
        else:
            prior_l, wide_at = (t["start_tap"] + ms(PRIOR_LEAD_MS)) / hop, None
        place = tr.l if tr.locked else prior_l
        window_start = pos / hop - params.window_ms / HOP_MS
        if t.get("abort_tap") is not None and pos > t["abort_tap"] + ms(params.window_ms):
            break
        ended = (
            t.get("end_tap") is not None and t["end_tap"] < pos and t.get("end_pcm_ms") is not None
        )
        if ended and window_start > place + t["end_pcm_ms"] / HOP_MS:
            break
        count = frames_at(pos, hop)
        tr.set_reference(own)
        for j in range(count - 1 if done is None else done + 1, count):
            if aborted_by is None or j < aborted_by:
                before = tr.l
                tr.frame(env, j)
                if tr.locked and tr.l != before:
                    history.append(((j + 2) * hop, tr.l))
            disp[j] = tr.l if tr.locked else place
        done = count - 1
        ref, own_off = splice_at(t, own, by_id, arrivals, pos)
        wide = not tr.locked and tr.held_no_lock >= WIDEN_AFTER
        centre = wide_at if wide and wide_at is not None else place
        search = WIDE_SEARCH_MS if wide else params.search_ms
        since = 0.0 if last_tick_pos is None else (pos - last_tick_pos) / fs * 1000
        rec = tr.tick(env, count, ref, own_off, centre, search, since)
        if rec is None:
            pos += ms(BEFORE_LOCK_MS)
            continue
        last_tick_pos = pos
        rec.update(pos=pos, prior_l=centre, wide=wide, spliced=own_off > 0)
        ticks.append(rec)
        if tr.locked:
            history.append((pos, tr.l))
        pos += ms(AFTER_LOCK_MS if tr.locked else BEFORE_LOCK_MS)
    if done is not None:
        tr.finish(done)
    return {
        "ticks": ticks,
        "disp": disp,
        "history": history,
        "events": tr.events,
        "freezes": tr.freezes,
        "lost_at": tr.lost_at,
        "tracker": tr,
    }


def tracked_place_at(res, pos):
    """Where :func:`replay_turn_tracked`'s result ``res`` had its turn at tap ``pos`` once locked
    (received frames of its index 0, freezes and confirms included), or None before its first
    lock."""
    hist = res["history"] if res else []
    keys = [k for k, _ in hist]
    i = int(np.searchsorted(keys, pos, side="right")) - 1
    return hist[i][1] if i >= 0 else None


def chain_prior_for(t, placed_tracked, fs, hop):
    """Rule 5's ``pos -> frames or None`` for ``t``: its predecessor's tracked position + its
    ``prev_turn_pcm_ms`` once the predecessor has locked; None for an unchained turn or one whose
    predecessor was not replayed."""
    prev = t.get("prev_turn_id")
    pcm = t.get("prev_turn_pcm_ms") or 0
    if not prev or pcm <= 0 or prev not in placed_tracked:
        return None
    jf = round(pcm * fs / 1000) / hop
    res = placed_tracked[prev]

    def at(pos):
        l = tracked_place_at(res, pos)
        return None if l is None else l + jf

    return at


def replay_all_tracked(
    env, hop, fs, n_samples, sync, params=dsp.DEFAULTS, stall=dsp.STALL, r72_all=False
):
    """Every turn of a capture through :func:`replay_turn_tracked`, in order, each turn's priors
    from the turns before it as the page and the SDK have them: the R72 gap prior (the previous
    turn's tracked place, recorded gap priors only unless ``r72_all``) and, with
    ``stall.chain_prior``, rule 5's chained prior.

    Returns ``{turn_id: {"res", "prior_l_at", "r72", "join"}}``: the tracked result (None without an
    envelope), the pre-lock prior as a function of the tap (``pos -> frames``), the R72 prior
    function (or None) and the R72 join (or None).
    """
    out, placed_res = {}, {}
    turns = sync["turns"]
    by_id = {t["turn_id"]: t for t in turns}
    for n, t in enumerate(turns):
        prior_at = page_prior(t, fs, hop)
        default_l = (t["start_tap"] + round(PRIOR_LEAD_MS * fs / 1000)) / hop
        join = gap_join(t, turns[n - 1] if n else None)
        r72 = None
        if join is not None and join[0] in out:
            jf = round(join[1] * fs / 1000) / hop
            prev = out[join[0]]

            def r72(pos, prev=prev, jf=jf):
                l = tracked_place_at(prev["res"], pos)
                return (l if l is not None else prev["prior_l_at"](pos)) + jf

        use_r72 = r72 if r72_all or recorded_gap_prior(t, fs) else None
        chain = chain_prior_for(t, placed_res, fs, hop) if stall.chain_prior else None

        def prior_l_at(pos, prior_at=prior_at, default_l=default_l, use_r72=use_r72, chain=chain):
            o = prior_at(pos)
            c = chain(pos) if chain is not None else None
            if c is not None:
                return dsp.chained_prior(c, None if o is None else o / hop, stall, HOP_MS)[0]
            if o is not None:
                return o / hop
            return use_r72(pos) if use_r72 is not None else default_l

        res = None
        if len(b64(t["reference"])):
            res = replay_turn_tracked(
                t, env, hop, fs, n_samples, params, by_id, sync["arrivals"], use_r72, stall, chain
            )
        if res is not None:
            placed_res[t["turn_id"]] = res
        out[t["turn_id"]] = {"res": res, "prior_l_at": prior_l_at, "r72": r72, "join": join}
    return out


# ---------------------------------------------------------------- gaps: the dense local truth
#
# The stall report's truth (--gaps): a stall moves the rest of a turn by up to seconds, beyond the
# window-content truth's +/- 150 ms. Every 25 ms, 300 ms of the turn's own reference (at least
# half of it over -45 dBFS) is aligned over +/- 2 s around the turn's first lock; a window is kept
# at peak >= 0.9 and peak / second >= 1.1 when a window within two places of it agrees within
# 10 ms (``strict``: peak >= 0.95, ratio >= 1.15, for scoring accepts). A step between consecutive
# kept windows of 15 ms or more is a gap.

DENSE_WIN, DENSE_STEP, DENSE_SPAN = 60, 5, 400  # frames: 300 ms, 25 ms, +/- 2 s


def _ncc_wide(env, ref_db, a, b, lo, hi, floor):
    seg = np.maximum(ref_db[a:b], floor)
    sc = seg - seg.mean()
    ls = np.arange(int(lo), int(hi) + 1)
    ls = ls[(ls + a >= 0) & (ls + b <= len(env))]
    if not len(ls):
        return None, None
    idx = (ls + a)[:, None] + np.arange(b - a)[None, :]
    w = np.maximum(env[idx], floor)
    wc = w - w.mean(1, keepdims=True)
    den = np.sqrt((wc**2).sum(1) * (sc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(den > 0, wc @ sc / den, 0)
    return ls, c


def dense_track(env, ref_db, centre, strict=False, floor=-70.0):
    """``[(a, l, peak)]``: reference window start ``a`` (frames) and the position ``l`` it puts the
    turn's index 0 at, kept as described above."""
    min_pk, min_ratio = (0.95, 1.15) if strict else (0.9, 1.1)
    lo, hi = centre - DENSE_SPAN, centre + DENSE_SPAN
    tr = []
    for a in range(0, len(ref_db) - DENSE_WIN + 1, DENSE_STEP):
        b = a + DENSE_WIN
        if np.mean(ref_db[a:b] > PAUSE_DB) < 0.5:
            continue
        ls, c = _ncc_wide(env, ref_db, a, b, lo, hi, floor)
        if ls is None:
            continue
        pk, sec, _, est, _ = dsp.peak_features(ls.astype(np.float64), c, ls[0], ls[-1])
        if pk >= min_pk and (sec <= 0 or pk / sec >= min_ratio):
            tr.append((a, float(est), float(pk)))
    keep = []
    for i, (a, e, pk) in enumerate(tr):
        nb = [tr[k][1] for k in (i - 2, i - 1, i + 1, i + 2) if 0 <= k < len(tr)]
        if any(abs(e - x) * HOP_MS <= 10 for x in nb):
            keep.append((a, e, pk))
    return keep


def dense_truth_frames(tr, n_env):
    """Per received frame: the truth position where the windows covering it agree within 10 ms
    (their mean), else NaN."""
    lo = np.full(n_env, np.inf)
    hi = np.full(n_env, -np.inf)
    acc = np.zeros(n_env)
    cnt = np.zeros(n_env)
    for a, e, _ in tr:
        j0 = round(e + a)
        j1 = min(n_env, j0 + DENSE_WIN)
        if j0 < 0 or j0 >= n_env:
            continue
        lo[j0:j1] = np.minimum(lo[j0:j1], e)
        hi[j0:j1] = np.maximum(hi[j0:j1], e)
        acc[j0:j1] += e
        cnt[j0:j1] += 1
    out = np.full(n_env, np.nan)
    ok = (cnt > 0) & ((hi - lo) * HOP_MS <= 10)
    out[ok] = acc[ok] / cnt[ok]
    return out


def _gap_bounds(env, own_db, l_pre, l_post, j_lo, j_hi):
    """``(j_pre_end, j_res)``: the last frame of 3 matching the pre-gap position, and after it the
    first of 6 frames matching the post-gap position (3 dB, over -50 dB) that the pre-gap position
    does not explain."""

    def match(l, j, tol, lvl):
        k = round(j - l)
        return 0 <= k < len(own_db) and env[j] >= lvl and abs(env[j] - own_db[k]) <= tol

    j_lo, j_hi = int(j_lo), int(min(j_hi, len(env) - 6))
    pre = [j for j in range(j_lo, j_hi) if all(match(l_pre, j - d, 4, -60) for d in range(3))]
    for jr in range(j_lo, j_hi):
        if not all(match(l_post, jr + d, 3, -50) for d in range(6)):
            continue
        if sum(match(l_pre, jr + d, 3, -50) for d in range(6)) >= 3:
            continue
        before = [j for j in pre if j < jr]
        if before:
            return before[-1], jr
    return None, None


def gap_events(res, tr, env, own_db):
    """Each gap of the dense truth: where the content stopped and resumed, the shift, the re-align
    time (from the resume until the face's position stays within 10 ms of the new truth for
    100 ms), how far ahead the face ran during it (frames shown past the last content before it)
    and the largest error from the resume to the next gap."""
    disp = res["disp"]
    js = np.array(sorted(disp))
    if not len(js):
        return []
    dl = np.array([disp[j] for j in js])
    steps = [
        (a0, e0, a1, e1)
        for (a0, e0, _), (a1, e1, _) in itertools.pairwise(tr)
        if abs(e1 - e0) * HOP_MS >= 15
    ]
    events = []
    for n, (a0, e0, a1, e1) in enumerate(steps):
        j_last, j_res = _gap_bounds(env, own_db, e0, e1, e0 + a0, e1 + a1 + DENSE_WIN)
        if j_res is None:
            continue
        zz = np.flatnonzero(env[int(j_last) : int(e1 + a1 + DENSE_WIN)] <= -95)
        if len(zz):
            j_res = max(j_res, int(j_last) + int(zz[-1]) + 1)
        t_re = None
        post = js >= j_res
        jj, dd = js[post], dl[post]
        for i in range(len(jj)):
            if abs(dd[i] - e1) * HOP_MS <= 10 and all(
                abs(v - e1) * HOP_MS <= 10 for v in dd[i : i + 20]
            ):
                t_re = float((jj[i] - j_res) * HOP_MS)
                break
        ins = (js > j_last) & (js < j_res)
        k_gap = j_last - e0
        ahead = ((js[ins] - dl[ins]) - k_gap) * HOP_MS if ins.any() else np.array([0.0])
        j_stop = (
            (steps[n + 1][1] + steps[n + 1][0] + DENSE_WIN) if n + 1 < len(steps) else js[-1] + 1
        )
        win = (js >= j_res) & (js < j_stop)
        events.append(
            {
                "j_pre_end": int(j_last),
                "j_res": int(j_res),
                "shift_ms": float((e1 - e0) * HOP_MS),
                "inserted_ms": float((j_res - j_last) * HOP_MS),
                "realign_ms": t_re,
                "max_ahead_ms": float(np.max(ahead)),
                "max_err_after_ms": float(np.max(np.abs(dl[win] - e1)) * HOP_MS)
                if win.any()
                else None,
                "speech_after_ms": float((min(j_stop, js[-1]) - j_res) * HOP_MS),
            }
        )
    return events


def accepts_of(res):
    """``(count, l, window frames, via)`` of every accept: full and newer-half ticks and confirms."""
    out = [
        (k["count"], k["l"], 100 if k["via"] == "half" else 200, k["via"])
        for k in res["ticks"]
        if k["accepted"]
    ]
    for f in res["freezes"]:
        if f["how"] == "confirm":
            out.append((f["closed"] + 1, f["confirm_l"], f["closed"] + 1 - f["resume"], "confirm"))
    return out


def score_accepts(res, truth):
    """``(false outside a gap, false straddling a gap, scored, worst)`` against per-frame truth:
    an accept is false when it is more than 20 ms off the truth of the frames its window covered;
    a window whose truth spans more than 10 ms straddles a gap and must land on one of its real
    positions. ``worst`` is the largest error of any accept outside a gap (the margin to 20 ms)."""
    false, straddle, scored, worst = [], [], 0, 0.0
    for count, l, m, via in accepts_of(res):
        seg = truth[max(0, count - m) : count]
        seg = seg[~np.isnan(seg)]
        if not len(seg):
            continue
        scored += 1
        if (seg.max() - seg.min()) * HOP_MS <= 10:
            e = abs(l - seg.mean()) * HOP_MS
            worst = max(worst, e)
            if e > FALSE_LOCK_MS:
                false.append((count, via, round(e, 1)))
        else:
            e = float(np.min(np.abs(seg - l)) * HOP_MS)
            if e > FALSE_LOCK_MS:
                straddle.append((count, via, round(e, 1)))
    return false, straddle, scored, worst


def stall_capture_report(env, hop, fs, n_samples, sync, rules, params=dsp.DEFAULTS):
    """Each rule in ``rules`` (``{name: StallParams}``) over a capture, scored on the dense truth.

    Returns ``{name: [per-turn dict]}`` with the tracked result's freezes, half accepts, sync-lost,
    false accepts (outside a gap / straddling one), the gap events and the loss clock at the turn's
    last tick. The truth's centre is today's first lock, so every rule is scored on the same truth.
    """
    runs = {
        name: replay_all_tracked(env, hop, fs, n_samples, sync, params, st)
        for name, st in rules.items()
    }
    today = (
        replay_all_tracked(env, hop, fs, n_samples, sync, params, dsp.TODAY)
        if "today" not in runs
        else runs["today"]
    )
    out = {name: [] for name in rules}
    for n, t in enumerate(sync["turns"]):
        own = b64(t["reference"])
        if not len(own):
            continue
        own_db = dsp.dequantise(own)
        base = today[t["turn_id"]]["res"]
        acc = [k["l"] for k in (base["ticks"] if base else []) if k["accepted"]]
        centre = acc[0] if acc else today[t["turn_id"]]["prior_l_at"](t["start_tap"])
        tr = dense_track(env, own_db, centre)
        truth = dense_truth_frames(dense_track(env, own_db, centre, strict=True), len(env))
        for name in rules:
            res = runs[name][t["turn_id"]]["res"]
            if res is None:
                continue
            false, straddle, scored, _ = score_accepts(res, truth)
            last = res["ticks"][-1] if res["ticks"] else {}
            out[name].append(
                {
                    "turn": n,
                    "turn_id": t["turn_id"],
                    "aborted": t.get("abort_tap") is not None,
                    "accepted": sum(k["accepted"] for k in res["ticks"]),
                    "half": sum(k["via"] == "half" for k in res["ticks"]),
                    "freezes": res["freezes"],
                    "lost": len(res["lost_at"]),
                    "loss_clock_ms": last.get("unlocked_ms"),
                    "false": false,
                    "false_straddle": straddle,
                    "scored": scored,
                    "gaps": gap_events(res, tr, env, own_db),
                }
            )
    return out


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


def realign_anchors(acc, jump_ms=REALIGN_MS):
    """The positions of a turn's accepted ticks split where consecutive accepts jump by more than
    ``jump_ms`` (a re-align after a stall): the last position of each segment."""
    anchors, prev = [], None
    for k in acc:
        if prev is not None and abs(k["l"] - prev) * HOP_MS > jump_ms:
            anchors.append(prev)
        prev = k["l"]
    if prev is not None:
        anchors.append(prev)
    return anchors


def segmented_content_windows(t, env, hop, l_truth, end, acc):
    """:func:`content_windows` around the whole-turn alignment and, when the turn re-aligned
    (accepted positions jumping by more than ``REALIGN_MS``, e.g. after a stall), also around each
    segment's position; a window found more than once keeps its best peak. Each window is aligned
    only +/- 150 ms around its anchor, so without this a turn whose audio moved by a stall has no
    windows (or the wrong ones) on one side of it, and the ticks there are scored against the
    other side's position."""
    windows, own_off = content_windows(t, env, hop, l_truth, end)
    anchors = realign_anchors(acc)
    if len(anchors) < 2:
        return windows, own_off
    best = {s: (s, l, pk) for s, l, pk in windows}
    for a in anchors:
        for s, l, pk in content_windows(t, env, hop, a, end)[0]:
            if s not in best or pk > best[s][2]:
                best[s] = (s, l, pk)
    return [best[s] for s in sorted(best)], own_off


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


def print_stall_section(results, hop):
    """Per turn: the freezes (where the pointer stopped in the turn's audio, received frame of the
    start, length, how the episode closed and, for a confirm, after how much audio), newer-half
    accepts, sync-lost and the loss clock at the turn's last tick."""
    print("\n## Stall rule: freezes, newer-half accepts, loss clock\n")
    print(
        "| turn | freezes (at ms in the turn: start s, frozen ms, closed by [after ms]) | half accepts | sync-lost | loss clock ms at last tick |"
    )
    print("|---|---|---|---|---|")
    n_f = n_c = n_h = n_l = 0
    for r in results:
        fz = r.get("freezes") or []
        if not (fz or r.get("half_accepts") or r.get("sync_lost")):
            continue
        n_f += len(fz)
        n_c += sum(f["how"] == "confirm" for f in fz)
        n_h += r["half_accepts"]
        n_l += r["sync_lost"]
        cells = "; ".join(
            f"{f['at_ms']:.0f}: {f['start'] * HOP_MS / 1000:.2f} s, "
            f"{f['frames'] * HOP_MS:.0f} ms, {f['how']}"
            + (f" [{f['confirm_after_ms']:.0f}]" if f["how"] == "confirm" else "")
            for f in fz
        )
        print(
            f"| {r['label']} | {cells or '-'} | {r['half_accepts']} | {r['sync_lost']} "
            f"| {fmt(r['loss_clock_ms'], 0)} |"
        )
    print(f"\n{n_f} freezes ({n_c} confirmed), {n_h} newer-half accepts, {n_l} sync-lost")


def print_gaps_section(env, hop, fs, n_samples, sync, params, stall):
    """The rule and today's on the dense local truth: per gap the shift, re-align time, the face's
    lead during it and the largest error after it; false accepts outside / across gaps."""
    rules = {"rule": stall} if stall == dsp.TODAY else {"today": dsp.TODAY, "rule": stall}
    rep = stall_capture_report(env, hop, fs, n_samples, sync, rules, params)
    print("\n## Gaps on the dense local truth (300 ms windows every 25 ms, +/- 2 s)\n")
    print(
        "| turn | shift ms | inserted ms | rule | re-align ms | face ahead in gap ms | max err after ms |"
    )
    print("|---|---|---|---|---|---|---|")
    for i, row in enumerate(rep["rule"]):
        for name in rules:
            r = rep[name][i]
            for g in r["gaps"]:
                print(
                    f"| {r['turn']} | {g['shift_ms']:+.0f} | {g['inserted_ms']:.0f} | {name} "
                    f"| {fmt(g['realign_ms'], 0) if g['realign_ms'] is not None else 'never'} "
                    f"| {g['max_ahead_ms']:.0f} | {fmt(g['max_err_after_ms'], 0)} |"
                )
    for name in rules:
        rows = rep[name]
        re_ = [g["realign_ms"] for r in rows for g in r["gaps"] if g["realign_ms"] is not None]
        never = sum(g["realign_ms"] is None for r in rows for g in r["gaps"])
        print(
            f"\n{name}: gaps {sum(len(r['gaps']) for r in rows)}, re-align p50 {fmt(pct(re_, 50), 0)} ms, "
            f"p90 {fmt(pct(re_, 90), 0)} ms, never {never}; sync-lost {sum(r['lost'] for r in rows)}; "
            f"accepts scored {sum(r['scored'] for r in rows)}, false outside a gap "
            f"{sum(len(r['false']) for r in rows)}, across a gap {sum(len(r['false_straddle']) for r in rows)}"
        )


def main(argv=None):
    """Replay one capture directory, print the tables, and return the per-turn numbers."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture_dir")
    ap.add_argument("--causal", action="store_true", help="the browser's causal 30 Hz high-pass")
    ap.add_argument(
        "--r72-prior",
        action="store_true",
        help="replay gap turns with the R72 prior (previous turn's place + pcm_ms + gap_ms) "
        "instead of the arrival prior the capture recorded",
    )
    ap.add_argument("--thr", help="replay with these thresholds: peak,ratio,sharpness,half_tol_ms")
    ap.add_argument("--features", help="write sim.py stage-A features for thrsearch.py")
    ap.add_argument("--json", help="write the per-turn numbers as JSON")
    ap.add_argument(
        "--stall-rule",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="replay with the stall rule (dsp.STALL: freeze, confirm, newer-half re-lock, frozen "
        "time off the loss clock, rule 5's chained prior); --no-stall-rule replays today's rule "
        "(dsp.TODAY), which is the probe page's tick for tick",
    )
    ap.add_argument(
        "--gaps",
        action="store_true",
        help="score the rule on the dense local truth: every gap's re-align time, the face's lead "
        "during it, false accepts outside / across gaps (today's rule beside it)",
    )
    args = ap.parse_args(argv)
    stall = dsp.STALL if args.stall_rule else dsp.TODAY

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
        f"{'' if params == dsp.DEFAULTS else ' (NOT the design defaults)'}; "
        f"{'stall rule' if args.stall_rule else 'today' + chr(39) + 's rule (no stall rule)'}\n"
    )
    tracked = replay_all_tracked(env, hop, fs, len(x), sync, params, stall, args.r72_prior)
    hdr = (
        "| turn | kind | ref ms | rx ms | truth peak | drift ms | held/acc | first lock ms | offset ms "
        "| prior err ms (window / whole) | err window p95 | err window max | false (window) | fallback "
        "| err phrase p95 | false (phrase) | err whole p95 | page first lock ms | page acc | page err p95 | page false |"
    )
    print(hdr)
    print("|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|")
    results, all_err, all_local, all_page_err, feats = [], [], [], [], []
    for n_turn, t in enumerate(sync["turns"]):
        own = b64(t["reference"])
        tid = f"{n_turn:2d} {t['turn_id'][8:14]}"  # index and the id's hhmmss
        prior_at = page_prior(t, fs, hop)
        tk = tracked[t["turn_id"]]
        join, r72, prior_l_at = tk["join"], tk["r72"], tk["prior_l_at"]
        if not len(own):
            print(f"| {tid} | no reference |" + " |" * (hdr.count("|") - 3))
            continue
        res_t = tk["res"]
        ticks = res_t["ticks"] if res_t else []
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
        window, fallback, prior_err_w, content, join_d = [], [], None, [], None
        prelock_err, r72_err = None, None
        if l_truth is not None:
            content, own_off = segmented_content_windows(
                t, env, hop, l_truth, received_end_frame(t, hop, len(env)), acc
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
            join_d = join_delta(t, content, own_off)
            # the prior's truth: where the turn's own first content window puts it
            own_first = [l for s_, l, _ in content if s_ >= own_off]
            if own_first:
                prior_err_w = (prior_l - own_first[0]) * HOP_MS
                if acc:
                    # what placed the frames up to the first lock: the prior at that tick
                    at = acc[0]["pos"]
                    prelock_err = (prior_l_at(at) - own_first[0]) * HOP_MS
                    if r72 is not None and prior_at(at) is None:
                        r72_err = (r72(at) - own_first[0]) * HOP_MS
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
            "join_delta_ms": None if join_d is None else join_d[0],
            "join_windows_ms": None if join_d is None else join_d[1:],
            "r72": join is not None,
            "prelock_prior_error_ms": prelock_err,
            "r72_prior_error_ms": r72_err,
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
            "half_accepts": sum(k.get("via") == "half" for k in ticks),
            "freezes": [] if res_t is None else res_t["freezes"],
            "sync_lost": 0 if res_t is None else len(res_t["lost_at"]),
            "loss_clock_ms": ticks[-1]["unlocked_ms"] if ticks else None,
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

    print("\n## The prior before the first lock of the gap turns R72 places from the turn before\n")
    print(
        f"replayed with the {'R72' if args.r72_prior else 'recorded'} prior; error at the first "
        "lock against the turn's first content window, ms (R72: none where an onset prior was in effect)\n"
    )
    print("| turn | gap ms | first lock ms | replayed prior err ms | R72 prior err ms |")
    print("|---|---|---|---|---|")
    for r in results:
        if r["r72"]:
            print(
                f"| {r['label']} | {r['gap_ms']:.0f} | {fmt(r['first_lock_ms'], 0)} "
                f"| {fmt(r['prelock_prior_error_ms'], 1)} | {fmt(r['r72_prior_error_ms'], 1)} |"
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

    print_stall_section(results, hop)
    if args.gaps:
        print_gaps_section(env, hop, fs, len(x), sync, params, stall)

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
