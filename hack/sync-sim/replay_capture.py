"""Replay the alignment rule on a paired capture and measure it against the whole-turn alignment.

usage: python replay_capture.py CAPTURE_DIR [--causal] [--thr C,R,S,TOL_MS] [--features OUT.pkl] [--json OUT.json]

A paired capture is what the probe page saves (``examples/probe/README.md``, "The capture"):
``capture.wav`` (the bot's track as the browser decoded it), ``capture.sync.json`` (per turn the
reference envelope reassembled by index, the splice, the tap positions of ``turn_start`` /
``turn_end`` / ``turn_abort``, the onset prior and the page's own ticks) and ``capture.jsonl``
(every RTVI message with its tap position).

What it does, per turn:

* **The rule** (design 3.2), replayed with :func:`dsp.align` and the cadence of the page and the
  web SDK: a tick every 50 ms of received audio from the arrival of ``turn_start`` until the first
  lock, every 200 ms after it; the prior is the onset prior when the page found one (R57: used
  unconditionally), else ``turn_start`` + 200 ms, and after the first lock the last accepted
  position; after 10 held ticks with speech before a lock the search widens to +/- 600 ms around
  the onset prior; the ticks stop once the 1 s window lies past the turn's end (``turn_end`` known
  by then) or 1 s after ``turn_abort``. The reference is the one the page correlated (spliced onto
  the previous turn when the page spliced it). The received envelope is the oracle's: the whole
  track high-passed at 30 Hz with the zero-phase filter (:func:`dsp.envelope_db`, ``dc_block``);
  ``--causal`` uses the browser's first-order IIR instead (sync.js ``ReceivedEnvelope``).
* **The truth**: the turn's own reference correlated once against the received envelope over the
  whole received turn (up to ``turn_abort`` for an aborted turn), +/- 600 ms, parabolic peak.
* **Per tick**: the error of every accepted tick against the truth; a false lock is an accepted
  tick more than 20 ms off. ``first lock`` is the speech received at the first accepted tick, in
  ms from where that lock puts ``onsets_ms[0]`` (the page's definition).
* **Local truth**: the turn's phrases (split at every quiet run of 150 ms or more, the onset rule's
  silence; quiet = a reference frame under -45 dBFS, about the onset rule's |pcm16| 328) aligned
  one by one. ``drift`` is the range of their alignments: how much the received audio moved inside
  the turn (a jitter buffer stretching or shrinking it). ``err local`` measures every accepted tick
  against the overlap-weighted alignment of the phrases inside its 1 s window, so a turn that
  really moved is not charged to the rule.
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


# ---------------------------------------------------------------- inputs


def load(capture_dir):
    x, fs = dsp.read_wav(os.path.join(capture_dir, "capture.wav"))
    with open(os.path.join(capture_dir, "capture.sync.json")) as fh:
        sync = json.load(fh)
    onsets = {}
    with open(os.path.join(capture_dir, "capture.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)["message"].get("data")
            if not isinstance(d, dict):
                continue
            for k in ("turn_start", "turn_onsets"):
                if k in d and isinstance(d[k], dict):
                    onsets.setdefault(d[k]["turn_id"], []).extend(d[k].get("onsets_ms") or [])
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


def replay_turn(t, env, hop, fs, n_samples, params=dsp.DEFAULTS):
    """The page's SyncTracker.tick loop for one turn, with dsp.align. Returns the ticks."""
    own = b64(t["reference"])
    if t.get("splice"):
        ref, own_off = b64(t["splice"]["values"]), t["splice"]["own_offset"]
    else:
        ref, own_off = own, 0
    if not len(own):
        return []
    ms = lambda v: round(v * fs / 1000)
    onset_prior = t.get("onset_prior_pos")
    locked, last_l, held_no_lock = False, None, 0
    pos = t["start_tap"]
    ticks = []
    hist = round(params.floor_history_ms / HOP_MS)
    while pos <= n_samples:
        pos = -(-pos // BLOCK) * BLOCK
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
            t.get("end_tap") is not None and pos >= t["end_tap"] and t.get("end_pcm_ms") is not None
        )
        if ended and window_start > place + t["end_pcm_ms"] / HOP_MS:
            break
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
                "wide": wide,
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


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def fmt(v, nd=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture_dir")
    ap.add_argument("--causal", action="store_true", help="the browser's causal 30 Hz high-pass")
    ap.add_argument("--thr", help="replay with these thresholds: peak,ratio,sharpness,half_tol_ms")
    ap.add_argument("--features", help="write sim.py stage-A features for thrsearch.py")
    ap.add_argument("--json", help="write the per-turn numbers as JSON")
    args = ap.parse_args()

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
        "| prior err ms | err p50 | err p95 | err max | false | err local p95 | page acc | page err p95 | page false |"
    )
    print(hdr)
    print("|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|")
    results, all_err, all_local, all_page_err, feats = [], [], [], [], []
    for n_turn, t in enumerate(sync["turns"]):
        own = b64(t["reference"])
        tid = f"{n_turn:2d} {t['turn_id'][8:14]}"  # index and the id's hhmmss
        if not len(own):
            print(f"| {tid} | no reference |" + " |" * (hdr.count("|") - 3))
            continue
        ticks = replay_turn(t, env, hop, fs, len(x), params)
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
        if l_truth is not None:
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
            "onsets": onset_rows,
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
            f"| {fmt(drift, 1)} | {len(held)}/{len(acc)} | {fmt(first_lock, 0)} | {fmt(offset, 1)} | {fmt(prior_err, 1)} | {fmt(pct(errs, 50), 2)} "
            f"| {fmt(pct(errs, 95), 2)} | {fmt(max(errs) if errs else None, 2)} | {false} | {fmt(pct(local, 95), 2)} | {len(page_acc)} "
            f"| {fmt(pct(page_err, 95), 2)} | {res['page_false']} |"
        )
        if args.features and l_truth is not None:
            feats += stage_a_features(t, env, hop, fs, l_truth)

    print("\nheld ticks by failed check (a tick can fail several):")
    for r in results:
        if r["held_reasons"]:
            print(f"  {r['turn_id']}: {r['held_reasons']}")

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

    locked_turns = [r for r in results if r["accepted"] or r["held"]]
    speech_turns = [r for r in locked_turns if (r["rx_ms"] or 0) >= 300]
    fl = [r["first_lock_ms"] for r in speech_turns]
    within = sum(v is not None and v <= 300 for v in fl)
    total_false = sum(r["false_locks"] for r in results)
    p95 = pct(all_err, 95)
    print("\n## Design 4.1 minimum\n")
    print(
        f"turns with speech ticks: {len(locked_turns)}; with >= 300 ms of received speech: {len(speech_turns)}"
    )
    print(
        f"accepted ticks: {len(all_err)}; false locks (> 20 ms): {total_false}  -> {'PASS' if total_false == 0 else 'FAIL'}"
    )
    print(
        f"first lock within 300 ms of speech: {within}/{len(speech_turns)} = "
        f"{within / max(1, len(speech_turns)):.1%} (>= 95 %) -> {'PASS' if within >= 0.95 * len(speech_turns) else 'FAIL'}"
    )
    print(
        f"  first lock ms: p50 {fmt(pct([v for v in fl if v is not None], 50), 0)}, "
        f"p95 {fmt(pct([v for v in fl if v is not None], 95), 0)}, never locked {sum(v is None for v in fl)}"
    )
    print(
        f"error p50 {fmt(pct(all_err, 50), 2)} ms, p95 {fmt(p95, 2)} ms (<= 5) -> {'PASS' if p95 <= 5 else 'FAIL'}"
    )
    print(
        f"  against the local (phrase) truth: p50 {fmt(pct(all_local, 50), 2)} ms, p95 {fmt(pct(all_local, 95), 2)} ms, "
        f"max {fmt(max(all_local) if all_local else None, 2)} ms"
    )
    pe = [abs(r["prior_error_ms"]) for r in speech_turns if r["prior_error_ms"] is not None]
    print(
        f"  before the first lock the frames follow the prior; |prior - truth|: p50 {fmt(pct(pe, 50), 1)} ms, "
        f"max {fmt(max(pe) if pe else None, 1)} ms"
    )
    print(
        f"page's own ticks (causal high-pass, live): accepted {len(all_page_err)}, "
        f"false {sum(e > FALSE_LOCK_MS for e in all_page_err)}, err p95 {fmt(pct(all_page_err, 95), 2)} ms"
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


if __name__ == "__main__":
    main()
