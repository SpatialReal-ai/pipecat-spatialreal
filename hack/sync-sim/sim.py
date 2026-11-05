"""Audio-fingerprint alignment simulation: R (the server's audio) -> Opus/WebRTC -> X (what the
browser receives); align X to R with an envelope fingerprint and compare with the onset heuristic.

usage: python3 sim.py OUT.pkl [--quick] [--captures DIR [--runs A,B,...]] [--jobs N]
                      [--thr C,R,S,TOL ...]
       python3 sim.py --stall OUT.json [--jobs N] [--seeds K]   (the stall rule, see below)

* Sources: ``clean/*.wav`` (24 kHz mono, from ``make_speech.sh``), plus, with ``--captures``, real
  bot turns cut out of capture directories (each ``RUN/capture.wav`` + ``RUN/capture.jsonl``, the
  probe's recording format; ``--runs`` names the runs and fixes their order, default: all, sorted --
  every run's random draws are seeded by its position, so the order matters for reproducing a
  published table). ``--quick`` takes a small subset.
* Channel: Opus at 32 and 24 kbps (``codec.py``) under ten conditions (gain, DC offset, noise,
  clock drift, deleted / concealed / inserted gaps), an unknown 0-400 ms delay and a 1 s pre-roll.
* Receiver: the rule of ``dsp.py`` (``dsp.features`` / ``dsp.passes``) replayed every 50 ms tick
  and, as the shipped cadence, every 200 ms. ``--thr`` lists the threshold sets (peak, ratio,
  sharpness, half-window tolerance in seconds) to replay; the per-tick features of every run are
  stored for ``thrsearch.py``.
* ``SYNC_SIM_WIN`` (seconds, default 1.0) sets the correlation window.

The lag convention here is relative to the turn message's arrival (lag 0 = the reference starts
when the message arrives, i.e. ``PRE`` into the track); ``dsp.align`` uses the received
envelope's own index axis (the same thing shifted by ``PRE``).
"""

import glob
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import dsp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FSX = 48000  # browser track rate
FSR = 24000  # server PCM rate

DB_LO = dsp.DB_MIN  # 8-bit quantisation range (dBFS), step 0.376 dB

# fingerprints: name -> (hop_s, bands)
FPS = {
    "F1": (0.005, [(None, None)]),
    "F2": (0.010, [(None, None)]),
    "F3": (0.010, [(None, 1000), (1000, 4000)]),
}
TICK = 0.05  # feature ticks every 50 ms; the 200 ms rule uses every 4th
WIN = float(
    os.environ.get("SYNC_SIM_WIN", "1.0")
)  # correlation window: last WIN s of received audio
SEARCH = 0.30  # +/- search around the prior
PRIOR0 = 0.20  # first prior: 200 ms after the turn message
WIDE = (-0.45, 0.95)  # lag range for which curves are computed (covers every prior)
PRE = 1.0  # pre-roll: the track exists before the turn message; message arrives at X time PRE

CONDS = {
    "clean": {},
    "gain+6": {"gain_db": 6},
    "gain-6": {"gain_db": -6},
    "dc": {"dc": 150},
    "noise20": {"snr": 20},
    "drift": {"drift": 0.001},
    "gap-del": {"gap": "del"},
    "gap-plc": {"gap": "plc"},
    "gap-ins": {"gap": "ins"},
    "all": {"gain_db": "rand", "dc": 150, "snr": 20, "drift": "rand", "gap": "ins"},
}
BITRATES = (32, 24)


# ---------------------------------------------------------------- sources


def onsets(x, fs, thr=328, sil=0.15):
    idx = np.flatnonzero(np.abs(x) > thr)
    if len(idx) == 0:
        return np.array([], int)
    gaps = np.diff(np.concatenate([[-(10**12)], idx]))
    return idx[gaps > sil * fs]


def load_sources(quick=False, captures=None, runs=None):
    srcs = []
    for f in sorted(glob.glob(os.path.join(HERE, "clean", "*.wav"))):
        r, fs = dsp.read_wav(f)
        assert fs == FSR
        srcs.append(("tts", os.path.basename(f)[:-4], r))
    runs = runs or (
        sorted(
            d
            for d in os.listdir(captures)
            if os.path.exists(os.path.join(captures, d, "capture.wav"))
        )
        if captures
        else []
    )
    for run in runs:
        x, fs = dsp.read_wav(os.path.join(captures, run, "capture.wav"))
        ev = []
        with open(os.path.join(captures, run, "capture.jsonl")) as fh:
            lines = fh.readlines()
        for line in lines:
            j = json.loads(line)
            m = j["message"]
            d = m.get("data", {})
            kind = m.get("type")
            if isinstance(d, dict):
                for k in ("turn_start", "turn_end", "turn_abort"):
                    if k in d:
                        kind = k
            ev.append((j["tap"], kind))
        starts = [t for t, k in ev if k == "turn_start"]
        for n, s in enumerate(starts):
            ends = [
                t + int(0.3 * fs) if k == "turn_end" else t
                for t, k in ev
                if t > s
                and k
                in (
                    "turn_end",
                    "turn_start",
                    "bot-interrupted",
                    "user-started-speaking",
                    "turn_abort",
                )
            ]
            e = min(ends) if ends else len(x)
            seg = x[s:e]
            if len(seg) < fs * 1.0 or len(onsets(seg, fs)) == 0:
                continue
            r = dsp.fft_resample(seg, len(seg) // 2)
            srcs.append(("real", f"{run}#{n}", r))
    if quick:
        srcs = srcs[:6] + [s for s in srcs if s[0] == "real"][:4]
    return srcs


# ---------------------------------------------------------------- channel


def pink(n, rng):
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.arange(len(W))
    W[1:] /= np.sqrt(f[1:])
    W[0] = 0
    p = np.fft.irfft(W, n)
    return p / p.std()


def build_x(r, kbps, cond, rng, pkt_cache):
    """Returns X (48 kHz float, int16 scale), rtime (R seconds per X sample, NaN = none),
    events dict."""
    import codec  # here, not at the top: it loads libopus, which the oracle test does not need

    p = dict(CONDS[cond])
    if p.get("gain_db") == "rand":
        p["gain_db"] = rng.choice([-6, 6])
    if p.get("drift") == "rand":
        p["drift"] = rng.choice([-0.001, 0.001])
    x48 = dsp.fft_resample(r, 2 * len(r))
    key = kbps
    if key not in pkt_cache:
        pkt_cache[key] = codec.encode(np.round(x48).clip(-32768, 32767).astype(np.int16), kbps)
    packets, pre = pkt_cache[key]
    ev = {}
    loud = np.flatnonzero(np.abs(r) > 328)
    span = (loud[0] / FSR, loud[-1] / FSR) if len(loud) else (0, len(r) / FSR)
    gap = p.get("gap")
    gpos = span[0] + rng.uniform(0.35, 0.65) * (span[1] - span[0])  # R seconds
    glen = rng.uniform(0.040, 0.120)
    drop, ins = (), None
    if gap in ("plc", "ins"):
        npk = max(2, round(glen / 0.02))
        glen = npk * 0.02
        i0 = int((gpos * FSX + pre) // codec.FRAME)
        if gap == "plc":
            drop = range(i0, i0 + npk)
        else:
            ins = (i0, npk)
    pcm, rt = codec.decode(packets, pre, len(x48), plc_drop=drop, plc_insert=ins)
    rtime = np.where(rt >= 0, rt / FSX, np.nan)
    if gap == "del":
        a = int(gpos * FSX)
        b = a + int(glen * FSX)
        pcm = np.concatenate([pcm[:a], pcm[b:]])
        rtime = np.concatenate([rtime[:a], rtime[b:]])
    if gap:
        ev["gap"] = gap
        ev["gap_len"] = glen
        ev["gap_r"] = gpos
    # unknown delay
    d = rng.uniform(0, 0.4) + rng.uniform(0, 1) / FSX
    ds = d * FSX
    ni = int(np.floor(ds))
    fr = ds - ni
    pcm = dsp.frac_delay(pcm, ds)
    rtime = np.concatenate([np.full(ni, np.nan), rtime - fr / FSX, np.full(64, np.nan)])
    rtime = rtime[: len(pcm)]
    ev["d"] = d
    # clock drift
    e = p.get("drift", 0)
    if e:
        n_new = round(len(pcm) * (1 + e))
        pcm = dsp.fft_resample(pcm, n_new)
        pos = np.arange(n_new) / (1 + e)
        valid = ~np.isnan(rtime)
        rt0 = np.where(valid, rtime, 0)
        rtn = np.interp(pos, np.arange(len(rtime)), rt0)
        vn = np.interp(pos, np.arange(len(rtime)), valid.astype(float))
        rtime = np.where(vn > 0.999, rtn, np.nan)
        ev["drift"] = e
    # pre-roll (track audio before the turn message) and trailing silence
    npre = int(PRE * FSX)
    pcm = np.concatenate([np.zeros(npre), pcm, np.zeros(FSX // 2)])
    rtime = np.concatenate([np.full(npre, np.nan), rtime, np.full(FSX // 2, np.nan)])
    if p.get("gain_db"):
        pcm = pcm * 10 ** (p["gain_db"] / 20)
        ev["gain_db"] = p["gain_db"]
    if p.get("snr"):
        fr_ = len(pcm) // 480
        ms = (pcm[: fr_ * 480].reshape(fr_, 480) ** 2).mean(1)
        act = ms[ms > ms.max() * 1e-3]
        nz = pink(len(pcm), rng) * np.sqrt(act.mean() / 10 ** (p["snr"] / 10))
        pcm = pcm + nz
    if p.get("dc"):
        pcm = pcm + p["dc"]
    # event time on the X timeline
    t = np.arange(len(pcm)) / FSX
    off = t - PRE - rtime  # offset relative to the turn message
    if gap in ("del", "ins"):
        v = np.flatnonzero(~np.isnan(off))
        jump = np.flatnonzero(np.abs(np.diff(off[v])) > 0.01)
        ev["gap_t"] = t[v[jump[0] + 1]] if len(jump) else None
        ev["pre"] = off[v[jump[0]]] if len(jump) else None
        ev["post"] = off[v[jump[0] + 1]] if len(jump) else None
    elif gap == "plc":
        k = np.flatnonzero(~np.isnan(rtime) & (rtime >= drop[0] * codec.FRAME / FSX - pre / FSX))
        ev["gap_t"] = t[k[0]]
        ev["pre"] = ev["post"] = off[k[0]]
    return pcm, rtime, ev


# ---------------------------------------------------------------- fingerprints


def fingerprint(r, name):
    """Quantised envelope(s) of the server's audio ``r`` (F1 is the production envelope)."""
    hop, bands = FPS[name]
    return [dsp.quantise(dsp.envelope_db(r, FSR, hop, b)) for b in bands]


# ---------------------------------------------------------------- receiver


def tick_curves(fp_q, xenv, hop, t_ticks, win, end_off=0.0):
    """For each tick t, NCC curve over WIDE lags for the X window [t-end_off-win, t-end_off].
    The silence floor comes from the receiver's last 3 s (up to t).  Returns lags (s),
    curves [n_ticks, n_lags] (NaN where the window has no speech) and the speech gate."""
    nb = len(fp_q)
    fp = [dsp.dequantise(q) for q in fp_q]
    M = round(win / hop)
    E = round(end_off / hop)
    l0, l1 = int(np.floor(WIDE[0] / hop)), int(np.ceil(WIDE[1] / hop))
    lags = np.arange(l0, l1 + 1)
    pre_f = round(PRE / hop)
    P = M + E + abs(l0) + abs(l1) + pre_f + 10
    fpp = [dsp.pad_reference(f, P) for f in fp]
    curves = np.full((len(t_ticks), len(lags)), np.nan)
    gate = np.zeros(len(t_ticks), bool)
    nx = len(xenv[0])
    for ti, t in enumerate(t_ticks):
        jt = min(int(np.floor((t - 2 * hop) / hop)) + 1, nx)  # frames fully inside [0, t]
        j1 = jt - E
        j0 = j1 - M
        if j0 < 0:
            continue
        acc = np.zeros(len(lags))
        ok = True
        for b in range(nb):
            floor = dsp.floor_db(xenv[b][max(0, jt - int(3 / hop)) : jt])
            c = dsp.ncc_curve(xenv[b][j0:j1], fpp[b], j0 - pre_f + P, lags, floor)
            if c is None:
                ok = False
                break
            acc += c
        if not ok:
            continue
        gate[ti] = True
        curves[ti] = acc / nb
    return lags * hop, curves, gate


SHARP_AT = 0.020  # sharpness = peak - max(curve at peak +/- 20 ms)
HALF_SEARCH = 0.040
PARAMS = dsp.AlignParams(sharp_at_ms=SHARP_AT * 1000, half_search_ms=HALF_SEARCH * 1000)


def features(lags, C, G, ti, lo, hi):
    """Acceptance features of tick ti for a search range [lo, hi] (seconds).
    C/G: curves and gates for 'full' (last WIN s) and the halves 'h1' (older) / 'h2' (newer).
    Returns (peak, ratio, sharpness, half agreement, estimate, interpolated estimate, edge),
    times in seconds."""
    halves = [C[h][ti] if G[h][ti] else None for h in ("h1", "h2")]
    f = dsp.features(lags * 1000, C["full"][ti], halves, lo * 1000, hi * 1000, PARAMS)
    return (
        f.peak,
        f.ratio,
        f.sharpness,
        f.agree_ms / 1000,
        f.est_ms / 1000,
        f.est_interp_ms / 1000,
        f.edge,
    )


def accepted(f, thr):
    """Thresholds ``thr`` = (peak, ratio, sharpness, half tolerance in seconds)."""
    cmin, rmin, smin, tol = thr
    return all(
        dsp.passes(
            dsp.Features(*f[:3], f[3] * 1000, f[4] * 1000, f[5] * 1000, f[6]),
            cmin,
            rmin,
            smin,
            tol * 1000,
        )
    )


def truth_at(t, toff_t, toff_v):
    k = np.searchsorted(toff_t, t)
    k = np.clip(k, 0, len(toff_t) - 1)
    k2 = np.clip(k - 1, 0, len(toff_t) - 1)
    return np.where(np.abs(toff_t[k] - t) < np.abs(toff_t[k2] - t), toff_v[k], toff_v[k2])


def tick_error(est, ti, truth_c, straddle, ev):
    if straddle[ti]:
        return min(abs(est - ev["pre"]), abs(est - ev["post"]))
    return abs(est - truth_c[ti])


def replay(lags, C, G, t_ticks, step, thr, truth_c, truth_now, straddle, ev, onset_x):
    """Online rule: every `step` ticks, search +/-SEARCH around the prior (first prior: the
    turn message + PRIOR0; then the last accepted value); accept only a sharp, self-consistent
    peak; otherwise HOLD the last alignment."""
    prior = PRIOR0
    applied = None
    acc_err, acc_err_i, app_err = [], [], []
    n_gate = n_acc = n_false = n_hold = 0
    first_lock = None
    relock = None
    for ti in range(0, len(t_ticks), step):
        t = t_ticks[ti]
        if G["full"][ti]:
            n_gate += 1
            f = features(lags, C, G, ti, prior - SEARCH, prior + SEARCH)
            if accepted(f, thr):
                est, est_i = f[4], f[5]
                n_acc += 1
                e = tick_error(est_i, ti, truth_c, straddle, ev)
                if not straddle[ti]:
                    acc_err.append(abs(est - truth_c[ti]))
                    acc_err_i.append(e)
                if e > 0.020:
                    n_false += 1
                prior = est_i
                applied = est_i
                if first_lock is None and e <= 0.020:
                    first_lock = t - onset_x
            else:
                n_hold += 1
        if applied is not None and not np.isnan(truth_now[ti]):
            app_err.append(abs(applied - truth_now[ti]))
            if (
                ev.get("gap_t") is not None
                and ev["gap"] in ("del", "ins")
                and t > ev["gap_t"]
                and relock is None
                and abs(applied - ev["post"]) <= 0.005
            ):
                relock = t - ev["gap_t"]
    return {
        "n_gate": n_gate,
        "n_acc": n_acc,
        "n_false": n_false,
        "n_hold": n_hold,
        "acc_err": np.array(acc_err),
        "acc_err_i": np.array(acc_err_i),
        "app_err": np.array(app_err),
        "first_lock": first_lock,
        "relock": relock,
    }


def onset_replay(onR, onX, toff_t, toff_v):
    """Today's heuristic over every server onset: the first pair sets the alignment, each later
    server onset is matched to the first received onset in [expected-60 ms, expected+200 ms]
    and re-aligns the turn.  Returns per-onset signed errors (None = no match)."""
    onX = onX[onX >= PRE * FSX]
    if not len(onR) or not len(onX):
        return None
    A = onX[0] / FSX - PRE - onR[0] / FSR
    errs = [float(A - truth_at(onX[0] / FSX, toff_t, toff_v))]
    for k in range(1, len(onR)):
        exp = PRE + onR[k] / FSR + A
        tx = onX / FSX
        cand = tx[(tx >= exp - 0.06) & (tx <= exp + 0.2)]
        if not len(cand):
            errs.append(None)
            continue
        a_k = cand[0] - PRE - onR[k] / FSR
        errs.append(float(a_k - truth_at(cand[0], toff_t, toff_v)))
        A = a_k
    return errs


def run_one(args):
    _, kind, name, r, kbps, cond, seed, thr_list = args
    rng = np.random.default_rng(seed)
    cache = {}
    x, rtime, ev = build_x(r, kbps, cond, rng, cache)
    t_all = np.arange(len(x)) / FSX
    off = t_all - PRE - rtime
    v = ~np.isnan(off)
    toff_t, toff_v = t_all[v], off[v]
    onR = onsets(r, FSR)
    onX = onsets(x, FSX)
    f4_all = onset_replay(onR, onX, toff_t, toff_v)
    f4 = f4_all[0] if f4_all else None
    onset_x = PRE + ((onR[0] / FSR + ev["d"]) if len(onR) else ev["d"])
    out = {
        "kind": kind,
        "name": name,
        "kbps": kbps,
        "cond": cond,
        "ev": ev,
        "f4": f4,
        "f4_all": f4_all,
        "n_on_r": len(onR),
        "dur": len(x) / FSX,
        "res": {},
        "feat": {},
    }
    t_ticks = PRE + np.arange(1, int((len(x) / FSX - PRE) / TICK) + 1) * TICK
    tc = np.maximum(t_ticks - WIN / 2, PRE)
    truth_c = truth_at(tc, toff_t, toff_v)
    truth_now = truth_at(t_ticks, toff_t, toff_v)
    truth_now = np.where(t_ticks >= toff_t[0], truth_now, np.nan)
    straddle = np.zeros(len(t_ticks), bool)
    if ev.get("gap_t") is not None and ev["gap"] in ("del", "ins"):
        straddle = (t_ticks - WIN <= ev["gap_t"]) & (t_ticks >= ev["gap_t"])
    for fpn, (hop, bands) in FPS.items():
        fpq = fingerprint(r, fpn)
        xenv = [dsp.envelope_db(x, FSX, hop, b, dc_block=True) for b in bands]
        C, G = {}, {}
        lags, C["full"], G["full"] = tick_curves(fpq, xenv, hop, t_ticks, WIN)
        _, C["h1"], G["h1"] = tick_curves(fpq, xenv, hop, t_ticks, WIN / 2, WIN / 2)
        _, C["h2"], G["h2"] = tick_curves(fpq, xenv, hop, t_ticks, WIN / 2, 0.0)
        # stage A: features with the fixed first prior, every tick (for threshold search)
        F = np.full((len(t_ticks), 9), np.nan)
        for ti in range(len(t_ticks)):
            if not G["full"][ti]:
                continue
            f = features(lags, C, G, ti, PRIOR0 - SEARCH, PRIOR0 + SEARCH)
            F[ti, :7] = f
            F[ti, 7] = tick_error(f[5], ti, truth_c, straddle, ev)
            F[ti, 8] = straddle[ti]
        out["feat"][fpn] = F.astype(np.float32)
        # stage B: the online rule for the given thresholds
        for thr in thr_list:
            for step, sname in ((4, "200ms"), (1, "50ms")):
                out["res"][(fpn, thr, sname)] = replay(
                    lags, C, G, t_ticks, step, thr, truth_c, truth_now, straddle, ev, onset_x
                )
    return out


# ---------------------------------------------------------------- stall conditions (the stall rule)
#
# ``sim.py --stall OUT.json``: the page's tick loop (``replay_capture.replay_turn_tracked``) on one
# synthetic turn per run, with exact truth, today's rule against the stall rule. Sources: the real
# bot turns of the committed fixtures (``STALL_RUNS``), so a run needs no capture outside the repo.

STALL_FIXTURES = os.path.join(HERE, "..", "..", "tests", "fixtures", "sync")
STALL_RUNS = ["daily-clean", "livekit-clean", "daily-splice"]
STALL_CONDS = (
    ["clean"]
    + [f"stall-{s}" for s in (80, 150, 250, 500, 1200)]
    + [f"zeros-{s}" for s in (150, 250, 500)]
    + [f"loss-{d}" for d in (200, 400)]
    + ["ins", "del"]
)
STALL_HOP = FSX * dsp.HOP_MS // 1000  # received frame, samples at 48 kHz
STALL_SEEDS = 3


def build_stall_x(r, cond, rng):
    """One synthetic received track for a stall condition.

    Channel: Opus 32 kbps 20 ms (``codec.py``) and

    * ``clean``;
    * ``stall-S``: receiver-side stall of S ms (packets late): the jitter buffer inserts S ms
      (rounded to 20 ms), the decoder's concealment for the first P ms (P ~ U(60, 130), as seen in
      the stalled Chrome captures) then exact zeros, and L ~ U(0, 60) ms of content lost just
      before it (concealed in place, timeline kept); the timeline shifts by S;
    * ``zeros-S``: sender-side: S ms of digital zeros inserted into the source before encoding;
      shift S;
    * ``loss-D``: D ms of packets lost (burst loss), concealment P ms then zeros, timeline kept;
    * ``ins``: 40-120 ms of decoder concealment inserted (a jitter-buffer expand); shift > 0;
    * ``del``: 40-120 ms deleted; shift < 0.

    The impairment sits 35-65 % into the turn's speech span; then an unknown 0-400 ms delay, 1 s of
    pre-roll and 1 s of trailing silence. Returns ``(x, rt, ev)``: the 48 kHz track, per sample the
    source sample it carries (-1: none, -2: concealment in place of lost content) and the event.
    """
    import codec  # libopus, loaded only for the simulation

    loud = np.flatnonzero(np.abs(r) > 328)
    span = (loud[0] / FSR, loud[-1] / FSR)
    gpos = span[0] + rng.uniform(0.35, 0.65) * (span[1] - span[0])  # R seconds
    kind, _, arg = cond.partition("-")
    x48 = dsp.fft_resample(r, 2 * len(r))
    ev = {"cond": cond, "shift_ms": 0}
    if kind == "zeros":
        n_z = int(int(arg) / 1000 * FSX)
        a = int(gpos * FSX)
        x48 = np.concatenate([x48[:a], np.zeros(n_z), x48[a:]])
        src = np.concatenate([np.arange(a), np.full(n_z, -1), np.arange(a, len(x48) - n_z)])
        packets, pre = codec.encode(np.round(x48).clip(-32768, 32767).astype(np.int16), 32)
        pcm, rt = codec.decode(packets, pre, len(x48))
        pcm = pcm.astype(np.float64)
        rt = np.where(rt >= 0, src[np.clip(rt, 0, len(src) - 1)], -1)
        ev["shift_ms"] = int(arg)
    else:
        packets, pre = codec.encode(np.round(x48).clip(-32768, 32767).astype(np.int16), 32)
        i0 = int((gpos * FSX + pre) // codec.FRAME)
        drop, ins, zero_after = (), None, None
        if kind == "stall":
            plc = rng.uniform(60, 130)
            nl = round(rng.uniform(0, 60) / 20)
            drop = range(i0 - nl, i0)
            ins = (i0, max(1, round(int(arg) / 20)))
            zero_after = plc
            ev.update(shift_ms=ins[1] * 20, lost_ms=nl * 20, plc_ms=plc)
        elif kind == "loss":
            drop = range(i0, i0 + round(int(arg) / 20))
            zero_after = rng.uniform(60, 130)
            ev["plc_ms"] = zero_after
        elif kind == "ins":
            n = max(2, round(rng.uniform(0.04, 0.12) / 0.02))
            ins = (i0, n)
            ev["shift_ms"] = n * 20
        pcm, rt = codec.decode(packets, pre, len(x48), plc_drop=drop, plc_insert=ins)
        pcm = pcm.astype(np.float64)
        if zero_after is not None:
            # NetEq-like: the concealment fades for zero_after ms, then digital silence
            if kind == "stall":
                bad = np.flatnonzero(rt < 0)
            else:
                lo_s = drop[0] * codec.FRAME - pre
                hi_s = drop[-1] * codec.FRAME + codec.FRAME - pre
                bad = np.flatnonzero((rt >= lo_s) & (rt < hi_s))
                rt = np.where((rt >= lo_s) & (rt < hi_s), -2, rt)
            if len(bad):
                start = bad[0]
                if kind == "stall" and ev.get("lost_ms"):
                    start = max(0, bad[0] - int(ev["lost_ms"] / 1000 * FSX))
                z0 = start + int(zero_after / 1000 * FSX)
                if z0 <= bad[-1]:
                    pcm[z0 : bad[-1] + 1] = 0.0
        if kind == "stall" and ev.get("lost_ms"):
            lo_s = (i0 - round(ev["lost_ms"] / 20)) * codec.FRAME - pre
            hi_s = i0 * codec.FRAME - pre
            rt = np.where((rt >= lo_s) & (rt < hi_s), -2, rt)
        if kind == "del":
            glen = rng.uniform(0.040, 0.120)
            a = int(gpos * FSX)
            b = a + int(glen * FSX)
            pcm = np.concatenate([pcm[:a], pcm[b:]])
            rt = np.concatenate([rt[:a], rt[b:]])
            ev["shift_ms"] = -glen * 1000
    pre_n = FSX + int(rng.uniform(0, 0.4) * FSX)
    x = np.concatenate([np.zeros(pre_n), pcm, np.zeros(FSX)])
    rt = np.concatenate([np.full(pre_n, -1), rt, np.full(FSX, -1)])
    return x, rt.astype(np.float64), ev


def stall_truth_frames(rt, n_env):
    """Per received frame ``j``: the received frame of the reference's index 0 its content puts
    there (NaN where the frame's centre carries no source content)."""
    idx = np.clip(np.arange(n_env) * STALL_HOP + STALL_HOP, 0, len(rt) - 1)
    src = rt[idx]
    ok = src >= 0
    out = np.full(n_env, np.nan)
    out[ok] = np.arange(n_env)[ok] - (src[ok] - STALL_HOP) / STALL_HOP
    return out


def build_stall_case(name, r, cond, seed):
    """Everything a stall run needs, from its seed: track envelope (causal high-pass, as the
    browser), truth per frame, the fake turn (prior: the truth +/- U(0, 100) ms) and the event's
    frames (``jump``: last frame before / first frame after the impairment)."""
    import replay_capture as rc

    rng = np.random.default_rng(seed)
    x, rt, ev = build_stall_x(r, cond, rng)
    env = dsp._window_db(rc.causal_high_pass(x, FSX), STALL_HOP, 2 * STALL_HOP)
    tl = stall_truth_frames(rt, len(env))
    ref = dsp.envelope_bytes(np.round(r).clip(-32768, 32767), FSR)
    valid = np.flatnonzero(~np.isnan(tl))
    prior_l = tl[valid[0]] + rng.uniform(-100, 100) / dsp.HOP_MS
    start_tap = round(prior_l * STALL_HOP) - int(0.2 * FSX)
    t = {
        "turn_id": f"sim-{name}-{cond}-{seed}",
        "start_tap": start_tap,
        "end_tap": start_tap + int(0.1 * FSX),
        "end_pcm_ms": len(r) / FSR * 1000,
        "abort_tap": None,
        "ticks": [],
        "onsets_ms": [],
    }
    arrivals = {t["turn_id"]: [(start_tap - 1, 0, ref)]}
    jump = None
    if cond.startswith("loss"):
        gaps = np.flatnonzero(np.diff(valid) > 1)
        if len(gaps):
            jump = (int(valid[gaps[0]]), int(valid[gaps[0] + 1]))
    elif ev["shift_ms"]:
        big = np.flatnonzero(np.abs(np.diff(tl[valid])) > 2)
        if len(big):
            jump = (int(valid[big[0]]), int(valid[big[0] + 1]))
    return {
        "name": name,
        "cond": cond,
        "seed": seed,
        "ev": ev,
        "env": env.astype(np.float32),
        "tl": tl,
        "n_samples": len(x),
        "turn": t,
        "arrivals": arrivals,
        "jump": jump,
    }


def window_false(l, truth, tol_ms=20.0):
    """Whether an accept at ``l`` is false against the truth of the frames its window covered:
    ``(error_ms, straddles)``. A window whose truth spans more than 10 ms straddles a gap; its
    accept must land on one of the real positions inside it (the nearest counts)."""
    seg = truth[~np.isnan(truth)]
    if not len(seg):
        return None, False
    if (seg.max() - seg.min()) * dsp.HOP_MS <= 10:
        return abs(l - seg.mean()) * dsp.HOP_MS, False
    return float(np.min(np.abs(seg - l)) * dsp.HOP_MS), True


def evaluate_stall_case(case, stall, params=dsp.DEFAULTS):
    """Replay one case with ``stall`` and score it against the exact truth."""
    import replay_capture as rc

    env = case["env"].astype(np.float64)
    tl = case["tl"]
    hop = STALL_HOP
    res = rc.replay_turn_tracked(
        case["turn"], env, hop, FSX, case["n_samples"], params, {}, case["arrivals"], None, stall
    )
    disp = res["disp"]
    js = np.array(sorted(disp))
    dl = np.array([disp[j] for j in js])
    acc = [k for k in res["ticks"] if k["accepted"]]
    false, false_straddle, scored, worst = [], [], 0, 0.0
    accepts = [(k["count"], k["l"], 100 if k["via"] == "half" else 200, k["via"]) for k in acc]
    for f in res["freezes"]:
        if f["how"] == "confirm":
            accepts.append(
                (f["closed"] + 1, f["confirm_l"], f["closed"] + 1 - f["resume"], "confirm")
            )
    for count, l, m, via in accepts:
        e, straddles = window_false(l, tl[max(0, count - m) : count])
        if e is None:
            continue
        scored += 1
        if not straddles:
            worst = max(worst, e)
        if e > 20:
            (false_straddle if straddles else false).append((via, round(e, 1)))
    first = next((k["count"] for k in acc), None)
    off20_post = 0.0
    if first is not None:
        post = (js >= first) & ~np.isnan(tl[js])
        off20_post = float((np.abs(dl[post] - tl[js[post]]) * dsp.HOP_MS > 20).sum() * dsp.HOP_MS)
    out = {
        "off20_post_ms": off20_post,
        "n_acc": len(acc),
        "n_half": sum(k["via"] == "half" for k in acc),
        "scored": scored,
        "false": false,
        "false_straddle": false_straddle,
        "worst_ms": worst,
        "lost": len(res["lost_at"]),
        "freezes": [(f["frames"] * dsp.HOP_MS, f["how"]) for f in res["freezes"]],
        "first_lock": first,
    }
    jump = case["jump"]
    if jump is not None and first is not None and first < jump[0]:
        jp, jr = jump
        post = js >= jr
        jj, dd, tt = js[post], dl[post], tl[js[post]]
        t_re = None
        for i in range(len(jj)):
            if np.isnan(tt[i]):
                continue
            win = slice(i, i + 20)
            ok = [
                abs(a - b) * dsp.HOP_MS <= 10 for a, b in zip(dd[win], tt[win]) if not np.isnan(b)
            ]
            if ok and all(ok):
                t_re = int((jj[i] - jr) * dsp.HOP_MS)
                break
        ins = (js > jp) & (js < jr)
        k_gap = jp - tl[jp]
        ahead = float(np.max((js[ins] - dl[ins]) - k_gap) * dsp.HOP_MS) if ins.any() else 0.0
        err = np.abs((dd - tt) * dsp.HOP_MS)
        err = err[~np.isnan(err)]
        valid = np.flatnonzero(~np.isnan(tl))
        out["event"] = {
            "realign_ms": t_re,
            "ahead_ms": ahead,
            "max_after_ms": float(err.max()) if len(err) else None,
            "off20_after_ms": float((err > 20).sum() * dsp.HOP_MS) if len(err) else 0.0,
            "speech_after_ms": int((valid[-1] - jr) * dsp.HOP_MS),
        }
    return out


def stall_sources():
    """The real bot turns of the committed fixtures (``STALL_RUNS``)."""
    return [s for s in load_sources(captures=STALL_FIXTURES, runs=STALL_RUNS) if s[0] == "real"]


def stall_tasks(seeds=STALL_SEEDS, conds=STALL_CONDS):
    """``(name, r, cond, seed)`` for every source x condition x seed; the seed depends only on
    the source's and the condition's positions, so a run is reproducible."""
    tasks = []
    for si, (_, name, r) in enumerate(stall_sources()):
        for cond in conds:
            ci = STALL_CONDS.index(cond)
            for k in range(seeds):
                tasks.append((name, r, cond, 1000 * si + 10 * ci + k))
    return tasks


def _stall_build(task):
    return build_stall_case(*task)


def build_stall_cases(jobs=10, seeds=STALL_SEEDS):
    """Every stall case, built in parallel."""
    with Pool(jobs) as pool:
        return pool.map(_stall_build, stall_tasks(seeds), chunksize=4)


def summarise_stall(rows):
    """Per condition: runs, runs with sync-lost, re-align p50/p90 and never (events with >= 1.2 s
    of speech after the impairment), the face ahead during the gap p50/p90, the error after it
    p50/max and ms off by > 20 ms p50/p90, false locks (outside a gap / straddling it with no real
    position), freezes, confirmed freezes, newer-half accepts."""
    out = {}
    for cond in STALL_CONDS:
        rs = [r for r in rows if r["cond"] == cond]
        if not rs:
            continue
        ev = [r["event"] for r in rs if r.get("event") and r["event"]["speech_after_ms"] >= 1200]
        re_ = [e["realign_ms"] for e in ev if e["realign_ms"] is not None]
        q = lambda v, p: float(np.percentile(v, p)) if len(v) else None
        out[cond] = {
            "runs": len(rs),
            "lost": sum(r["lost"] > 0 for r in rs),
            "events": len(ev),
            "realign_p50": q(re_, 50),
            "realign_p90": q(re_, 90),
            "never": sum(e["realign_ms"] is None for e in ev),
            "ahead_p50": q([e["ahead_ms"] for e in ev], 50),
            "ahead_p90": q([e["ahead_ms"] for e in ev], 90),
            "max_after_p50": q(
                [e["max_after_ms"] for e in ev if e["max_after_ms"] is not None], 50
            ),
            "max_after_max": q(
                [e["max_after_ms"] for e in ev if e["max_after_ms"] is not None], 100
            ),
            "off20_p50": q([e["off20_after_ms"] for e in ev], 50),
            "off20_p90": q([e["off20_after_ms"] for e in ev], 90),
            "false": sum(len(r["false"]) for r in rs),
            "false_straddle": sum(len(r["false_straddle"]) for r in rs),
            "freezes": sum(len(r["freezes"]) for r in rs),
            "confirmed": sum(h == "confirm" for r in rs for _, h in r["freezes"]),
            "half": sum(r["n_half"] for r in rs),
            "accepted": sum(r["n_acc"] for r in rs),
        }
    return out


def _stall_eval(args):
    case, stall = args
    out = evaluate_stall_case(case, stall)
    out.update(name=case["name"], cond=case["cond"], seed=case["seed"], ev=case["ev"])
    return out


def stall_main(argv):
    """``sim.py --stall OUT.json [--jobs N] [--seeds K]``: today's rule and the stall rule on every
    stall condition; prints the per-condition table and writes the per-run numbers."""
    outp = argv[argv.index("--stall") + 1]
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 10
    seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else STALL_SEEDS
    t0 = time.time()
    cases = build_stall_cases(jobs, seeds)
    print(f"{len(cases)} cases built in {time.time() - t0:.0f}s", flush=True)
    result = {}
    with Pool(jobs) as pool:
        for mode, stall in (("today", dsp.TODAY), ("stall", dsp.STALL)):
            rows = pool.map(_stall_eval, [(c, stall) for c in cases], chunksize=4)
            result[mode] = {"rows": rows, "summary": summarise_stall(rows)}
    print_stall_table(result)
    with open(outp, "w") as fh:
        json.dump(result, fh, default=float)


def print_stall_table(result):
    """Markdown table of ``{mode: {"summary": summarise_stall(...)}}``."""
    fmt = lambda v: "-" if v is None else f"{v:.0f}"
    print(
        "| condition | rule | runs | sync-lost | re-align p50/p90 ms (never/events) | ahead in gap "
        "p50/p90 ms | max err after p50/max ms | >20 ms after p50/p90 ms | false (outside gap / "
        "straddling) | freezes (confirmed) | half accepts |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cond in STALL_CONDS:
        for mode, res in result.items():
            s = res["summary"].get(cond)
            if not s:
                continue
            print(
                f"| {cond} | {mode} | {s['runs']} | {s['lost']} | {fmt(s['realign_p50'])}/"
                f"{fmt(s['realign_p90'])} ({s['never']}/{s['events']}) | {fmt(s['ahead_p50'])}/"
                f"{fmt(s['ahead_p90'])} | {fmt(s['max_after_p50'])}/{fmt(s['max_after_max'])} | "
                f"{fmt(s['off20_p50'])}/{fmt(s['off20_p90'])} | {s['false']} / {s['false_straddle']} | "
                f"{s['freezes']} ({s['confirmed']}) | {s['half']} |"
            )


def main():
    if "--stall" in sys.argv:
        return stall_main(sys.argv[1:])
    outp = sys.argv[1]
    argv = sys.argv[1:]
    quick = "--quick" in argv
    captures = argv[argv.index("--captures") + 1] if "--captures" in argv else None
    jobs_n = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 10
    thr_list = []
    if "--thr" in argv:
        tail = argv[argv.index("--thr") + 1 :]
        thr_list = [tuple(float(v) for v in a.split(",")) for a in tail if not a.startswith("--")]
    runs = argv[argv.index("--runs") + 1].split(",") if "--runs" in argv else None
    srcs = load_sources(quick, captures, runs)
    print(
        "sources:",
        sum(s[0] == "tts" for s in srcs),
        "tts,",
        sum(s[0] == "real" for s in srcs),
        "real turns",
    )
    jobs = []
    seed = 1
    for si, (kind, name, r) in enumerate(srcs):
        for kbps in BITRATES:
            for cond in CONDS:
                jobs.append((si, kind, name, r, kbps, cond, seed, thr_list))
                seed += 1
    t0 = time.time()
    with Pool(jobs_n) as pool:
        res = pool.map(run_one, jobs, chunksize=4)
    print(f"{len(res)} runs in {time.time() - t0:.0f}s")
    with open(outp, "wb") as fh:
        pickle.dump(res, fh)


if __name__ == "__main__":
    main()
