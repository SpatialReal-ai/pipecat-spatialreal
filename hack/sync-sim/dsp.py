"""The normative definition of the envelope fingerprint and of the alignment rule.

Everything an implementation must agree on is in this file; the web SDK ports it verbatim. The
public entry points:

* :func:`envelope_bytes` -- the 8-bit log-RMS envelope of the audio the plugin releases. The
  package's ``Envelope`` (``src/pipecat_spatialreal/fingerprint.py``) must produce the same bytes;
  ``tests/test_fingerprint_oracle.py`` asserts it.
* :func:`align` -- one alignment tick: where in the received envelope does the turn's reference
  envelope sit, and may we trust it?
* :class:`TurnTracker` -- the stall rule (freeze, confirm, newer-half re-lock, the loss clock) as
  one turn's state on top of :func:`align`, and :func:`chained_prior` (rule 5). Its fixed terms
  are :class:`StallParams` (the defaults, ``STALL``, are ``thrsearch_stall.py``'s choice);
  ``README.md`` ("The stall rule") states them.

The building blocks the simulation (``sim.py``) shares with :func:`align` are public too, so the
simulation and the rule cannot drift apart.

Fixed terms of the rule (changing any of them means re-running ``thrsearch.py``):

* Envelope window: 10 ms, hop 5 ms, value ``10*log10(mean_square / 32768**2 + 1e-10)`` dB. A
  window counts only when fully inside the audio.
* Quantisation: clamp to [-96, 0] dB, ``floor((db + 96) / 96 * 255 + 0.5)`` (round half up).
* Floor: the 5th percentile of the last 3 s of the *received* envelope plus 6 dB, never below
  -70 dB; both envelopes are clipped at it before correlating.
* Speech gate: the clipped received window must have a standard deviation of at least 2 dB.
* Correlation: mean-removed Pearson over the last 1 s of the received envelope, at every lag
  within +/- 300 ms of the prior.
* Second-highest peak: the maximum of the curve outside the main lobe, found by walking down
  both sides of the peak monotonically.
* Half-window check: the same correlation on the older and the newer 0.5 s of the same window
  (same tick), each searched within +/- 40 ms of the full-window peak; a half with no speech is
  skipped; the check holds when every half peaks within 10 ms of the full peak (the full peak
  is the parabolic one) and neither half peaks at the edge of its +/- 40 ms range.
* Sharpness: the interpolated peak value minus the larger of the curve values at the
  interpolated peak +/- 20 ms.
* Parabolic interpolation over the three lags around the peak.
* Thresholds: peak >= 0.95, peak / second peak >= 1.2, sharpness >= 0.03, halves within 10 ms,
  and the peak is not at the edge of the search range.
* Known difference to the browser: :func:`band_mask_filter` (the receiver's 30 Hz high-pass in
  the simulation) is zero phase, applied in the FFT domain. The browser's will be causal.
"""

import wave
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------- constants

FULL_SCALE_SQUARED = 32768.0**2  # int16 magnitude squared: the 0 dB reference
EPSILON = 1e-10  # added to the normalised mean square so silence is finite (-100 dB)
DB_MIN = -96.0  # byte 0
DB_MAX = 0.0  # byte 255
DB_RANGE = DB_MAX - DB_MIN
HOP_MS = 5
WINDOW_MS = 10


# ---------------------------------------------------------------- signal helpers


def read_wav(path):
    """Read a mono 16-bit WAV file; return ``(samples as float64, sample_rate)``."""
    with wave.open(path) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
        return pcm, w.getframerate()


def fft_resample(x, n_out):
    """Band-limited, zero-phase resample of a whole signal to ``n_out`` samples."""
    n = len(x)
    pad = 2048  # reduce wrap-around ringing
    xp = np.concatenate([x, np.zeros(pad)])
    np_out = round((n + pad) * n_out / n)
    X = np.fft.rfft(xp)
    m = np_out // 2 + 1
    Y = np.zeros(m, complex)
    k = min(m, len(X))
    Y[:k] = X[:k]
    y = np.fft.irfft(Y, np_out) * (np_out / (n + pad))
    return y[:n_out]


def frac_delay(x, delay_samples):
    """Delay by a (possibly fractional) number of samples; the output is longer than the input."""
    ni = int(np.floor(delay_samples))
    fr = delay_samples - ni
    n = len(x) + ni + 64
    xp = np.concatenate([np.zeros(ni), x, np.zeros(64)])
    if fr:
        N = 1 << int(np.ceil(np.log2(n + 64)))
        X = np.fft.rfft(xp, N)
        f = np.fft.rfftfreq(N)
        xp = np.fft.irfft(X * np.exp(-2j * np.pi * f * fr), N)[:n]
    return xp


def band_mask_filter(x, fs, lo, hi, order=4):
    """Zero-phase Butterworth-magnitude band filter applied in the FFT domain."""
    N = 1 << int(np.ceil(np.log2(len(x) + 1)))
    X = np.fft.rfft(x, N)
    f = np.fft.rfftfreq(N, 1 / fs)
    H = np.ones_like(f)
    if hi:
        H *= 1 / np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        with np.errstate(divide="ignore"):
            H *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-9)) ** (2 * order))
    return np.fft.irfft(X * H, N)[: len(x)]


# ---------------------------------------------------------------- envelope


def _window_db(x, hop, win):
    """dB level of every window of ``win`` samples that starts at a multiple of ``hop`` and lies
    fully inside ``x``."""
    n = (len(x) - win) // hop + 1
    if n <= 0:
        return np.zeros(0)
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    mean_square = (x[idx] ** 2).mean(1)
    return 10 * np.log10(mean_square / FULL_SCALE_SQUARED + EPSILON)


def envelope_db(x, fs, hop_s=0.005, band=(None, None), dc_block=False):
    """Float dB envelope of ``x`` (int16 scale): windows of ``2 * hop`` every ``hop`` seconds.

    ``band`` and ``dc_block`` filter first (zero phase); ``dc_block`` is the receiver's 30 Hz
    high-pass. The simulation uses this for the received track and for the other fingerprints
    (F2, F3); the production envelope is :func:`envelope_bytes`.
    """
    if dc_block or band != (None, None):
        lo, hi = band
        if dc_block and not lo:
            lo = 30.0
        x = band_mask_filter(x, fs, lo, hi)
    h = round(hop_s * fs)
    return _window_db(x, h, 2 * h)


def quantise(db):
    """Map dB to bytes: clamp to [-96, 0], ``floor((db + 96) / 96 * 255 + 0.5)``.

    This is the package's formula, operation for operation (round half up, clamp first).
    """
    db = np.clip(np.asarray(db, dtype=np.float64), DB_MIN, DB_MAX)
    return np.floor((db - DB_MIN) / DB_RANGE * 255 + 0.5).astype(np.uint8)


def dequantise(q):
    """Map bytes back to dB (0 -> -96, 255 -> 0)."""
    return np.asarray(q).astype(np.float64) * DB_RANGE / 255 + DB_MIN


def envelope_bytes(pcm_int16, sample_rate):
    """The production envelope of mono int16 audio: one byte per 5 ms hop, 10 ms windows.

    Window and hop lengths in samples are ``ms * sample_rate // 1000`` (the package's rule); a
    window counts only when it lies fully inside ``pcm_int16``, so 1 s of audio gives 199 values
    and value ``k`` covers samples ``[k * hop, k * hop + window)``.

    Args:
        pcm_int16: Mono samples, any integer or float array holding int16-range values.
        sample_rate: Samples per second.

    Returns:
        ``np.ndarray`` of ``uint8``.
    """
    hop = HOP_MS * sample_rate // 1000
    win = WINDOW_MS * sample_rate // 1000
    x = np.asarray(pcm_int16).astype(np.float64)
    return quantise(_window_db(x, hop, win))


# ---------------------------------------------------------------- the rule


@dataclass(frozen=True)
class AlignParams:
    """The fixed terms and thresholds of the rule (defaults are the frozen values).

    Lengths are milliseconds; ``hop_ms`` is the envelope's hop.
    """

    hop_ms: float = 5.0
    window_ms: float = 1000.0
    search_ms: float = 300.0
    half_search_ms: float = 40.0
    sharp_at_ms: float = 20.0
    floor_history_ms: float = 3000.0
    floor_percentile: float = 5.0
    floor_margin_db: float = 6.0
    floor_min_db: float = -70.0
    gate_std_db: float = 2.0
    min_peak: float = 0.95
    min_ratio: float = 1.2
    min_sharpness: float = 0.03
    half_tol_ms: float = 10.0
    interpolate: bool = True


@dataclass(frozen=True)
class AlignChecks:
    """What one :func:`align` tick measured and which of the checks held.

    The measurements are ``None`` when the tick got no further than the gate. ``candidate_ms`` is
    the interpolated peak whether or not it was accepted.
    """

    speech: bool
    window_ok: bool
    peak: float | None = None
    ratio: float | None = None
    sharpness: float | None = None
    half_deviation_ms: float | None = None
    edge: bool | None = None
    candidate_ms: float | None = None
    peak_ok: bool = False
    ratio_ok: bool = False
    sharpness_ok: bool = False
    halves_ok: bool = False
    interior_ok: bool = False

    @property
    def accepted(self):
        """True when the gate and all four checks held."""
        return (
            self.speech
            and self.window_ok
            and self.peak_ok
            and self.ratio_ok
            and self.sharpness_ok
            and self.halves_ok
            and self.interior_ok
        )


@dataclass(frozen=True)
class AlignResult:
    """``offset_ms`` is the accepted alignment, or ``None`` (hold: change nothing)."""

    offset_ms: float | None
    checks: AlignChecks


class Features(NamedTuple):
    """Acceptance features of one tick; lags and offsets in ms."""

    peak: float
    ratio: float
    sharpness: float
    agree_ms: float
    est_ms: float  # peak lag, no interpolation
    est_interp_ms: float  # peak lag, parabolic
    edge: bool


DEFAULTS = AlignParams()


def pad_reference(ref_db, pad):
    """Reference dB values with ``pad`` values of silence (-96 dB) on both sides."""
    fill = np.full(pad, DB_MIN)
    return np.concatenate([fill, np.asarray(ref_db, dtype=np.float64), fill])


def floor_db(received_hist, p=DEFAULTS):
    """Silence floor from the received envelope's recent history (``received_hist``, dB)."""
    return max(p.floor_min_db, np.percentile(received_hist, p.floor_percentile) + p.floor_margin_db)


def ncc_curve(window, ref_padded, first, lags, floor, gate_std_db=2.0):
    """Mean-removed Pearson correlation curve of one received window against the reference.

    Args:
        window: The received window, dB, one value per hop.
        ref_padded: The reference in dB, padded (see :func:`pad_reference`).
        first: Index in ``ref_padded`` of the reference value that lines up with ``window[0]`` at
            lag 0. Lag ``l`` pairs ``window[m]`` with ``ref_padded[first + m - l]``.
        lags: Integer lags (in hops) to evaluate.
        floor: Both sides are clipped at this level (dB).
        gate_std_db: The clipped window needs at least this standard deviation (speech gate).

    Returns:
        The curve (one value per lag), or ``None`` when the window fails the speech gate.
    """
    w = np.maximum(window, floor)
    if w.std() < gate_std_db:
        return None
    idx = (first + np.arange(len(w)))[None, :] - np.asarray(lags)[:, None]
    seg = np.maximum(ref_padded[idx], floor)
    wc = w - w.mean()
    sc = seg - seg.mean(1, keepdims=True)
    den = np.sqrt((sc**2).sum(1) * (wc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (sc @ wc) / den, 0.0)


def peak_features(lags_ms, c, lo_ms, hi_ms, interpolate=True):
    """Peak, second-highest peak and parabolic estimate of curve ``c`` inside ``[lo, hi]`` ms.

    Returns ``(peak, second, est_ms, est_interp_ms, edge)``. ``second`` is the maximum outside
    the main lobe (walk down monotonically on both sides of the peak); ``edge`` is true when the
    peak sits at either end of the range.
    """
    sel = np.flatnonzero((lags_ms >= lo_ms - 1e-9) & (lags_ms <= hi_ms + 1e-9))
    cs = c[sel]
    i = int(np.argmax(cs))
    pk = cs[i]
    a = i
    while a > 0 and cs[a - 1] <= cs[a]:
        a -= 1
    b = i
    while b < len(cs) - 1 and cs[b + 1] <= cs[b]:
        b += 1
    rest = np.concatenate([cs[:a], cs[b + 1 :]])
    second = rest.max() if len(rest) else -1.0
    step = lags_ms[1] - lags_ms[0]
    est = lags_ms[sel[i]]
    est_i = est
    if interpolate and 0 < i < len(cs) - 1:
        y0, y1, y2 = cs[i - 1], cs[i], cs[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            est_i = est + 0.5 * (y0 - y2) / den * step
    edge = i == 0 or i == len(cs) - 1
    return pk, second, est, est_i, edge


def features(lags_ms, c_full, c_halves, lo_ms, hi_ms, p=DEFAULTS):
    """Acceptance features of one tick.

    Args:
        lags_ms: Lag axis of the curves, ms.
        c_full: Correlation curve of the full window.
        c_halves: Curves of the older and the newer half (``None`` for a half without speech).
        lo_ms, hi_ms: The search range for the full-window peak.
        p: Parameters.
    """
    pk, sec, est, est_i, edge = peak_features(lags_ms, c_full, lo_ms, hi_ms, p.interpolate)
    ratio = pk / sec if sec > 1e-3 else np.inf
    side = np.interp([est_i - p.sharp_at_ms, est_i + p.sharp_at_ms], lags_ms, c_full)
    sharp = pk - side.max()
    devs = []
    for c in c_halves:
        if c is not None:
            _, _, _, he, hedge = peak_features(
                lags_ms, c, est_i - p.half_search_ms, est_i + p.half_search_ms, p.interpolate
            )
            devs.append(np.inf if hedge else abs(he - est_i))
    agree = max(devs) if devs else np.inf
    return Features(pk, ratio, sharp, agree, est, est_i, bool(edge))


def passes(f, min_peak, min_ratio, min_sharpness, half_tol_ms):
    """The four acceptance checks and the edge check on ``f``: ``(peak, ratio, sharpness, halves,
    interior)``, each a bool."""
    return (
        bool(f.peak >= min_peak),
        bool(f.ratio >= min_ratio),
        bool(f.sharpness >= min_sharpness),
        bool(f.agree_ms <= half_tol_ms),
        not f.edge,
    )


def align(received_env, reference_env, prior_index, params=DEFAULTS):
    """One alignment tick.

    Args:
        received_env: The received track's envelope so far, dB (float), one value per hop,
            fully-inside windows only (the output of :func:`envelope_db` on the track, which the
            simulation high-passes first). The window the rule uses is its last 1 s.
        reference_env: The turn's envelope as sent (``uint8``, from :func:`envelope_bytes`),
            starting at reference index 0. Outside its range the reference counts as silence.
        prior_index: Where the rule expects reference index 0 to sit in ``received_env``, in
            hops (may be fractional). The search is +/- ``params.search_ms`` around it.
        params: Fixed terms and thresholds.

    Returns:
        An :class:`AlignResult`. ``offset_ms`` is the interpolated position of reference index 0
        on ``received_env``'s index axis, in ms: received value ``j`` corresponds to reference
        value ``j - offset_ms / hop_ms``. It is ``None`` unless every check held, and then the
        caller keeps its current alignment.
    """
    hop = params.hop_ms
    rec = np.asarray(received_env, dtype=np.float64)
    jt = len(rec)
    m_full = round(params.window_ms / hop)
    m_half = m_full // 2
    if jt < m_full:
        return AlignResult(None, AlignChecks(speech=False, window_ok=False))
    ref_db = dequantise(reference_env)
    floor = floor_db(rec[max(0, jt - int(params.floor_history_ms / hop)) : jt], params)

    prior_ms = prior_index * hop
    lo_ms, hi_ms = prior_ms - params.search_ms, prior_ms + params.search_ms
    margin = params.half_search_ms + 2 * hop
    lag_min = int(np.floor((lo_ms - margin) / hop))
    lag_max = int(np.ceil((hi_ms + margin) / hop))
    lags = np.arange(lag_min, lag_max + 1)
    lags_ms = lags * hop
    pad = max(lag_max - (jt - m_full), (jt - lag_min) - len(ref_db), 0) + 1
    ref_padded = pad_reference(ref_db, pad)

    def curve(length, end):
        return ncc_curve(
            rec[end - length : end], ref_padded, pad + end - length, lags, floor, params.gate_std_db
        )

    c_full = curve(m_full, jt)
    if c_full is None:
        return AlignResult(None, AlignChecks(speech=False, window_ok=True))
    halves = [curve(m_half, jt - m_half), curve(m_half, jt)]
    f = features(lags_ms, c_full, halves, lo_ms, hi_ms, params)
    ok = passes(f, params.min_peak, params.min_ratio, params.min_sharpness, params.half_tol_ms)
    checks = AlignChecks(
        speech=True,
        window_ok=True,
        peak=float(f.peak),
        ratio=float(f.ratio),
        sharpness=float(f.sharpness),
        half_deviation_ms=float(f.agree_ms),
        edge=f.edge,
        candidate_ms=float(f.est_interp_ms),
        peak_ok=ok[0],
        ratio_ok=ok[1],
        sharpness_ok=ok[2],
        halves_ok=ok[3],
        interior_ok=ok[4],
    )
    return AlignResult(float(f.est_interp_ms) if checks.accepted else None, checks)


# ---------------------------------------------------------------- the stall rule
#
# The per-turn state on top of :func:`align`. The page and the SDK run the rule as a tick loop
# (cadence, priors, splice: ``replay_capture.replay_turn_tracked``); what happens between ticks and
# what a tick may accept besides the full window is defined here, so the replay, the simulation and
# the SDK's port share one definition. The fixed terms: :class:`StallParams` and ``README.md``
# ("The stall rule").


@dataclass(frozen=True)
class StallParams:
    """The stall rule's fixed terms (lengths in ms). ``TODAY`` switches every part off: the tracker
    is then today's rule plus the SDK's 2 s loss clock.

    * ``freeze``: after the turn's first lock, once ``freeze_after_ms`` of consecutive received
      frames are each at or below ``silent_db`` while the turn's own reference at the (advancing)
      pointer is above ``ref_speech_db``, the reference pointer stops (retroactively over the run)
      and stays stopped while the received stays silent, for at most ``freeze_max_ms``. While a
      confirm is pending (below) "silent" means digital silence, at or below
      ``confirming_silent_db``: after a wrong freeze (a burst loss, where the timeline did not
      move) the received pauses where the shifted reference speaks, and freezing there would
      push the pointer further and discard the confirm that undoes it; a second real stall still
      ends in exact zeros (the jitter buffer's concealment fades to digital silence) and freezes.
      Such a freeze keeps the earlier freeze's bracket (``carry_bracket``): the confirm's range
      still starts at the pointer before the earlier freeze and spans everything received since
      the last frame that matched before it.
    * ``confirm``: when received audio resumes, a correlation of the received audio since the
      resume against the own reference, every ``confirm_every_ms`` from ``confirm_from_ms`` to
      ``confirm_until_ms`` of it, at positions in [pointer before the freeze - ``confirm_back_ms``,
      pointer before the freeze + everything received since the last frame that matched the
      reference before the freeze + ``confirm_fwd_ms``]; accepted at an interior peak >=
      ``confirm_min_peak``, peak / second-highest peak >= ``confirm_min_ratio`` and sharpness >=
      ``confirm_min_sharpness`` (curve at the interpolated peak +/- ``confirm_sharp_at_ms``), both
      sides clipped at ``confirm_floor_db`` (:func:`confirm_position`). The last
      matching frame: the newest frame within ``match_back_ms`` before the run whose
      ``match_frames`` consecutive frames are each over ``match_min_db`` and within
      ``match_tol_db`` of the reference at the pointer then; without one there is no confirm.
    * ``newer_half``: a tick after the first lock whose full window is not accepted searches the
      newer 0.5 s alone within +/- ``half_search_ms`` of the pointer (peak, ratio, sharpness and
      interior as :func:`align`); it is accepted when the previous tick's newer-half candidate
      agrees within ``half_agree_ms`` (two consecutive ticks, 200 ms apart after the lock), and
      refused when the newer half holds a run of ``half_silence_ms`` at or below
      ``digital_silence_db`` (digital silence: a stall inside it).
    * ``hold_clock``: frozen time does not count toward the ``loss_ms`` loss clock.
    * ``chain_prior`` (rule 5, R74): a chained turn whose predecessor is locked takes the
      predecessor's position + ``prev_turn_pcm_ms`` as its prior; before its own first lock an
      onset prior re-anchors it only within ``chain_onset_tol_ms`` of that position
      (:func:`chained_prior`).
    """

    freeze: bool = True
    confirm: bool = True
    newer_half: bool = True
    hold_clock: bool = True
    chain_prior: bool = True
    confirming_silent_db: float = -95.0
    carry_bracket: bool = True
    silent_db: float = -60.0
    ref_speech_db: float = -50.0
    freeze_after_ms: float = 20.0
    freeze_max_ms: float = 2000.0
    match_back_ms: float = 400.0
    match_frames: int = 3
    match_min_db: float = -60.0
    match_tol_db: float = 4.0
    confirm_from_ms: float = 250.0
    confirm_until_ms: float = 1200.0
    confirm_every_ms: float = 50.0
    confirm_back_ms: float = 150.0
    confirm_fwd_ms: float = 20.0
    confirm_floor_db: float = -70.0
    confirm_min_peak: float = 0.95
    confirm_min_ratio: float = 1.2
    confirm_min_sharpness: float = 0.03
    confirm_sharp_at_ms: float = 20.0
    half_search_ms: float = 300.0
    half_agree_ms: float = 10.0
    half_silence_ms: float = 40.0
    digital_silence_db: float = -95.0
    loss_ms: float = 2000.0
    chain_onset_tol_ms: float = 200.0


STALL = StallParams()
TODAY = StallParams(
    freeze=False, confirm=False, newer_half=False, hold_clock=False, chain_prior=False
)


def longest_run(mask):
    """Length of the longest run of True in ``mask``."""
    m = np.asarray(mask, dtype=np.int8)
    if not len(m):
        return 0
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((ends - starts).max()) if len(starts) else 0


def newer_half_candidate(
    received_env, reference_env, prior_index, params=DEFAULTS, search_ms=300.0
):
    """The newer 0.5 s of the window alone, searched within +/- ``search_ms`` of the prior.

    Same arguments and lag axis as :func:`align`. Returns the interpolated peak (ms on the received
    axis, as ``align``'s ``offset_ms``) when peak, ratio, sharpness and interior all hold, else
    None. The half-window check of ``align`` does not apply (this *is* a half).
    """
    hop = params.hop_ms
    rec = np.asarray(received_env, dtype=np.float64)
    jt = len(rec)
    m_half = round(params.window_ms / hop) // 2
    if jt < m_half:
        return None
    ref_db = dequantise(reference_env)
    floor = floor_db(rec[max(0, jt - int(params.floor_history_ms / hop)) : jt], params)
    prior_ms = prior_index * hop
    lo_ms, hi_ms = prior_ms - search_ms, prior_ms + search_ms
    margin = 2 * hop + params.sharp_at_ms
    lag_min = int(np.floor((lo_ms - margin) / hop))
    lag_max = int(np.ceil((hi_ms + margin) / hop))
    lags = np.arange(lag_min, lag_max + 1)
    pad = max(lag_max - (jt - m_half), (jt - lag_min) - len(ref_db), 0) + 1
    ref_padded = pad_reference(ref_db, pad)
    c = ncc_curve(
        rec[jt - m_half : jt], ref_padded, pad + jt - m_half, lags, floor, params.gate_std_db
    )
    if c is None:
        return None
    lags_ms = lags * hop
    pk, sec, _, est_i, edge = peak_features(lags_ms, c, lo_ms, hi_ms, params.interpolate)
    ratio = pk / sec if sec > 1e-3 else np.inf
    side = np.interp([est_i - params.sharp_at_ms, est_i + params.sharp_at_ms], lags_ms, c)
    sharp = pk - side.max()
    ok = (
        pk >= params.min_peak
        and ratio >= params.min_ratio
        and sharp >= params.min_sharpness
        and not edge
    )
    return float(est_i) if ok else None


def confirm_position(env, own_db, j_res, j_end, lo, hi, stall=STALL, hop_ms=5.0):
    """The confirm correlation: received frames ``[j_res, j_end)`` against the turn's own reference
    (dB) at every position ``l`` in ``[floor(lo), ceil(hi)]`` (received frame of the reference's
    index 0; frame ``j`` pairs with reference frame ``j - l``), both clipped at
    ``confirm_floor_db``, mean-removed Pearson as :func:`ncc_curve`. Peak, second-highest peak,
    parabolic estimate and edge as :func:`peak_features` over that range; accepted when the peak
    is >= ``confirm_min_peak``, peak / second >= ``confirm_min_ratio``, sharpness (interpolated
    peak minus the curve at it +/- ``confirm_sharp_at_ms``) >= ``confirm_min_sharpness`` and the
    peak is not at either end of the range. Returns the interpolated position or None."""
    floor = stall.confirm_floor_db
    w = np.maximum(np.asarray(env[j_res:j_end], dtype=np.float64), floor)
    if len(w) < 2 or w.std() < DEFAULTS.gate_std_db:
        return None
    sa = stall.confirm_sharp_at_ms / hop_ms
    ext = int(np.ceil(sa)) + 1
    a, b = int(np.floor(lo)), int(np.ceil(hi))
    if b - a < 2:
        return None
    ls = np.arange(a - ext, b + ext + 1)
    k = np.arange(j_res, j_end)[None, :] - ls[:, None]
    inside = (k >= 0) & (k < len(own_db))
    v = np.where(inside, own_db[np.clip(k, 0, max(0, len(own_db) - 1))], DB_MIN)
    v = np.maximum(v, floor)
    wc = w - w.mean()
    vc = v - v.mean(1, keepdims=True)
    den = np.sqrt((vc**2).sum(1) * (wc**2).sum())
    c = np.where(den > 0, (vc @ wc) / np.where(den > 0, den, 1.0), 0.0)
    pk, sec, _, est, edge = peak_features(ls.astype(np.float64), c, a, b)
    ratio = pk / sec if sec > 1e-3 else np.inf
    if edge or pk < stall.confirm_min_peak or ratio < stall.confirm_min_ratio:
        return None
    side = np.interp([est - sa, est + sa], ls, c)
    if pk - side.max() < stall.confirm_min_sharpness:
        return None
    return float(est)


def chained_prior(chain_l, onset_l, stall=STALL, hop_ms=5.0):
    """Rule 5 (R74): the prior of a chained turn whose predecessor is locked, before its own first
    lock. ``chain_l``: the predecessor's position + ``prev_turn_pcm_ms`` (frames); ``onset_l``: the
    onset prior (frames) or None. Returns ``(frames, kind)``: the onset prior (``'onset'``) when it
    lies within ``chain_onset_tol_ms`` of the chain position, else the chain position
    (``'chain'``)."""
    if onset_l is not None and abs(onset_l - chain_l) * hop_ms <= stall.chain_onset_tol_ms:
        return onset_l, "onset"
    return chain_l, "chain"


def ref_speech(ref_u8, own_off, l, count, floor, params=DEFAULTS):
    """The SDK's ``refSpeech``: whether the reference, where it is believed to sit (``l``: received
    frame of the turn's own index 0; ``ref_u8`` spliced with the turn's own index 0 at
    ``own_off``), has speech in the received window ``[count - 1 s, count)``."""
    m = round(params.window_ms / params.hop_ms)
    ref_db = dequantise(ref_u8)
    k = np.arange(count - m, count) - round(l) + own_off
    inside = (k >= 0) & (k < len(ref_db))
    v = np.where(inside, ref_db[np.clip(k, 0, max(0, len(ref_db) - 1))], DB_MIN)
    return bool(np.maximum(v, floor).std() >= params.gate_std_db)


class TurnTracker:
    """One turn's alignment state: today's rule (:func:`align` per tick, hold, the 2 s loss clock)
    plus the stall rule (:class:`StallParams`).

    The caller feeds every received frame once, in order (:meth:`frame`, the envelope value at
    index ``j`` must exist), and runs :meth:`tick` at its cadence with the reference as arrived.
    Positions (``l``) are received frames of the turn's own reference index 0, as in the replay.

    ``events`` lists ``(frame, event)`` with the SDK's sync event shapes for the freeze:
    ``{'kind': 'frozen', 'turnId', 'atMs'}`` when a freeze starts (``atMs``: the turn's own audio
    position, ms, where the pointer stopped) and ``{'kind': 'unfrozen', 'turnId', 'frozenMs',
    'confirmed'}`` when its episode closes: the confirm accepted (``confirmed`` true), or a full or
    newer-half lock, the confirm's time ran out, no confirm was possible, a new freeze started, or
    the turn ended (false). ``lost_at`` lists the frames where the loss clock reached ``loss_ms``
    (the SDK's ``sync-lost``). ``freezes`` keeps each closed episode for reports.
    """

    def __init__(self, turn_id, params=DEFAULTS, stall=STALL):
        self.turn_id = turn_id
        self.p = params
        self.s = stall
        hop = params.hop_ms
        self.hop = hop
        self.n_min = max(1, round(stall.freeze_after_ms / hop))
        self.n_max = round(stall.freeze_max_ms / hop)
        self.n_back = round(stall.match_back_ms / hop)
        self.n_from = round(stall.confirm_from_ms / hop)
        self.n_until = round(stall.confirm_until_ms / hop)
        self.n_every = max(1, round(stall.confirm_every_ms / hop))
        self.n_half_silence = round(stall.half_silence_ms / hop)
        self.own_db = np.zeros(0)
        self.locked = False
        self.l = None
        self.held_no_lock = 0
        self.unlocked_ms = 0.0
        self.lost = False
        self.lost_at = []
        self.events = []
        self.freezes = []
        self._run = 0
        self._run_k = None
        self._episode = None
        self._pending = None
        self._prev_half = None
        self._frozen_since_tick = 0

    # ------------------------------------------------------------ inputs

    def set_reference(self, own_u8):
        """The turn's own reference envelope as arrived so far (``uint8``)."""
        if len(own_u8) != len(self.own_db):
            self.own_db = dequantise(own_u8)

    @property
    def frozen(self):
        """Whether the pointer is held now."""
        ep = self._episode
        return ep is not None and ep["resume"] is None and ep["frames"] < self.n_max

    def _matches(self, env, l, j):
        k = round(j - l)
        return (
            0 <= k < len(self.own_db)
            and j >= 0
            and env[j] >= self.s.match_min_db
            and abs(env[j] - self.own_db[k]) <= self.s.match_tol_db
        )

    def _close(self, j, how, **extra):
        ep = self._episode
        if ep is None:
            return
        ep.update(how=how, closed=j, **extra)
        self.events.append(
            (
                j,
                {
                    "kind": "unfrozen",
                    "turnId": self.turn_id,
                    "frozenMs": ep["frames"] * self.hop,
                    "confirmed": how == "confirm",
                },
            )
        )
        self.freezes.append(ep)
        self._episode = None
        self._pending = None

    def frame(self, env, j):
        """Received frame ``j`` (``env[j]``, dB) has completed."""
        if not (self.locked and self.s.freeze):
            return
        s = self.s
        level = s.silent_db if self._pending is None else min(s.silent_db, s.confirming_silent_db)
        silent = env[j] <= level
        if silent:
            if self._run < self.n_min:
                # the run that triggers a freeze: every frame silent where the reference at the
                # pointer (still advancing) has speech
                k = round(j - self.l)
                if 0 <= k < len(self.own_db) and self.own_db[k] > s.ref_speech_db:
                    if self._run == 0:
                        self._run_k = k
                    self._run += 1
                    if self._run == self.n_min:
                        self._start(env, j)
                else:
                    self._run = 0
            else:
                self._run += 1
                ep = self._episode
                if ep is not None and ep["resume"] is None and ep["frames"] < self.n_max:
                    self.l += 1
                    ep["frames"] += 1
                    self._frozen_since_tick += 1
        else:
            if self._run >= self.n_min and self._episode is not None:
                self._resume(j)
            self._run = 0
        pd = self._pending
        if pd is not None:
            n_post = j + 1 - pd["j_res"]
            if n_post > self.n_until:
                self._close(j, "expired")
            elif n_post >= self.n_from and (n_post - self.n_from) % self.n_every == 0:
                est = confirm_position(
                    env, self.own_db, pd["j_res"], j + 1, pd["lo"], pd["hi"], s, self.hop
                )
                if est is not None:
                    self.l = est
                    self.unlocked_ms = 0.0
                    self.lost = False
                    self._close(j, "confirm", confirm_l=est, confirm_after_ms=n_post * self.hop)

    def _start(self, env, j):
        j0 = j - self.n_min + 1
        carried = None
        if self._episode is not None:
            ep = self._episode
            if self._pending is not None and self.s.carry_bracket:
                carried = (ep["l_pre"], ep["j_match"])
            self._close(j, "superseded")
        if carried is not None:
            # unconfirmed: the bracket still starts at the last frame that matched before the
            # earlier freeze, so the confirm can still choose "nothing moved" (a burst loss) or
            # "everything since moved" (a second stall)
            l_pre, j_match = carried
        else:
            l_pre = self.l
            j_match = None
            lowest = max(j0 - self.n_back, self.s.match_frames - 1)
            for jm in range(j0 - 1, lowest - 1, -1):
                if all(self._matches(env, l_pre, jm - d) for d in range(self.s.match_frames)):
                    j_match = jm
                    break
        n = min(self.n_min, self.n_max)
        self.l += n
        self._frozen_since_tick += n
        self._episode = {
            "start": j0,
            "at_ms": self._run_k * self.hop,
            "l_pre": l_pre,
            "j_match": j_match,
            "frames": n,
            "resume": None,
            "carried": carried is not None,
        }
        self.events.append(
            (j, {"kind": "frozen", "turnId": self.turn_id, "atMs": self._run_k * self.hop})
        )

    def _resume(self, j):
        ep = self._episode
        ep["resume"] = j
        if not self.s.confirm:
            self._close(j, "no-confirm")
        elif ep["j_match"] is None:
            self._close(j, "no-match")
        else:
            self._pending = {
                "j_res": j,
                "lo": ep["l_pre"] - self.s.confirm_back_ms / self.hop,
                "hi": ep["l_pre"] + (j - ep["j_match"] - 1) + self.s.confirm_fwd_ms / self.hop,
            }

    def finish(self, j):
        """The turn's ticks are over: close an open freeze."""
        self._close(j, "end")

    # ------------------------------------------------------------ the tick

    def tick(self, env, count, ref, own_off, centre, search_ms, since_ms):
        """One alignment tick on received frames ``[0, count)``.

        ``ref``: the reference as correlated (spliced: the turn's own index 0 at ``own_off``);
        ``centre``/``search_ms``: the prior and its search before the first lock (after it the
        tracker uses its own position and ``params.search_ms``); ``since_ms``: received time since
        the previous tick that returned a record. Returns None when the window is not complete yet
        (not a tick), else the tick's record.
        """
        p, s, hop = self.p, self.s, self.hop
        if self.locked:
            centre, search_ms = self.l, p.search_ms
        hist = round(p.floor_history_ms / hop)
        base = max(0, count - hist)
        rec = np.asarray(env[base:count], dtype=np.float64)
        pp = AlignParams(**{**p.__dict__, "search_ms": search_ms})
        r = align(rec, ref, centre - own_off - base, pp)
        if not r.checks.window_ok:
            return None
        since = since_ms
        if s.hold_clock:
            since = max(0.0, since - self._frozen_since_tick * hop)
        self._frozen_since_tick = 0
        c = r.checks
        accepted = r.offset_ms is not None
        l_acc = r.offset_ms / hop + own_off + base if accepted else None
        via = "full" if accepted else None
        half_l, half_refused = None, False
        if not accepted and s.newer_half and self.locked and c.speech:
            hc = newer_half_candidate(rec, ref, centre - own_off - base, pp, s.half_search_ms)
            if hc is not None:
                m_half = round(p.window_ms / hop) // 2
                if longest_run(rec[-m_half:] <= s.digital_silence_db) >= self.n_half_silence:
                    hc, half_refused = None, True
            if hc is not None:
                half_l = hc / hop + own_off + base
                if (
                    self._prev_half is not None
                    and abs(half_l - self._prev_half) * hop <= s.half_agree_ms
                ):
                    accepted, l_acc, via = True, half_l, "half"
                self._prev_half = half_l
            else:
                self._prev_half = None
        elif accepted:
            self._prev_half = None
        if accepted:
            self.unlocked_ms = 0.0
            self.lost = False
        elif not c.speech:
            l_now = self.l if self.locked else centre
            if ref_speech(ref, own_off, l_now, count, floor_db(rec, p), p):
                self.unlocked_ms += since
        else:
            self.unlocked_ms += since
        if self.unlocked_ms >= s.loss_ms and not self.lost:
            self.lost = True
            self.lost_at.append(count)
        if accepted:
            if self._episode is not None:
                self._close(count - 1, via)
            self._run = 0
            self.locked, self.l = True, l_acc
            self.held_no_lock = 0
        elif c.speech and not self.locked:
            self.held_no_lock += 1
        return {
            "count": count,
            "speech": c.speech,
            "accepted": accepted,
            "via": via,
            "l": l_acc,
            "candidate_l": None
            if c.candidate_ms is None
            else c.candidate_ms / hop + own_off + base,
            "half_l": half_l,
            "half_refused": half_refused,
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
            "unlocked_ms": self.unlocked_ms,
            "lost": self.lost,
        }
