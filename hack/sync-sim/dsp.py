"""The normative definition of the envelope fingerprint and of the alignment rule.

Everything an implementation must agree on is in this file; the web SDK ports it verbatim. Two
public entry points:

* :func:`envelope_bytes` -- the 8-bit log-RMS envelope of the audio the plugin releases. The
  package's ``Envelope`` (``src/pipecat_spatialreal/fingerprint.py``) must produce the same bytes;
  ``tests/test_fingerprint_oracle.py`` asserts it.
* :func:`align` -- one alignment tick: where in the received envelope does the turn's reference
  envelope sit, and may we trust it?

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
