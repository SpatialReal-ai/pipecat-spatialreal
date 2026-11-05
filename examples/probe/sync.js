// The envelope-fingerprint alignment rule, in the browser: a port of hack/sync-sim/dsp.py.
//
// The processor sends `turn_envelope`: one byte per 5 ms of the audio it released (10 ms windows,
// dB of the mean square, [-96, 0] dB on [0, 255]). This module measures the same envelope on the
// track the browser receives and finds where the turn's reference sits in it (design §3.2,
// backend-ng docs/superpowers/specs/2026-11-03-pipecat-sync-fingerprint-design.md).
//
// dsp.py is normative. envelopeBytes() and align() mirror envelope_bytes() and align() term for
// term; tests/probe_check.mjs holds them to the oracle's numbers (tests/fixtures/sync/oracle.json).
// One documented difference: the oracle's receiver high-pass is a zero-phase 4th-order filter
// applied in the FFT domain over the whole signal; here it is causal, a first-order IIR at 30 Hz,
// as the browser must run it (design §3.2, "One known difference").
//
// SyncTracker runs the rule on a live track the way the web SDK does: the cadence (§3.2 step 8),
// the prior and the widened search before the first lock (steps 4 and 10, with R57, R72 and rule
// 5 / R76), the splice of the previous turn's envelope (§3.3), and the stall rule per turn
// (TurnTracker, a port of dsp.TurnTracker: freeze, confirm, newer-half re-lock, frozen time off
// the loss clock). It never moves anything on screen: it records what each tick saw and the
// freezes.
//
// No dependencies, no DOM, no clock.

export const HOP_MS = 5
export const WINDOW_MS = 10
export const DB_MIN = -96
export const DB_MAX = 0
const DB_RANGE = DB_MAX - DB_MIN
const FULL_SCALE_SQUARED = 32768 * 32768 // int16 magnitude squared: the 0 dB reference
const EPSILON = 1e-10 // added to the normalised mean square so silence is finite (-100 dB)

/** The rule's fixed terms and thresholds (dsp.AlignParams; lengths in ms). */
export const DEFAULTS = Object.freeze({
  hopMs: 5,
  windowMs: 1000,
  searchMs: 300,
  halfSearchMs: 40,
  sharpAtMs: 20,
  floorHistoryMs: 3000,
  floorPercentile: 5,
  floorMarginDb: 6,
  floorMinDb: -70,
  gateStdDb: 2,
  minPeak: 0.95,
  minRatio: 1.2,
  minSharpness: 0.03,
  halfTolMs: 10,
  interpolate: true,
})

/** The receiver's high-pass corner, Hz. */
export const HIGH_PASS_HZ = 30

/** What a tick's prior was (its `prior_kind`): see SyncTracker.prior(). */
export const PRIOR_KINDS = Object.freeze(['locked', 'chain', 'onset', 'gap', 'arrival'])

/** How a live tracker ticks (design §3.2 steps 4, 8 and 10). */
export const CADENCE = Object.freeze({
  beforeLockMs: 50, // a tick every 50 ms of new audio until the turn's first lock
  afterLockMs: 200, // then every 200 ms
  priorLeadMs: 200, // the first prior: turn_start arrival + 200 ms
  widenAfterTicks: 10, // ticks with speech and no lock before the search widens
  wideSearchMs: 600, // the widened search, re-centred on the onset-detected position
})

// ------------------------------------------------------------------ numbers

/** Python's round(): half to even (the oracle sizes the received hop with it). */
export function pyRound(x) {
  const f = Math.floor(x)
  const d = x - f
  if (d > 0.5) return f + 1
  if (d < 0.5) return f
  return f % 2 === 0 ? f : f + 1
}

/** dB to the contract's byte: clamp to [-96, 0], then floor((db + 96) / 96 * 255 + 0.5). */
export function quantise(db) {
  const c = Math.min(DB_MAX, Math.max(DB_MIN, db))
  return Math.floor(((c - DB_MIN) / DB_RANGE) * 255 + 0.5)
}

/** A byte back to dB (0 -> -96, 255 -> 0). */
export function dequantise(byte) {
  return (byte * DB_RANGE) / 255 + DB_MIN
}

/** numpy.percentile(values, q) with the default linear method, operation for operation. */
export function percentile(values, q) {
  const a = Float64Array.from(values).sort()
  const n = a.length
  const quantile = q / 100
  const virtual = n * quantile + (1 + quantile * (1 - 1 - 1)) - 1
  if (virtual >= n - 1) return a[n - 1]
  if (virtual < 0) return a[0]
  const prev = Math.floor(virtual)
  const gamma = virtual - prev
  const lo = a[prev]
  const hi = a[prev + 1]
  const diff = hi - lo
  return gamma >= 0.5 ? hi - diff * (1 - gamma) : lo + diff * gamma
}

/** A sample array in int16 units: an Int16Array as is, a Float32Array (Web Audio, [-1, 1]) x 32768. */
function int16Scale(pcm) {
  if (pcm instanceof Int16Array) return pcm
  if (pcm instanceof Float32Array) return pcm.map(x => x * 32768) // exact: a power-of-two scale
  throw new TypeError('pcm must be an Int16Array or a Float32Array')
}

/** dB of every window of `win` samples starting at a multiple of `hop` and fully inside `x` (dsp._window_db). */
function windowDb(x, hop, win) {
  const n = Math.floor((x.length - win) / hop) + 1
  const out = new Float64Array(Math.max(0, n))
  for (let k = 0; k < n; k++) {
    let sum = 0
    for (let i = k * hop, end = k * hop + win; i < end; i++) sum += x[i] * x[i]
    out[k] = 10 * Math.log10(sum / win / FULL_SCALE_SQUARED + EPSILON)
  }
  return out
}

// ------------------------------------------------------------------ envelopes

/**
 * The production envelope of mono audio (dsp.envelope_bytes): one byte per 5 ms hop, 10 ms
 * windows, `ms * sampleRate // 1000` samples each, only windows fully inside the audio (1 s gives
 * 199 values; value k covers samples [k * hop, k * hop + window)).
 *
 * @param {Int16Array|Float32Array} pcm Int16 samples, or Web Audio floats in [-1, 1] (measured x 32768).
 * @param {number} sampleRate
 * @returns {Uint8Array}
 */
export function envelopeBytes(pcm, sampleRate) {
  const hop = Math.floor((HOP_MS * sampleRate) / 1000)
  const win = Math.floor((WINDOW_MS * sampleRate) / 1000)
  return Uint8Array.from(windowDb(int16Scale(pcm), hop, win), quantise)
}

/** First-order high-pass y[n] = a (y[n-1] + x[n] - x[n-1]), a = RC / (RC + dt); state starts at 0. */
export function highPassCoefficient(hz, sampleRate) {
  const rc = 1 / (2 * Math.PI * hz)
  const dt = 1 / sampleRate
  return rc / (rc + dt)
}

/**
 * The received track's envelope, streaming (dsp.envelope_db with the receiver's high-pass): dB per
 * hop of `pyRound(0.005 * rate)` samples, windows of two hops, fully-inside windows only, after a
 * causal first-order high-pass (`highPassHz: null` measures the raw samples). Frame g covers samples
 * [g * hop, g * hop + 2 * hop) of everything pushed since construction.
 */
export class ReceivedEnvelope {
  constructor(sampleRate, { highPassHz = HIGH_PASS_HZ } = {}) {
    this.sampleRate = sampleRate
    this.hop = pyRound((HOP_MS / 1000) * sampleRate)
    this.alpha = highPassHz == null ? null : highPassCoefficient(highPassHz, sampleRate)
    this.frames = new Float64Array(1024)
    this.count = 0 // complete frames
    this.samples = 0 // samples pushed
    this.prevSum = null // sum of squares of the previous hop
    this.partial = 0
    this.partialN = 0
    this.prevX = 0
    this.prevY = 0
  }

  /** Feed the next block. `int16` says the block is already in int16 units (else Web Audio floats). */
  push(block, { int16 = block instanceof Int16Array } = {}) {
    const scale = int16 ? 1 : 32768
    const hop = this.hop
    for (let i = 0; i < block.length; i++) {
      let x = block[i] * scale
      if (this.alpha !== null) {
        const y = this.alpha * (this.prevY + x - this.prevX)
        this.prevX = x
        this.prevY = y
        x = y
      }
      this.partial += x * x
      if (++this.partialN === hop) {
        if (this.prevSum !== null) this.append(10 * Math.log10((this.prevSum + this.partial) / (2 * hop) / FULL_SCALE_SQUARED + EPSILON))
        this.prevSum = this.partial
        this.partial = 0
        this.partialN = 0
      }
    }
    this.samples += block.length
  }

  append(db) {
    if (this.count === this.frames.length) {
      const grown = new Float64Array(this.frames.length * 2)
      grown.set(this.frames)
      this.frames = grown
    }
    this.frames[this.count++] = db
  }

  /** The complete frames, as a view (do not keep it across pushes). */
  values() {
    return this.frames.subarray(0, this.count)
  }
}

/** The received envelope of a whole signal: ReceivedEnvelope over one block. */
export function receivedEnvelope(pcm, sampleRate, opts = {}) {
  const env = new ReceivedEnvelope(sampleRate, opts)
  env.push(pcm)
  return Float64Array.from(env.values())
}

// ------------------------------------------------------------------ the rule (dsp.align)

/** Silence floor from the received history, dB (dsp.floor_db). */
export function floorDb(history, p = DEFAULTS) {
  return Math.max(p.floorMinDb, percentile(history, p.floorPercentile) + p.floorMarginDb)
}

/**
 * Mean-removed Pearson correlation of one received window against the padded reference at every
 * lag (dsp.ncc_curve); lag l pairs window[m] with refPadded[first + m - l]. null when the clipped
 * window's standard deviation is under the gate.
 */
function nccCurve(window, refPadded, first, lags, floor, gateStdDb) {
  const m = window.length
  const w = new Float64Array(m)
  let mean = 0
  for (let i = 0; i < m; i++) {
    w[i] = Math.max(window[i], floor)
    mean += w[i]
  }
  mean /= m
  let wss = 0
  for (let i = 0; i < m; i++) {
    w[i] -= mean
    wss += w[i] * w[i]
  }
  if (Math.sqrt(wss / m) < gateStdDb) return null
  const out = new Float64Array(lags.length)
  const seg = new Float64Array(m)
  for (let li = 0; li < lags.length; li++) {
    const base = first - lags[li]
    let sm = 0
    for (let i = 0; i < m; i++) {
      seg[i] = Math.max(refPadded[base + i], floor)
      sm += seg[i]
    }
    sm /= m
    let dot = 0
    let sss = 0
    for (let i = 0; i < m; i++) {
      const s = seg[i] - sm
      dot += s * w[i]
      sss += s * s
    }
    const den = Math.sqrt(sss * wss)
    out[li] = den > 0 ? dot / den : 0
  }
  return out
}

/** Peak, second peak (outside the main lobe) and parabolic estimate of `c` within [lo, hi] ms (dsp.peak_features). */
function peakFeatures(lagsMs, c, loMs, hiMs, interpolate) {
  const sel = []
  for (let i = 0; i < lagsMs.length; i++) if (lagsMs[i] >= loMs - 1e-9 && lagsMs[i] <= hiMs + 1e-9) sel.push(i)
  const cs = sel.map(i => c[i])
  let i = 0
  for (let k = 1; k < cs.length; k++) if (cs[k] > cs[i]) i = k
  const pk = cs[i]
  let a = i
  while (a > 0 && cs[a - 1] <= cs[a]) a--
  let b = i
  while (b < cs.length - 1 && cs[b + 1] <= cs[b]) b++
  let second = -1
  let any = false
  for (let k = 0; k < cs.length; k++) {
    if (k >= a && k <= b) continue
    second = any ? Math.max(second, cs[k]) : cs[k]
    any = true
  }
  const step = lagsMs[1] - lagsMs[0]
  const est = lagsMs[sel[i]]
  let estI = est
  if (interpolate && i > 0 && i < cs.length - 1) {
    const y0 = cs[i - 1]
    const y1 = cs[i]
    const y2 = cs[i + 1]
    const den = y0 - 2 * y1 + y2
    if (den < 0) estI = est + ((0.5 * (y0 - y2)) / den) * step
  }
  return { pk, second, est, estI, edge: i === 0 || i === cs.length - 1 }
}

/** numpy.interp on an ascending grid, clamped at the ends. */
function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0]
  const n = xs.length
  if (x >= xs[n - 1]) return ys[n - 1]
  let j = 0
  while (xs[j + 1] < x) j++
  if (xs[j + 1] === x) return ys[j + 1]
  const t = (x - xs[j]) / (xs[j + 1] - xs[j])
  return ys[j] + (ys[j + 1] - ys[j]) * t
}

/**
 * One alignment tick (dsp.align).
 *
 * @param {ArrayLike<number>} received The received envelope so far, dB per hop (the rule uses its last 1 s).
 * @param {Uint8Array|number[]} reference The turn's envelope bytes from reference index 0; silence outside.
 * @param {number} priorIndex Where reference index 0 is expected on `received`'s index axis, hops (fractional allowed).
 * @param {object} [params] DEFAULTS, or a copy with some terms changed (searchMs to widen).
 * @returns {{offsetMs: number|null, checks: object}} offsetMs: reference index 0 on the received
 *   axis, ms (received value j is reference value j - offsetMs / hopMs), or null: hold.
 */
export function align(received, reference, priorIndex, params = DEFAULTS) {
  const p = { ...DEFAULTS, ...params }
  const hop = p.hopMs
  const jt = received.length
  const mFull = pyRound(p.windowMs / hop)
  const mHalf = Math.floor(mFull / 2)
  const blank = { speech: false, windowOk: false, peak: null, ratio: null, sharpness: null, halfDeviationMs: null, edge: null, candidateMs: null, peakOk: false, ratioOk: false, sharpnessOk: false, halvesOk: false, interiorOk: false, accepted: false }
  if (jt < mFull) return { offsetMs: null, checks: blank }
  const history = []
  for (let j = Math.max(0, jt - Math.trunc(p.floorHistoryMs / hop)); j < jt; j++) history.push(received[j])
  const floor = floorDb(history, p)

  const priorMs = priorIndex * hop
  const loMs = priorMs - p.searchMs
  const hiMs = priorMs + p.searchMs
  const margin = p.halfSearchMs + 2 * hop
  const lagMin = Math.floor((loMs - margin) / hop)
  const lagMax = Math.ceil((hiMs + margin) / hop)
  const lags = []
  for (let l = lagMin; l <= lagMax; l++) lags.push(l)
  const lagsMs = lags.map(l => l * hop)
  const pad = Math.max(lagMax - (jt - mFull), jt - lagMin - reference.length, 0) + 1
  const refPadded = new Float64Array(reference.length + 2 * pad).fill(DB_MIN)
  for (let k = 0; k < reference.length; k++) refPadded[pad + k] = dequantise(reference[k])

  const curve = (length, end) => {
    const window = new Float64Array(length)
    for (let i = 0; i < length; i++) window[i] = received[end - length + i]
    return nccCurve(window, refPadded, pad + end - length, lags, floor, p.gateStdDb)
  }
  const full = curve(mFull, jt)
  if (full === null) return { offsetMs: null, checks: { ...blank, windowOk: true, floor } }
  const halves = [curve(mHalf, jt - mHalf), curve(mHalf, jt)]

  const f = peakFeatures(lagsMs, full, loMs, hiMs, p.interpolate)
  const ratio = f.second > 1e-3 ? f.pk / f.second : Infinity
  const sharpness = f.pk - Math.max(interp(f.estI - p.sharpAtMs, lagsMs, full), interp(f.estI + p.sharpAtMs, lagsMs, full))
  const devs = []
  for (const c of halves) {
    if (c === null) continue
    const h = peakFeatures(lagsMs, c, f.estI - p.halfSearchMs, f.estI + p.halfSearchMs, p.interpolate)
    devs.push(h.edge ? Infinity : Math.abs(h.estI - f.estI))
  }
  const agree = devs.length ? Math.max(...devs) : Infinity
  const checks = {
    speech: true,
    windowOk: true,
    peak: f.pk,
    ratio,
    sharpness,
    halfDeviationMs: agree,
    edge: f.edge,
    candidateMs: f.estI,
    peakOk: f.pk >= p.minPeak,
    ratioOk: ratio >= p.minRatio,
    sharpnessOk: sharpness >= p.minSharpness,
    halvesOk: agree <= p.halfTolMs,
    interiorOk: !f.edge,
    floor,
  }
  checks.accepted = checks.peakOk && checks.ratioOk && checks.sharpnessOk && checks.halvesOk && checks.interiorOk
  return { offsetMs: checks.accepted ? f.estI : null, checks }
}

// ------------------------------------------------------------------ the turn's reference

function decodeBase64(text) {
  const binary = atob(text)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

/** Bytes as standard base64. */
export function encodeBase64(bytes) {
  let binary = ''
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(binary)
}

/**
 * A turn's envelope from its `turn_envelope` messages, placed by `first_index` (arrival order does
 * not matter). Indices no message covered are 0 (-96 dB) and counted in `missing`.
 *
 * @param {{first_index: number, values: string}[]} chunks The messages' bodies (values base64).
 * @returns {{values: Uint8Array, missing: number, overlaps: number}}
 */
export function reassembleEnvelope(chunks) {
  // A message with no values covers nothing (the processor never sends one; the analysis flags it).
  const parts = chunks.map(c => ({ at: c.first_index, bytes: decodeBase64(c.values) })).filter(p => p.bytes.length)
  const length = parts.reduce((n, p) => Math.max(n, p.at + p.bytes.length), 0)
  const values = new Uint8Array(length)
  const seen = new Uint8Array(length)
  let overlaps = 0
  for (const p of parts) {
    for (let i = 0; i < p.bytes.length; i++) {
      if (seen[p.at + i]) overlaps++
      seen[p.at + i] = 1
      values[p.at + i] = p.bytes[i]
    }
  }
  return { values, missing: seen.reduce((n, s) => n + (s ? 0 : 1), 0), overlaps }
}

/**
 * The previous turn's envelope, `joinMs` after its first sample this turn's (design §3.3): the
 * previous values, then floor (byte 0) up to the join, then this turn's. The join is rounded to
 * the hop grid. `ownOffset` is the index of this turn's index 0 in `values`.
 */
export function spliceReference(previous, own, joinMs, hopMs = HOP_MS) {
  const ownOffset = Math.max(0, Math.round(joinMs / hopMs))
  const values = new Uint8Array(ownOffset + own.length)
  values.set(previous.subarray(0, Math.min(previous.length, ownOffset)))
  values.set(own, ownOffset)
  return { values, ownOffset }
}

/**
 * Which earlier turn a turn's reference is spliced onto, and where (design §3.3), or null:
 * a chained turn (`prev_turn_id`) at `prev_turn_pcm_ms`; an unchained one with `gap_ms > 0` at the
 * previous turn's `turn_end.pcm_ms + gap_ms`; never with `gap_ms` 0 on an unchained turn, and never
 * onto a turn that was aborted (or, for an unchained turn, whose turn_end has not arrived).
 *
 * @param {{prevTurnId: string, prevTurnPcmMs: number, gapMs: number}} turn
 * @param {{id: string, endPcmMs: number|null, aborted: boolean}|null} previous The turn announced
 *   just before it (for a chained turn: the one it names).
 */
export function splicePlan(turn, previous) {
  if (!previous || previous.aborted) return null
  if (turn.prevTurnId) {
    if (previous.id !== turn.prevTurnId || !(turn.prevTurnPcmMs > 0)) return null
    return { kind: 'chained', from: previous.id, joinMs: turn.prevTurnPcmMs }
  }
  if (!(turn.gapMs > 0) || previous.endPcmMs == null) return null
  return { kind: 'gap', from: previous.id, joinMs: previous.endPcmMs + turn.gapMs }
}

// ------------------------------------------------------------------ the stall rule (dsp.TurnTracker)
//
// When the bot's host or the receiver stalls, the jitter buffer plays a concealment fade and then
// exact zeros, and the rest of the turn arrives late by about the stall. The stall rule is per-turn
// state on top of align() (unchanged): freeze the pointer while the received is silent where the
// reference speaks, confirm the new position once audio resumes, re-lock on the newer half-window,
// keep frozen time off the loss clock, and place a chained turn from its locked predecessor
// (hack/sync-sim/README.md, "The stall rule"). Term for term dsp.StallParams, dsp.TurnTracker,
// dsp.confirm_position, dsp.newer_half_candidate, dsp.chained_prior and dsp.ref_speech;
// tests/probe_check.mjs holds the tracker to the oracle's ticks and events.

/** The stall rule's fixed terms (dsp.StallParams; lengths in ms). STALL is dsp.STALL. */
export const STALL = Object.freeze({
  freeze: true,
  confirm: true,
  newerHalf: true,
  holdClock: true,
  chainPrior: true,
  confirmingSilentDb: -95,
  carryBracket: true,
  revertUnconfirmed: false,
  noshiftTolMs: 10,
  noshiftFromMs: 200,
  silentDb: -60,
  refSpeechDb: -50,
  freezeAfterMs: 20,
  freezeMaxMs: 2000,
  matchBackMs: 400,
  matchFrames: 3,
  matchMinDb: -60,
  matchTolDb: 4,
  confirmFromMs: 300,
  confirmUntilMs: 1200,
  confirmEveryMs: 50,
  confirmBackMs: 150,
  confirmFwdMs: 20,
  confirmFloorDb: -70,
  confirmMinPeak: 0.95,
  confirmMinRatio: 1.2,
  confirmMinSharpness: 0.03,
  confirmSharpAtMs: 20,
  halfSearchMs: 300,
  halfAgreeMs: 10,
  halfSilenceMs: 40,
  digitalSilenceDb: -95,
  lossMs: 2000,
  chainOnsetTolMs: 200,
})

/** Every part of the stall rule off (dsp.TODAY): today's rule plus the 2 s loss clock. */
export const TODAY = Object.freeze({
  ...STALL,
  freeze: false,
  confirm: false,
  newerHalf: false,
  holdClock: false,
  chainPrior: false,
  revertUnconfirmed: false,
  noshiftTolMs: 0,
})

/** Length of the longest run of true in `mask` (dsp.longest_run). */
export function longestRun(mask) {
  let best = 0
  let run = 0
  for (const m of mask) {
    run = m ? run + 1 : 0
    if (run > best) best = run
  }
  return best
}

/** The window's standard deviation after clipping at `floor` (numpy's std: population). */
function clippedStd(values, floor) {
  const n = values.length
  let mean = 0
  for (let i = 0; i < n; i++) mean += Math.max(values[i], floor)
  mean /= n
  let ss = 0
  for (let i = 0; i < n; i++) {
    const d = Math.max(values[i], floor) - mean
    ss += d * d
  }
  return Math.sqrt(ss / n)
}

/**
 * The newer 0.5 s of the window alone, searched within +/- `searchMs` of the prior
 * (dsp.newer_half_candidate). Same arguments and lag axis as align(). Returns the interpolated
 * peak (ms on the received axis, as align's offsetMs) when peak, ratio, sharpness and interior all
 * hold, else null.
 */
export function newerHalfCandidate(received, reference, priorIndex, params = DEFAULTS, searchMs = 300) {
  const p = { ...DEFAULTS, ...params }
  const hop = p.hopMs
  const jt = received.length
  const mHalf = Math.floor(pyRound(p.windowMs / hop) / 2)
  if (jt < mHalf) return null
  const history = []
  for (let j = Math.max(0, jt - Math.trunc(p.floorHistoryMs / hop)); j < jt; j++) history.push(received[j])
  const floor = floorDb(history, p)
  const priorMs = priorIndex * hop
  const loMs = priorMs - searchMs
  const hiMs = priorMs + searchMs
  const margin = 2 * hop + p.sharpAtMs
  const lagMin = Math.floor((loMs - margin) / hop)
  const lagMax = Math.ceil((hiMs + margin) / hop)
  const lags = []
  for (let l = lagMin; l <= lagMax; l++) lags.push(l)
  const pad = Math.max(lagMax - (jt - mHalf), jt - lagMin - reference.length, 0) + 1
  const refPadded = new Float64Array(reference.length + 2 * pad).fill(DB_MIN)
  for (let k = 0; k < reference.length; k++) refPadded[pad + k] = dequantise(reference[k])
  const window = new Float64Array(mHalf)
  for (let i = 0; i < mHalf; i++) window[i] = received[jt - mHalf + i]
  const c = nccCurve(window, refPadded, pad + jt - mHalf, lags, floor, p.gateStdDb)
  if (c === null) return null
  const lagsMs = lags.map(l => l * hop)
  const f = peakFeatures(lagsMs, c, loMs, hiMs, p.interpolate)
  const ratio = f.second > 1e-3 ? f.pk / f.second : Infinity
  const sharp = f.pk - Math.max(interp(f.estI - p.sharpAtMs, lagsMs, c), interp(f.estI + p.sharpAtMs, lagsMs, c))
  const ok = f.pk >= p.minPeak && ratio >= p.minRatio && sharp >= p.minSharpness && !f.edge
  return ok ? f.estI : null
}

/**
 * The confirm correlation (dsp.confirm_position): received frames [jRes, jEnd) against the turn's
 * own reference (dB) at every position l in [floor(lo), ceil(hi)] (received frame of the
 * reference's index 0), both clipped at confirmFloorDb. Accepted at an interior peak >=
 * confirmMinPeak with peak / second >= confirmMinRatio and sharpness >= confirmMinSharpness;
 * with `noshiftL`, a peak within noshiftTolMs of it is accepted on the peak alone while the
 * second-highest peak stays below confirmMinPeak (the no-shift check). Returns the interpolated
 * position or null.
 */
export function confirmPosition(env, ownDb, jRes, jEnd, lo, hi, stall = STALL, hopMs = HOP_MS, noshiftL = null, noshiftOnly = false) {
  const floor = stall.confirmFloorDb
  const m = jEnd - jRes
  if (m < 2) return null
  const w = new Float64Array(m)
  for (let i = 0; i < m; i++) w[i] = Math.max(env[jRes + i], floor)
  if (clippedStd(w, floor) < DEFAULTS.gateStdDb) return null
  const sa = stall.confirmSharpAtMs / hopMs
  const ext = Math.ceil(sa) + 1
  const a = Math.floor(lo)
  const b = Math.ceil(hi)
  if (b - a < 2) return null
  let wm = 0
  for (let i = 0; i < m; i++) wm += w[i]
  wm /= m
  let wss = 0
  for (let i = 0; i < m; i++) {
    w[i] -= wm
    wss += w[i] * w[i]
  }
  const ls = []
  const c = []
  const v = new Float64Array(m)
  for (let l = a - ext; l <= b + ext; l++) {
    let vm = 0
    for (let i = 0; i < m; i++) {
      const k = jRes + i - l
      v[i] = Math.max(k >= 0 && k < ownDb.length ? ownDb[k] : DB_MIN, floor)
      vm += v[i]
    }
    vm /= m
    let dot = 0
    let vss = 0
    for (let i = 0; i < m; i++) {
      const d = v[i] - vm
      dot += d * w[i]
      vss += d * d
    }
    const den = Math.sqrt(vss * wss)
    ls.push(l)
    c.push(den > 0 ? dot / den : 0)
  }
  const f = peakFeatures(ls, c, a, b, true)
  const ratio = f.second > 1e-3 ? f.pk / f.second : Infinity
  if (noshiftL !== null && stall.noshiftTolMs > 0 && f.pk >= stall.confirmMinPeak && f.second < stall.confirmMinPeak && Math.abs(f.estI - noshiftL) * hopMs <= stall.noshiftTolMs) return f.estI
  if (noshiftOnly) return null
  if (f.edge || f.pk < stall.confirmMinPeak || ratio < stall.confirmMinRatio) return null
  const side = Math.max(interp(f.estI - sa, ls, c), interp(f.estI + sa, ls, c))
  if (f.pk - side < stall.confirmMinSharpness) return null
  return f.estI
}

/**
 * Rule 5 (R74, dsp.chained_prior): the prior of a chained turn whose predecessor is locked, before
 * its own first lock. `chainL`: the predecessor's position + prev_turn_pcm_ms (frames); `onsetL`:
 * the onset prior (frames) or null. The onset prior when it lies within chainOnsetTolMs of the
 * chain position, else the chain position.
 */
export function chainedPrior(chainL, onsetL, stall = STALL, hopMs = HOP_MS) {
  if (onsetL !== null && Math.abs(onsetL - chainL) * hopMs <= stall.chainOnsetTolMs) return { frame: onsetL, kind: 'onset' }
  return { frame: chainL, kind: 'chain' }
}

/**
 * The SDK's refSpeech (dsp.ref_speech): whether the reference, where it is believed to sit (`l`:
 * received frame of the turn's own index 0; `ref` spliced with the turn's own index 0 at `ownOff`),
 * has speech in the received window [count - 1 s, count).
 */
export function refSpeech(ref, ownOff, l, count, floor, params = DEFAULTS) {
  const p = { ...DEFAULTS, ...params }
  const m = pyRound(p.windowMs / p.hopMs)
  const shift = -pyRound(l) + ownOff
  const v = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    const k = count - m + i + shift
    v[i] = k >= 0 && k < ref.length ? dequantise(ref[k]) : DB_MIN
  }
  return clippedStd(v, floor) >= p.gateStdDb
}

/**
 * One turn's alignment state (dsp.TurnTracker): today's rule (align per tick, hold, the 2 s loss
 * clock) plus the stall rule (STALL). Feed every received frame once, in order (frame(); the
 * envelope value at index j must exist), and run tick() at the cadence with the reference as
 * arrived. Positions (`l`) are received frames of the turn's own reference index 0.
 *
 * `events` lists { frame, event } with the SDK's sync event shapes: { kind: 'frozen', turnId,
 * atMs } when a freeze starts (atMs: the turn's own audio position, ms, where the pointer stopped)
 * and { kind: 'unfrozen', turnId, frozenMs, confirmed } when its episode closes (the confirm
 * accepted: true; a full or newer-half lock, the confirm's time ran out, no confirm possible, a new
 * freeze, the turn's end, or the 2 s cap: false). `lostAt` lists the frames where the loss clock
 * reached lossMs (sync-lost); `freezes` keeps each closed episode.
 */
export class TurnTracker {
  constructor(turnId, params = DEFAULTS, stall = STALL) {
    this.turnId = turnId
    this.p = { ...DEFAULTS, ...params }
    this.s = { ...STALL, ...stall }
    const hop = this.p.hopMs
    this.hop = hop
    const s = this.s
    this.nMin = Math.max(1, pyRound(s.freezeAfterMs / hop))
    this.nMax = pyRound(s.freezeMaxMs / hop)
    this.nBack = pyRound(s.matchBackMs / hop)
    this.nFrom = pyRound(s.confirmFromMs / hop)
    this.nUntil = pyRound(s.confirmUntilMs / hop)
    this.nEvery = Math.max(1, pyRound(s.confirmEveryMs / hop))
    this.nNoshift = pyRound(s.noshiftFromMs / hop)
    this.nHalfSilence = pyRound(s.halfSilenceMs / hop)
    this.ownDb = new Float64Array(0)
    this.locked = false
    this.l = null
    this.heldNoLock = 0
    this.unlockedMs = 0
    this.lost = false
    this.lostAt = []
    this.events = []
    this.freezes = []
    this.run = 0
    this.runK = null
    this.episode = null
    this.pending = null
    this.prevHalf = null
    this.frozenSinceTick = 0
  }

  /** The turn's own reference envelope as arrived so far (bytes); re-read when its length changes. */
  setReference(own) {
    if (own.length !== this.ownDb.length) this.ownDb = Float64Array.from(own, dequantise)
  }

  /** Whether the pointer is held now. */
  get frozen() {
    const ep = this.episode
    return ep !== null && ep.resume === null && ep.frames < this.nMax
  }

  matches(env, l, j) {
    const k = pyRound(j - l)
    return k >= 0 && k < this.ownDb.length && j >= 0 && env[j] >= this.s.matchMinDb && Math.abs(env[j] - this.ownDb[k]) <= this.s.matchTolDb
  }

  close(j, how, extra = {}) {
    const ep = this.episode
    if (ep === null) return
    Object.assign(ep, { how, closed: j }, extra)
    if (!ep.capped) this.emitUnfrozen(j, ep, how === 'confirm') // a capped freeze sent its unfrozen at the cap
    this.freezes.push(ep)
    this.episode = null
    this.pending = null
  }

  emitUnfrozen(j, ep, confirmed) {
    this.events.push({ frame: j, event: { kind: 'unfrozen', turnId: this.turnId, frozenMs: ep.frames * this.hop, confirmed } })
  }

  /** At the cap the pointer advances again and unfrozen (not confirmed) fires; the episode stays open for the confirm. */
  checkCap(j) {
    const ep = this.episode
    if (ep !== null && !ep.capped && ep.frames >= this.nMax) {
      ep.capped = true
      this.emitUnfrozen(j, ep, false)
    }
  }

  /** Received frame j (env[j], dB) has completed. */
  frame(env, j) {
    if (!(this.locked && this.s.freeze)) return
    const s = this.s
    const level = this.pending === null ? s.silentDb : Math.min(s.silentDb, s.confirmingSilentDb)
    const silent = env[j] <= level
    if (silent) {
      if (this.run < this.nMin) {
        // the run that triggers a freeze: every frame silent where the reference at the pointer
        // (still advancing) has speech
        const k = pyRound(j - this.l)
        if (k >= 0 && k < this.ownDb.length && this.ownDb[k] > s.refSpeechDb) {
          if (this.run === 0) this.runK = k
          this.run++
          if (this.run === this.nMin) this.start(env, j)
        } else {
          this.run = 0
        }
      } else {
        this.run++
        const ep = this.episode
        if (ep !== null && ep.resume === null && ep.frames < this.nMax) {
          this.l += 1
          ep.frames++
          this.frozenSinceTick++
          this.checkCap(j)
        }
      }
    } else {
      if (this.run >= this.nMin && this.episode !== null) this.resume(j)
      this.run = 0
    }
    const pd = this.pending
    if (pd === null) return
    const nPost = j + 1 - pd.jRes
    if (nPost > this.nUntil) {
      if (s.revertUnconfirmed) {
        // unconfirmed: assume the timeline did not move (a burst loss)
        this.l = this.episode.lPre
        this.close(j, 'expired', { reverted: true })
      } else {
        this.close(j, 'expired')
      }
    } else if (s.noshiftTolMs > 0 && this.nNoshift <= nPost && nPost < this.nFrom && (nPost - this.nNoshift) % this.nEvery === 0) {
      // before the confirm proper: only "nothing moved" (a burst loss) is tested
      const est = confirmPosition(env, this.ownDb, pd.jRes, j + 1, pd.lo, pd.hi, s, this.hop, this.episode.lPre, true)
      if (est !== null) this.confirmed(j, est, nPost)
    } else if (nPost >= this.nFrom && (nPost - this.nFrom) % this.nEvery === 0) {
      const first = nPost === this.nFrom
      const est = confirmPosition(env, this.ownDb, pd.jRes, j + 1, pd.lo, pd.hi, s, this.hop, first ? this.episode.lPre : null)
      if (est !== null) this.confirmed(j, est, nPost)
    }
  }

  confirmed(j, est, nPost) {
    this.l = est
    this.unlockedMs = 0
    this.lost = false
    this.close(j, 'confirm', { confirmL: est, confirmAfterMs: nPost * this.hop })
  }

  start(env, j) {
    const j0 = j - this.nMin + 1
    let carried = null
    if (this.episode !== null) {
      const ep = this.episode
      if (this.pending !== null && this.s.carryBracket) carried = [ep.lPre, ep.jMatch]
      this.close(j, 'superseded')
    }
    let lPre
    let jMatch = null
    if (carried !== null) {
      // unconfirmed: the bracket still starts at the last frame that matched before the earlier
      // freeze, so the confirm can still choose "nothing moved" or "everything since moved"
      ;[lPre, jMatch] = carried
    } else {
      lPre = this.l
      const lowest = Math.max(j0 - this.nBack, this.s.matchFrames - 1)
      for (let jm = j0 - 1; jm >= lowest; jm--) {
        let all = true
        for (let d = 0; d < this.s.matchFrames && all; d++) all = this.matches(env, lPre, jm - d)
        if (all) {
          jMatch = jm
          break
        }
      }
    }
    const n = Math.min(this.nMin, this.nMax)
    this.l += n
    this.frozenSinceTick += n
    this.episode = { start: j0, atMs: this.runK * this.hop, lPre, jMatch, frames: n, resume: null, carried: carried !== null }
    this.events.push({ frame: j, event: { kind: 'frozen', turnId: this.turnId, atMs: this.runK * this.hop } })
    this.checkCap(j)
  }

  resume(j) {
    const ep = this.episode
    ep.resume = j
    if (!this.s.confirm) this.close(j, 'no-confirm')
    else if (ep.jMatch === null) this.close(j, 'no-match')
    else this.pending = { jRes: j, lo: ep.lPre - this.s.confirmBackMs / this.hop, hi: ep.lPre + (j - ep.jMatch - 1) + this.s.confirmFwdMs / this.hop }
  }

  /** The turn's ticks are over: close an open freeze. */
  finish(j) {
    this.close(j, 'end')
  }

  /**
   * One alignment tick on received frames [0, count) (dsp.TurnTracker.tick). `ref`: the reference
   * as correlated (spliced: the turn's own index 0 at `ownOff`); `centre`/`searchMs`: the prior and
   * its search before the first lock (after it the tracker uses its own position); `sinceMs`:
   * received time since the previous tick that returned a record. Returns null when the window is
   * not complete yet (not a tick), else the tick's record.
   */
  tick(env, count, ref, ownOff, centre, searchMs, sinceMs) {
    const p = this.p
    const s = this.s
    const hop = this.hop
    if (this.locked) {
      centre = this.l
      searchMs = p.searchMs
    }
    const hist = pyRound(p.floorHistoryMs / hop)
    const base = Math.max(0, count - hist)
    const rec = env.subarray(base, count)
    const pp = { ...p, searchMs }
    const r = align(rec, ref, centre - ownOff - base, pp)
    if (!r.checks.windowOk) return null
    let since = sinceMs
    if (s.holdClock) since = Math.max(0, since - this.frozenSinceTick * hop)
    this.frozenSinceTick = 0
    const c = r.checks
    let accepted = r.offsetMs !== null
    let lAcc = accepted ? r.offsetMs / hop + ownOff + base : null
    let via = accepted ? 'full' : null
    let halfL = null
    let halfRefused = false
    if (!accepted && s.newerHalf && this.locked && c.speech) {
      let hc = newerHalfCandidate(rec, ref, centre - ownOff - base, pp, s.halfSearchMs)
      if (hc !== null) {
        const mHalf = Math.floor(pyRound(p.windowMs / hop) / 2)
        const tail = rec.subarray(rec.length - mHalf)
        if (longestRun(Array.from(tail, x => x <= s.digitalSilenceDb)) >= this.nHalfSilence) {
          hc = null
          halfRefused = true
        }
      }
      if (hc !== null) {
        halfL = hc / hop + ownOff + base
        if (this.prevHalf !== null && Math.abs(halfL - this.prevHalf) * hop <= s.halfAgreeMs) {
          accepted = true
          lAcc = halfL
          via = 'half'
        }
        this.prevHalf = halfL
      } else {
        this.prevHalf = null
      }
    } else if (accepted) {
      this.prevHalf = null
    }
    if (accepted) {
      this.unlockedMs = 0
      this.lost = false
    } else if (!c.speech) {
      const lNow = this.locked ? this.l : centre
      if (refSpeech(ref, ownOff, lNow, count, floorDb(rec, p), p)) this.unlockedMs += since
    } else {
      this.unlockedMs += since
    }
    if (this.unlockedMs >= s.lossMs && !this.lost) {
      this.lost = true
      this.lostAt.push(count)
    }
    if (accepted) {
      if (this.episode !== null) this.close(count - 1, via)
      this.run = 0
      this.locked = true
      this.l = lAcc
      this.heldNoLock = 0
    } else if (c.speech && !this.locked) {
      this.heldNoLock++
    }
    return {
      count,
      speech: c.speech,
      accepted,
      via,
      l: lAcc,
      candidateL: c.candidateMs == null ? null : c.candidateMs / hop + ownOff + base,
      halfL,
      halfRefused,
      checks: c,
      unlockedMs: this.unlockedMs,
      lost: this.lost,
    }
  }
}

// ------------------------------------------------------------------ the live tracker

const finiteOrNull = x => (Number.isFinite(x) ? x : null)

/** A tick's checks as plain JSON (an infinite ratio or deviation as the string "Infinity"). */
function plainChecks(c) {
  return {
    speech: c.speech,
    peak: finiteOrNull(c.peak),
    ratio: c.ratio === Infinity ? 'Infinity' : finiteOrNull(c.ratio),
    sharpness: finiteOrNull(c.sharpness),
    half_deviation_ms: c.halfDeviationMs === Infinity ? 'Infinity' : finiteOrNull(c.halfDeviationMs),
    edge: c.edge,
    peak_ok: c.peakOk,
    ratio_ok: c.ratioOk,
    sharpness_ok: c.sharpnessOk,
    halves_ok: c.halvesOk,
    interior_ok: c.interiorOk,
  }
}

/** A closed or open freeze episode as plain JSON (dsp.TurnTracker.freezes' keys). */
function plainEpisode(ep) {
  const out = { start: ep.start, at_ms: ep.atMs, l_pre: ep.lPre, j_match: ep.jMatch, frames: ep.frames, resume: ep.resume, carried: ep.carried }
  if (ep.capped) out.capped = true
  if (ep.how !== undefined) Object.assign(out, { how: ep.how, closed: ep.closed })
  if (ep.confirmL !== undefined) Object.assign(out, { confirm_l: ep.confirmL, confirm_after_ms: ep.confirmAfterMs })
  if (ep.reverted) out.reverted = true
  return out
}

/**
 * The rule on a live track, per turn, as the web SDK runs it. Feed it the received audio and the
 * avatar messages in arrival order; it ticks every turn that has a reference: every 50 ms of new
 * audio until the turn's first lock, then every 200 ms. Before the first lock the prior is, for a
 * chained turn whose predecessor has locked (rule 5, R74), the predecessor's position +
 * prev_turn_pcm_ms, re-anchored on the onset prior only within 200 ms of it; else the
 * onset-detected position when the caller has given one (setOnsetPrior); else, for an unchained
 * turn with gap_ms > 0 whose previous turn had ended (turn_end) and was not aborted when this
 * turn's turn_start arrived, the previous turn's place + its turn_end.pcm_ms + gap_ms (R72: the
 * processor released exactly that much, as a chain does); else turn_start arrival + 200 ms. After
 * the first lock it is the turn's tracked position. After 10 ticks with speech and no lock the
 * search widens to +/- 600 ms around the onset-detected position (R76: a chained turn's too).
 *
 * Each turn's state is a TurnTracker: with the stall rule (STALL, the default; `stall: TODAY`
 * switches it off) the tracked position freezes while the received is silent where the reference
 * speaks, a confirm or a newer-half re-lock moves it after a stall, and frozen time stays off the
 * 2 s loss clock. As the oracle's tick loop (replay_capture.replay_turn_tracked) runs it, a turn's
 * received frames go through its TurnTracker at its ticks, every frame up to the tick's last
 * complete one first, then the tick; a turn whose position another turn's prior needs (rule 5,
 * R72) has its frames up to now fed at that moment, so a freeze in progress counts.
 *
 * Positions are samples of the received track since the tracker started ("tap" positions).
 */
export class SyncTracker {
  constructor(sampleRate, { params = DEFAULTS, cadence = CADENCE, highPassHz = HIGH_PASS_HZ, stall = STALL } = {}) {
    this.rate = sampleRate
    this.params = { ...DEFAULTS, ...params }
    this.cadence = { ...CADENCE, ...cadence }
    this.stall = { ...STALL, ...stall }
    this.env = new ReceivedEnvelope(sampleRate, { highPassHz })
    this.hop = this.env.hop // received samples per envelope frame
    this.turns = new Map()
    this.order = [] // turn ids in turn_start order
    this.ticks = 0 // ticks recorded, all turns
  }

  /** Samples received so far. */
  get pos() {
    return this.env.samples
  }

  samples(ms) {
    return Math.round((ms * this.rate) / 1000)
  }

  /** One avatar message (`data` of the RTVI server message); `tap` is the position it arrived at. */
  message(data, tap) {
    const [kind] = Object.keys(data).filter(k => k !== 'tag')
    const body = data[kind]
    if (!body || typeof body !== 'object' || typeof body.turn_id !== 'string') return
    const turn = this.turns.get(body.turn_id)
    if (kind === 'turn_start') {
      if (turn) return
      const previous = this.order.length ? this.turns.get(this.order[this.order.length - 1]) : null
      const gapMs = body.gap_ms || 0
      // R72 (web-sdk ba14a15): decided when turn_start arrives, as the web SDK decides it.
      const gapFrom = !body.prev_turn_id && gapMs > 0 && previous && previous.endPcmMs !== null && previous.abortTap === null
        ? { id: previous.id, joinMs: previous.endPcmMs + gapMs }
        : null
      this.turns.set(body.turn_id, {
        id: body.turn_id,
        startTap: tap,
        sampleRate: body.sample_rate,
        onsetsMs: Array.isArray(body.onsets_ms) ? body.onsets_ms.slice() : [],
        prevTurnId: body.prev_turn_id || '',
        prevTurnPcmMs: body.prev_turn_pcm_ms || 0,
        gapMs,
        previous: previous ? previous.id : null,
        gapFrom, // R72: the turn whose place + joinMs is this turn's prior, or null
        chunks: [],
        reference: null, // cached reassembly
        endPcmMs: null,
        endTap: null,
        abortTap: null,
        onsetPrior: null,
        tracker: new TurnTracker(body.turn_id, this.params, this.stall),
        lastFed: null, // the last received frame fed to the tracker
        placeL: null, // the tracker's position when its last tick ended (dsp: tr.l before a tick's frames)
        lastTickPos: null, // the tap of the last tick that returned a record (the loss clock's step)
        eventsSeen: 0, // tracker events already attached to a tick record
        nextDue: tap,
        done: false,
        ticks: [],
        splice: null,
      })
      this.order.push(body.turn_id)
      return
    }
    if (!turn) return
    if (kind === 'turn_envelope') {
      turn.chunks.push({ first_index: body.first_index, values: body.values, hop_ms: body.hop_ms, window_ms: body.window_ms })
      turn.reference = null
    } else if (kind === 'turn_end') {
      turn.endPcmMs = body.pcm_ms
      turn.endTap = tap
    } else if (kind === 'turn_abort') {
      turn.abortTap = tap
    }
  }

  /** The turn's first sample per the onset rule (tap position), or null; the prior until the first lock. */
  setOnsetPrior(turnId, pos) {
    const turn = this.turns.get(turnId)
    if (turn) turn.onsetPrior = Number.isFinite(pos) ? pos : null
  }

  /**
   * The turn's received frames up to the last complete one through its TurnTracker (frames from
   * turn_abort on are not fed). Only after the turn's first tick has fed (as the oracle's loop).
   * A successor's chain prior can feed its predecessor here, ahead of the predecessor's own next
   * tick. Each frame is fed once either way; it only changes which instant's reference the frame
   * sees, so this matches the oracle's per-tick feeding as long as the envelope arrives ahead of the
   * audio (the processor sends it before the room plays the samples).
   */
  feed(turn) {
    if (turn.done || turn.lastFed === null) return
    const all = this.env.values()
    const count = all.length
    const abortedBy = turn.abortTap === null ? null : framesAt(turn.abortTap, this.hop)
    turn.tracker.setReference(this.ownReference(turn).values)
    for (let j = turn.lastFed + 1; j < count; j++) if (abortedBy === null || j < abortedBy) turn.tracker.frame(all, j)
    turn.lastFed = Math.max(turn.lastFed, count - 1)
  }

  /** Where a locked turn's first sample is now (received frames, freezes included), or null before its first lock. */
  trackedFrame(turn) {
    if (!turn.tracker.locked) return null
    this.feed(turn)
    return turn.tracker.l
  }

  /** Rule 5: a chained turn's predecessor's tracked position + prev_turn_pcm_ms (frames), or null. */
  chainFrame(turn) {
    if (!this.stall.chainPrior || !turn.prevTurnId || !(turn.prevTurnPcmMs > 0)) return null
    const prev = this.turns.get(turn.prevTurnId)
    if (!prev || !this.ownReference(prev).values.length) return null
    const l = this.trackedFrame(prev)
    return l === null ? null : l + pyRound((turn.prevTurnPcmMs * this.rate) / 1000) / this.hop
  }

  /**
   * Where the turn's first sample is believed to be now, received frames (the web SDK's frame 0),
   * and what puts it there (PRIOR_KINDS, recorded per tick as `prior_kind`): the tracked position
   * once locked ('locked'); else, for a chained turn whose predecessor has locked, rule 5's place
   * ('chain', or 'onset' when the onset prior is within 200 ms of it); else the onset prior
   * ('onset'; a chained turn's place from the analysis comes this way too); else (R72) the previous
   * turn's place + the gap turn's join ('gap'); else turn_start arrival + 200 ms ('arrival').
   * `wide` is the centre of the widened search (the onset prior, else a chain place), or null.
   *
   * Two simplifications against web-sdk ba14a15, which matter for outliers only: the web SDK
   * places a gap turn only when the previous turn has a frame 0, and re-anchors it only on an
   * onset within 200 ms of that place; here the previous turn always has a place (at worst its own
   * arrival prior), and any onset prior the page's analysis sets takes over.
   */
  prior(turn) {
    if (turn.tracker.locked) return { frame: this.trackedFrame(turn), kind: 'locked', wide: null }
    const onset = turn.onsetPrior !== null ? turn.onsetPrior / this.hop : null
    const chain = this.chainFrame(turn)
    if (chain !== null) {
      const { frame, kind } = chainedPrior(chain, onset, this.stall, this.params.hopMs)
      // R76: once the turn widens, the +/- 600 ms search centres on the onset prior when there is one
      return { frame, kind, wide: onset !== null ? onset : frame }
    }
    if (onset !== null) return { frame: onset, kind: 'onset', wide: onset }
    if (turn.gapFrom) return { frame: this.place(this.turns.get(turn.gapFrom.id)) + this.samples(turn.gapFrom.joinMs) / this.hop, kind: 'gap', wide: null }
    return { frame: (turn.startTap + this.samples(this.cadence.priorLeadMs)) / this.hop, kind: 'arrival', wide: null }
  }

  /** The frame of prior(). */
  place(turn) {
    return this.prior(turn).frame
  }

  /** The turn's own envelope, reassembled by index. */
  ownReference(turn) {
    if (!turn.reference) turn.reference = reassembleEnvelope(turn.chunks)
    return turn.reference
  }

  /** The reference the rule correlates for `turn`: its own, or spliced onto the previous turn's. */
  reference(turn) {
    const own = this.ownReference(turn)
    const prevId = turn.prevTurnId || turn.previous
    const prev = prevId ? this.turns.get(prevId) : null
    const plan = splicePlan(turn, prev ? { id: prev.id, endPcmMs: prev.endPcmMs, aborted: prev.abortTap !== null } : null)
    if (!plan) return { values: own.values, ownOffset: 0, splice: null }
    const { values, ownOffset } = spliceReference(this.ownReference(prev).values, own.values, plan.joinMs)
    return { values, ownOffset, splice: { ...plan, ownOffset } }
  }

  /** Feed received audio (Web Audio floats, or an Int16Array); runs every tick that fell due. */
  audio(block) {
    this.env.push(block)
    const pos = this.pos
    for (const id of this.order) {
      const turn = this.turns.get(id)
      if (!turn.done && pos >= turn.nextDue) this.tick(turn, pos)
    }
  }

  /** The turn's ticks are over: close its open freeze at the last frame it was fed. */
  end(turn) {
    turn.done = true
    if (turn.lastFed !== null) turn.tracker.finish(turn.lastFed)
  }

  /** The track has ended: close every turn's open freeze (as the oracle's replay does at the capture's end). */
  finish() {
    for (const id of this.order) {
      const turn = this.turns.get(id)
      if (!turn.done) this.end(turn)
    }
  }

  tick(turn, pos) {
    const hopMs = this.params.hopMs
    const cadence = this.cadence
    const tracker = turn.tracker
    const own = this.ownReference(turn)
    // A turn with nothing to correlate yet waits; it is due again on the next block.
    if (!own.values.length) return
    const { frame: priorL, kind: placeKind, wide: wideAt } = tracker.locked ? { frame: null, kind: 'locked', wide: null } : this.prior(turn)
    // Before this tick's frames: where the tracker had the turn when its last tick ended (the
    // oracle's tr.l; another turn's prior may have fed frames since), else the prior.
    const place = tracker.locked ? turn.placeL : priorL
    // Finished: the whole 1 s window lies past the turn's end (or 1 s past an abort).
    const windowStart = pos / this.hop - this.params.windowMs / hopMs
    if (turn.abortTap !== null && pos > turn.abortTap + this.samples(this.params.windowMs)) return void this.end(turn)
    if (turn.endPcmMs !== null && windowStart > place + turn.endPcmMs / hopMs) return void this.end(turn)

    // Every received frame up to the tick's last complete one goes through the tracker first.
    const all = this.env.values()
    const count = all.length
    if (turn.lastFed === null) turn.lastFed = count - 2
    this.feed(turn)

    const ref = this.reference(turn)
    turn.splice = ref.splice
    const wide = !tracker.locked && tracker.heldNoLock >= cadence.widenAfterTicks
    const centre = wide && wideAt !== null ? wideAt : place
    const priorKind = tracker.locked ? 'locked' : wide && turn.onsetPrior !== null ? 'onset' : placeKind
    const searchMs = wide ? cadence.wideSearchMs : this.params.searchMs
    const sinceMs = turn.lastTickPos === null ? 0 : ((pos - turn.lastTickPos) / this.rate) * 1000
    const r = tracker.tick(all, count, ref.values, ref.ownOffset, centre, searchMs, sinceMs)
    turn.placeL = tracker.l
    if (r === null) {
      // Less than one window of received envelope: nothing to decide, nothing to record.
      turn.nextDue = pos + this.samples(cadence.beforeLockMs)
      return
    }
    turn.lastTickPos = pos
    const toPos = l => (l === null ? null : l * this.hop)
    const events = tracker.events.slice(turn.eventsSeen).map(e => ({ frame: e.frame, ...e.event }))
    turn.eventsSeen = tracker.events.length
    const record = {
      pos,
      prior_pos: centre * this.hop,
      prior_kind: priorKind,
      onset_prior_pos: turn.onsetPrior,
      search_ms: searchMs,
      gate: r.speech,
      accepted: r.accepted,
      via: r.via,
      first_sample_pos: toPos(r.l),
      candidate_pos: toPos(r.candidateL),
      half_pos: toPos(r.halfL),
      half_refused: r.halfRefused,
      tracked_pos: tracker.locked ? toPos(tracker.l) : null,
      frozen: tracker.frozen,
      events,
      loss_clock_ms: r.unlockedMs,
      lost: r.lost,
      checks: plainChecks(r.checks),
    }
    turn.ticks.push(record)
    this.ticks++
    turn.nextDue = pos + this.samples(tracker.locked ? cadence.afterLockMs : cadence.beforeLockMs)
  }

  /**
   * A turn's freezes for the table: each { atMs, frozenMs, confirmed } in order, `confirmed` null
   * for one still open (its frozenMs so far).
   */
  freezesOf(turn) {
    const out = []
    for (const { event } of turn.tracker.events) {
      if (event.kind === 'frozen') out.push({ atMs: event.atMs, frozenMs: 0, confirmed: null })
      else if (event.kind === 'unfrozen' && out.length && out[out.length - 1].confirmed === null) Object.assign(out[out.length - 1], { frozenMs: event.frozenMs, confirmed: event.confirmed })
    }
    const ep = turn.tracker.episode
    if (out.length && out[out.length - 1].confirmed === null && ep) out[out.length - 1].frozenMs = ep.frames * this.params.hopMs
    // a capped freeze said unfrozen at the cap; a confirm after it moves the pointer silently
    return out
  }

  /** What the probe's analysis reads per turn: { [turnId]: { ticks, splice, referenceLength, missing, freezes, syncLost } }. */
  summary() {
    const out = {}
    for (const [id, turn] of this.turns) {
      const own = this.ownReference(turn)
      out[id] = { ticks: turn.ticks, splice: turn.splice, referenceLength: own.values.length, missing: own.missing, overlaps: own.overlaps, freezes: this.freezesOf(turn), syncLost: turn.tracker.lostAt.length }
    }
    return out
  }

  /** The paired capture's per-turn record: the reference as reassembled and spliced, every tick, and the stall rule's events. */
  export() {
    const turns = this.order.map(id => {
      const turn = this.turns.get(id)
      const own = this.ownReference(turn)
      const ref = this.reference(turn)
      const tr = turn.tracker
      return {
        turn_id: id,
        start_tap: turn.startTap,
        sample_rate: turn.sampleRate,
        onsets_ms: turn.onsetsMs,
        prev_turn_id: turn.prevTurnId,
        prev_turn_pcm_ms: turn.prevTurnPcmMs,
        gap_ms: turn.gapMs,
        end_pcm_ms: turn.endPcmMs,
        end_tap: turn.endTap,
        abort_tap: turn.abortTap,
        hop_ms: HOP_MS,
        window_ms: WINDOW_MS,
        reference: encodeBase64(own.values),
        reference_missing: own.missing,
        splice: ref.splice && { kind: ref.splice.kind, from: ref.splice.from, join_ms: ref.splice.joinMs, own_offset: ref.ownOffset, values: encodeBase64(ref.values) },
        onset_prior_pos: turn.onsetPrior,
        ticks: turn.ticks,
        events: tr.events.map(e => ({ frame: e.frame, tap: (e.frame + 2) * this.hop, ...e.event })),
        freezes: tr.freezes.map(plainEpisode).concat(tr.episode ? [{ ...plainEpisode(tr.episode), open: true }] : []),
        lost_at_frames: tr.lostAt.slice(),
      }
    })
    return {
      about: 'pipecat-spatialreal probe: per-turn reference envelopes and alignment ticks (examples/probe/README.md)',
      rate: this.rate,
      hop_samples: this.hop,
      high_pass_hz: this.env.alpha === null ? null : HIGH_PASS_HZ,
      params: this.params,
      cadence: this.cadence,
      stall: this.stall,
      turns,
    }
  }
}

/** Complete envelope frames once `pos` samples have arrived (fully-inside windows; replay_capture.frames_at). */
export function framesAt(pos, hop) {
  return Math.max(0, Math.floor(pos / hop) - 1)
}
