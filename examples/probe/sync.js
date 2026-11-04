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
// the prior and the widened search before the first lock (steps 4 and 10, with R57 and R72), and
// the splice of the previous turn's envelope (§3.3). It never moves anything: it records what each
// tick saw.
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

/**
 * The rule on a live track, per turn, as the web SDK runs it. Feed it the received audio and the
 * avatar messages in arrival order; it ticks every turn that has a reference: every 50 ms of new
 * audio until the turn's first lock, then every 200 ms. Before the first lock the prior is the
 * onset-detected position when the caller has given one (setOnsetPrior); else, for an unchained
 * turn with gap_ms > 0 whose previous turn had ended (turn_end) and was not aborted when this
 * turn's turn_start arrived, the previous turn's place + its turn_end.pcm_ms + gap_ms (R72: the
 * processor released exactly that much, as a chain does); else turn_start arrival + 200 ms. After
 * the first lock it is the last accepted alignment. After 10 ticks with speech and no lock the
 * search widens to +/- 600 ms around the onset-detected position.
 *
 * Positions are samples of the received track since the tracker started ("tap" positions).
 */
export class SyncTracker {
  constructor(sampleRate, { params = DEFAULTS, cadence = CADENCE, highPassHz = HIGH_PASS_HZ } = {}) {
    this.rate = sampleRate
    this.params = { ...DEFAULTS, ...params }
    this.cadence = { ...CADENCE, ...cadence }
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
        locked: false,
        lastL: null, // last accepted alignment: received frame of the turn's index 0
        heldNoLock: 0,
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
   * Where the turn's first sample is believed to be now, received frames (the web SDK's frame 0):
   * the last accepted alignment, else the onset prior, else (R72) the previous turn's place + the
   * gap turn's join, else turn_start arrival + 200 ms.
   */
  place(turn) {
    if (turn.locked) return turn.lastL
    if (turn.onsetPrior !== null) return turn.onsetPrior / this.hop
    if (turn.gapFrom) return this.place(this.turns.get(turn.gapFrom.id)) + this.samples(turn.gapFrom.joinMs) / this.hop
    return (turn.startTap + this.samples(this.cadence.priorLeadMs)) / this.hop
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

  tick(turn, pos) {
    const hopMs = this.params.hopMs
    const cadence = this.cadence
    const own = this.ownReference(turn)
    const atFrame = tapPos => tapPos / this.hop
    // A turn with nothing to correlate yet waits; it is due again on the next block.
    if (!own.values.length) return
    const place = this.place(turn)
    // Finished: the whole 1 s window lies past the turn's end (or 1 s past an abort).
    const windowStart = atFrame(pos) - this.params.windowMs / hopMs
    if (turn.abortTap !== null && pos > turn.abortTap + this.samples(this.params.windowMs)) return void (turn.done = true)
    if (turn.endPcmMs !== null && windowStart > place + turn.endPcmMs / hopMs) return void (turn.done = true)

    const ref = this.reference(turn)
    turn.splice = ref.splice
    const wide = !turn.locked && turn.heldNoLock >= cadence.widenAfterTicks
    const centre = wide && turn.onsetPrior !== null ? atFrame(turn.onsetPrior) : place
    const searchMs = wide ? cadence.wideSearchMs : this.params.searchMs
    const all = this.env.values()
    const base = Math.max(0, all.length - Math.trunc(this.params.floorHistoryMs / hopMs))
    const r = align(all.subarray(base), ref.values, centre - ref.ownOffset - base, { ...this.params, searchMs })
    if (!r.checks.windowOk) {
      // Less than one window of received envelope: nothing to decide, nothing to record.
      turn.nextDue = pos + this.samples(cadence.beforeLockMs)
      return
    }
    const accepted = r.offsetMs !== null
    const record = {
      pos,
      prior_pos: centre * this.hop,
      search_ms: searchMs,
      gate: r.checks.speech,
      accepted,
      first_sample_pos: accepted ? (r.offsetMs / hopMs + ref.ownOffset + base) * this.hop : null,
      candidate_pos: r.checks.candidateMs == null ? null : (r.checks.candidateMs / hopMs + ref.ownOffset + base) * this.hop,
      checks: plainChecks(r.checks),
    }
    turn.ticks.push(record)
    this.ticks++
    if (accepted) {
      turn.locked = true
      turn.lastL = r.offsetMs / hopMs + ref.ownOffset + base
    } else if (r.checks.speech && !turn.locked) {
      // Toward the widening: ticks before the first lock whose window passed the speech gate and
      // were not accepted. Gated ticks (silence) and too-short windows do not count.
      turn.heldNoLock++
    }
    turn.nextDue = pos + this.samples(turn.locked ? cadence.afterLockMs : cadence.beforeLockMs)
  }

  /** What the probe's analysis reads per turn: { [turnId]: { ticks, splice, referenceLength, missing } }. */
  summary() {
    const out = {}
    for (const [id, turn] of this.turns) {
      const own = this.ownReference(turn)
      out[id] = { ticks: turn.ticks, splice: turn.splice, referenceLength: own.values.length, missing: own.missing, overlaps: own.overlaps }
    }
    return out
  }

  /** The paired capture's per-turn record: the reference as reassembled and spliced, and every tick. */
  export() {
    const turns = this.order.map(id => {
      const turn = this.turns.get(id)
      const own = this.ownReference(turn)
      const ref = this.reference(turn)
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
      }
    })
    return {
      about: 'pipecat-spatialreal probe: per-turn reference envelopes and alignment ticks (examples/probe/README.md)',
      rate: this.rate,
      hop_samples: this.hop,
      high_pass_hz: this.env.alpha === null ? null : HIGH_PASS_HZ,
      params: this.params,
      cadence: this.cadence,
      turns,
    }
  }
}
