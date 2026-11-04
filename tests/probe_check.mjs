// Runs the analysis code of examples/probe/index.html (its <script id="analysis"> block) and the
// alignment rule next to it (examples/probe/sync.js) under Node. tests/test_probe_page.py runs it:
// usage `node probe_check.mjs <page> <fixture.json>`, where the fixture holds envelope messages
// built by pipecat_spatialreal.envelope and random audio with the onsets the package's
// OnsetTracker finds in it. The rule is checked against the normative oracle's own numbers,
// tests/fixtures/sync/oracle.json (written by make_oracle_fixture.py from hack/sync-sim/dsp.py).
import fs from 'node:fs'
import path from 'node:path'
import vm from 'node:vm'
import assert from 'node:assert/strict'
import { pathToFileURL } from 'node:url'

const html = fs.readFileSync(process.argv[2], 'utf8')
const match = html.match(/<script id="analysis">([\s\S]*?)<\/script>/)
assert(match, 'the analysis script is not in the page')
const sandbox = { atob: globalThis.atob }
vm.createContext(sandbox)
const api = vm.runInContext(match[1] + '\n;({ analyze, summarize, OnsetDetector, toPcm16, reassemble, decodeBase64, rowCells })', sandbox)
const { analyze, summarize, OnsetDetector, toPcm16 } = api

const FIXTURE = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'))
const E = FIXTURE.envelopes
const A = E.A_start.turn_start.turn_id, B = E.B_start.turn_start.turn_id, C = E.C_start.turn_start.turn_id
const RATE = 48000
const samples = ms => Math.round(ms * RATE / 1000)
const clone = x => JSON.parse(JSON.stringify(x))
const plain = x => JSON.parse(JSON.stringify(x))
let n = 0
const test = (name, fn) => { try { fn(); console.log('ok   ', name); n++ } catch (e) { console.log('FAIL ', name, '\n', e.message); process.exitCode = 1 } }

// --- the onset rule matches the processor's: leading run counts, threshold is strict, >= ceil(150 ms) of quiet
test('detector: leading onset, strict threshold, silent-run length', () => {
  const d = new OnsetDetector(RATE, 328, 150)
  const block = new Float32Array(48000)
  block[100] = 0.01                 // pcm16 327: quiet
  block[200] = 0.02                 // pcm16 655: audible -> the leading onset
  assert.deepEqual(plain(d.push(block)), [200])
  // strict: a sample AT the threshold is quiet (0.5 -> 16383), one above it is audible
  const d1 = new OnsetDetector(RATE, 16383, 150)
  const b1 = new Float32Array(10); b1[3] = 0.5; b1[6] = 0.51
  assert.deepEqual(plain(d1.push(b1)), [6])
  // 7199 quiet samples then audible: not an onset (min is ceil(0.15*48000)=7200); 7200 quiet: onset
  const d2 = new OnsetDetector(RATE, 328, 150)
  const b2 = new Float32Array(20000)
  b2[0] = 0.5
  b2[1 + 7199] = 0.5                // 7199 quiet in between
  b2[1 + 7199 + 1 + 7200] = 0.5     // 7200 quiet in between
  assert.deepEqual(plain(d2.push(b2)), [0, 1 + 7199 + 1 + 7200])
  // the rule holds across block boundaries
  const d3 = new OnsetDetector(RATE, 328, 150)
  const first = new Float32Array(5000); first[0] = 0.5
  const second = new Float32Array(5000); second[4999] = 0.5     // 9999 quiet between: an onset
  assert.deepEqual([plain(d3.push(first)), plain(d3.push(second))], [[0], [4999]])
  // the minimum silence rounds up: 151 ms at 44.1 kHz is 6659.1 samples, so 6660 quiet ones are needed
  const d4 = new OnsetDetector(44100, 328, 151)
  const b4 = new Float32Array(20000)
  b4[0] = 0.5
  b4[1 + 6659] = 0.5                // 6659 quiet in between: not an onset
  b4[1 + 6659 + 1 + 6660] = 0.5     // 6660 quiet in between: an onset
  assert.deepEqual(plain(d4.push(b4)), [0, 1 + 6659 + 1 + 6660])
  assert.equal(toPcm16(1), 32767); assert.equal(toPcm16(-1), -32768); assert.equal(toPcm16(0.01), 327)
})

function goodSession() {
  const log = [], onsets = [], markers = []
  const T0 = 1_700_000_000_000
  const rec = (t, data, tap = 0) => log.push({ t: T0 + t, tap, size: 3500, data: clone(data) })
  const P0 = 480000
  // Turn A: lead 180 ms, onsets_ms [180, 1400]
  rec(1000, E.A_start)
  E.A_seg0.forEach((c, i) => rec(1003 + i, c))
  onsets.push({ pos: P0, wall: T0 + 1180, level: 0.2 })
  rec(1100, E.A_seg1[0])
  onsets.push({ pos: P0 + samples(1400 - 180), wall: T0 + 2400, level: 0.2 })
  rec(3200, E.A_end)
  // Turn B chained to A (prev pcm 3000, o0 = 0): first onset lands at P0 - 180 ms + 3000 ms + 0 ms
  rec(3150, E.B_start)
  rec(3152, E.B_seg0[0])
  onsets.push({ pos: P0 - samples(180) + samples(3000), wall: T0 + 4000, level: 0.1 })
  rec(5000, E.B_end)
  // Turn C: aborted after the barge-in
  rec(8000, E.C_start)
  E.C_seg0.forEach((c, i) => rec(8001 + i, c))
  onsets.push({ pos: P0 + 700000, wall: T0 + 8300, level: 0.3 })
  markers.push({ t: T0 + 9000, type: 'user-started-speaking' })
  rec(9080, E.C_abort)
  return { log, onsets, markers, rate: RATE, T0 }
}

test('a good session: three turns pass, nothing else flagged', () => {
  const s = goodSession()
  const r = analyze(s)
  assert.deepEqual(plain(r.problems), [])
  assert.equal(r.strays.length, 0)
  assert.equal(r.turns.length, 3)
  const [a, b, c] = r.turns
  assert.equal(a.status, 'pass', a.fails.join('; '))
  assert.equal(b.status, 'pass', b.fails.join('; '))
  assert.equal(c.status, 'pass', c.fails.join('; '))
  assert.equal(a.lead, 180); assert.equal(a.skew, 0); assert.equal(a.o0, 180)
  assert.equal(a.seg0.chunks, 3); assert.equal(a.seg0.count, 3); assert.equal(a.seg0.bytes, 5120); assert.equal(a.seg0.complete, true)
  assert.equal(a.seg0.margin, 1180 - 1005)
  assert.deepEqual(plain(a.laterErrs.map(Math.round)), [0])
  assert.equal(Math.round(b.chainErr), 0)
  assert.equal(b.lead, 850)
  assert.equal(c.abort.sinceUser, 80)
  assert.equal(a.end.pcmMs, 3000); assert.equal(a.end.segments, 2)
})

/** A session in which turn A's segment 0 arrives as chunks 2, 0, 1. */
function segment0ArrivingOutOfOrder() {
  const s = goodSession()
  const ofSegment0 = r => r.data.frames && r.data.frames.turn_id === A && r.data.frames.segment_seq === 0
  const [c0, c1, c2] = s.log.filter(ofSegment0)
  assert.deepEqual([c0, c1, c2].map(c => c.data.frames.chunk_index), [0, 1, 2])
  // Each chunk keeps its chunk_index; only the order of arrival changes.
  const times = [c0.t, c1.t, c2.t]
  s.log.splice(s.log.indexOf(c0), 3, { ...c2, t: times[0] }, { ...c0, t: times[1] }, { ...c1, t: times[2] })
  const arrival = s.log.filter(ofSegment0)
  assert.deepEqual(arrival.map(c => c.data.frames.chunk_index), [2, 0, 1])
  return { s, arrival }
}
const SEGMENT0 = Buffer.from(Uint8Array.from({ length: 5120 }, (_, i) => i % 256))   // E.A_seg0's bytes

test('analyze() joins a segment\'s chunks in chunk_index order, not in the order they arrive', () => {
  const { s } = segment0ArrivingOutOfOrder()
  const result = analyze(s)
  assert.deepEqual(plain(result.problems), [])
  const [segment0, segment1] = result.turns[0].segments
  assert.equal(segment0.complete, true)
  assert.deepEqual(Buffer.from(segment0.wire), SEGMENT0)
  assert.deepEqual(plain([segment0.seq, segment0.chunks, segment0.count, segment0.bytes, segment0.end]), [0, 3, 3, 5120, false])
  assert.deepEqual(plain([segment1.seq, segment1.chunks, segment1.count, segment1.bytes, segment1.end]), [1, 1, 1, 1000, true])
})

test('reassemble() joins by chunk_index from a Map filled in arrival order, and not before every chunk is there', () => {
  const { arrival } = segment0ArrivingOutOfOrder()
  const filled = new Map(arrival.map(c => [c.data.frames.chunk_index, api.decodeBase64(c.data.frames.data)]))
  assert.deepEqual([...filled.keys()], [2, 0, 1])               // insertion order is arrival order
  assert.deepEqual(Buffer.from(api.reassemble({ count: 3, chunks: filled })), SEGMENT0)
  filled.delete(1)
  assert.equal(api.reassemble({ count: 3, chunks: filled }), null)
})

test('bad numbers are called out', () => {
  const s = goodSession()
  // turn A: turn_start only 50 ms ahead of its onset; segment 0 finishes after the onset;
  // the second onset is heard 60 ms late
  s.log.find(r => r.data.turn_start && r.data.turn_start.turn_id === A).t = s.T0 + 1130
  s.log.filter(r => r.data.frames && r.data.frames.turn_id === A && r.data.frames.segment_seq === 0).at(-1).t = s.T0 + 1190
  s.onsets[1].pos += samples(60)
  const a = analyze(s).turns[0]
  assert.equal(a.status, 'fail')
  assert(a.fails.some(f => /only 50 ms/.test(f)), a.fails.join('|'))
  assert(a.fails.some(f => /segment 0 was not whole/.test(f)), a.fails.join('|'))
  assert(a.fails.some(f => /onset 1 is 60 ms off/.test(f)), a.fails.join('|'))
})

test('a late turn_start (after the onset) gives a negative lead and fails', () => {
  const s = goodSession()
  s.log.find(r => r.data.turn_start && r.data.turn_start.turn_id === A).t = s.T0 + 1300
  const a = analyze(s).turns[0]
  assert(a.lead < 0 && a.status === 'fail', JSON.stringify(a.fails))
})

test('abort slower than 300 ms fails; no marker is a note, not a failure', () => {
  const s = goodSession()
  s.log.find(r => r.data.turn_abort).t = s.T0 + 9400
  assert(analyze(s).turns[2].fails.some(f => /400 ms after the user/.test(f)))
  s.markers.length = 0
  const c = analyze(s).turns[2]
  assert.equal(c.status, 'pass'); assert(c.notes.length === 1)
})

test('an aborted turn does not swallow the next turn\'s onset', () => {
  const s = goodSession()
  const D = 'dddddddd-4444-4444-8444-444444444444'
  const startD = clone(E.C_start); startD.turn_start.turn_id = D; startD.turn_start.onsets_ms = [100]
  s.log.push({ t: s.T0 + 12000, tap: 0, size: 3000, data: startD })
  const seg = clone(E.C_seg0[0]); seg.frames.turn_id = D
  s.log.push({ t: s.T0 + 12002, tap: 0, size: 3000, data: seg })
  s.onsets.push({ pos: 2_000_000, wall: s.T0 + 12150, level: 0.2 })
  const r = analyze(s)
  assert.equal(r.turns[3].lead, 150)
  assert.equal(r.turns[2].lead, 300)   // C kept its own
})

test('an onset no turn accounts for is reported', () => {
  const s = goodSession()
  s.onsets.push({ pos: 9_999_999, wall: s.T0 + 20000, level: 0.5 })
  const r = analyze(s)
  assert.equal(r.strays.length, 1)
  assert(r.problems.some(p => /no turn_start accounts/.test(p.text)))
})

test('a degraded turn_start and a degraded state are visible', () => {
  const s = goodSession()
  s.log.find(r => r.data.turn_start && r.data.turn_start.turn_id === C).data.turn_start.degraded = true
  s.log.push({ t: s.T0 + 9500, tap: 0, size: 100, data: E.degraded })
  const r = analyze(s)
  assert(r.turns[2].fails.includes('turn_start.degraded'))
  assert.equal(r.states.length, 1); assert.equal(r.states[0].status, 'STATUS_DEGRADED')
})

test('contract: 1-based chunk_index, chunk_count 0, end differing, duplicates, late frames, missing keys', () => {
  const s = goodSession()
  const frames = s.log.filter(r => r.data.frames && r.data.frames.turn_id === A && r.data.frames.segment_seq === 0)
  frames[0].data.frames.chunk_index = 1          // 1-based by mistake: now two chunks claim index 1
  frames[1].data.frames.end = true               // end differs between chunks of one segment
  delete frames[2].data.frames.segment_seq       // a missing key
  s.log.push({ t: s.T0 + 3300, tap: 0, size: 3000, data: clone(E.A_seg1[0]) })  // frames after turn_end
  const zero = clone(E.B_seg0[0]); zero.frames.chunk_count = 0
  s.log.push({ t: s.T0 + 5100, tap: 0, size: 3000, data: zero })
  const text = analyze(s).problems.map(p => p.text).join('\n')
  for (const re of [/chunk 1 arrived twice/, /end differs/, /segment_seq is missing/, /frames after turn_end/, /chunk_count 0, at least 1/]) {
    assert(re.test(text), `${re} not in:\n${text}`)
  }
})

test('contract: chunk_index counted from 1 is out of range, and the segment never completes', () => {
  const s = goodSession()
  const chunks = s.log.filter(r => r.data.frames && r.data.frames.turn_id === A && r.data.frames.segment_seq === 0)
  chunks.forEach(r => { r.data.frames.chunk_index += 1 })          // 1, 2, 3 of 3
  const result = analyze(s)
  assert(result.problems.some(p => /chunk_index 3 outside 0\.\.2 \(0-based\)/.test(p.text)), JSON.stringify(result.problems))
  assert(result.turns[0].fails.some(f => /segment 0 was not whole/.test(f)), result.turns[0].fails.join('|'))
})

test('contract: payload member rules, empty turn_onsets, state unspecified, oversize message', () => {
  const s = goodSession()
  s.log.push({ t: s.T0 + 100, tap: 0, size: 5000, data: { tag: 'spatialreal-avatar', turn_abort: { turn_id: 'x' }, state: { status: 'STATUS_OK', reason: '' } } })
  s.log.push({ t: s.T0 + 101, tap: 0, size: 100, data: { tag: 'spatialreal-avatar', turn_onsets: { turn_id: A, onsets_ms: [] } } })
  s.log.push({ t: s.T0 + 102, tap: 0, size: 100, data: { tag: 'spatialreal-avatar', state: { status: 'STATUS_UNSPECIFIED', reason: '' } } })
  const text = analyze(s).problems.map(p => p.text).join('\n')
  for (const re of [/2 payload members/, /exceeds Daily/, /empty list/, /STATUS_UNSPECIFIED/]) assert(re.test(text), `${re} not in:\n${text}`)
})

test('contract: segment numbering and the end flag', () => {
  const s = goodSession()
  const seg = clone(E.A_seg1[0]); seg.frames.segment_seq = 3        // 0, 1, 3: a gap
  s.log.push({ t: s.T0 + 1150, tap: 0, size: 3000, data: seg })
  const text = analyze(s).problems.map(p => p.text).join('\n')
  assert(/segment_seq runs 0, 1, 3/.test(text), text)
  assert(/turn_end counts 2 segments, 3 arrived/.test(text), text)
  // the clone keeps end=true, so now segments 1 and 3 both carry it
  assert(/aaaaaaaa: end is set on segments 1, 3; only the last carries it/.test(text), text)
})

test('contract: a segment that follows the one carrying end', () => {
  const s = goodSession()
  const after = clone(E.A_seg1[0]); after.frames.segment_seq = 2; after.frames.end = false   // 0, 1 (end), 2
  s.log.push({ t: s.T0 + 1150, tap: 0, size: 3000, data: after })
  const text = analyze(s).problems.map(p => p.text).join('\n')
  assert(/aaaaaaaa: segment 2 follows the end segment 1/.test(text), text)
  assert(!/end is set on segments/.test(text), text)          // one end flag only: the other rule stays quiet
})

test('contract: chunk_count must not change inside a segment', () => {
  const s = goodSession()
  const chunks = s.log.filter(r => r.data.frames && r.data.frames.turn_id === A && r.data.frames.segment_seq === 0)
  chunks[2].data.frames.chunk_count = 4                       // chunks 0 and 1 said 3
  const text = analyze(s).problems.map(p => p.text).join('\n')
  assert(/aaaaaaaa segment 0: chunk_count changes from 3 to 4/.test(text), text)
})

test('turn_onsets add to the announced onsets and are paired', () => {
  const s = goodSession()
  // C announces a second onset at 900 ms after turn_start; the tap hears it exactly 600 ms after the first
  s.log.push({ t: s.T0 + 8050, tap: 0, size: 100, data: E.C_onsets })
  s.onsets.push({ pos: 2_000_000 + 0, wall: s.T0 + 8900, level: 0.2 })
  s.onsets.splice(3, 0)   // keep order by pos/wall below
  s.onsets.sort((a, b) => a.wall - b.wall)
  s.onsets.find(o => o.wall === s.T0 + 8900).pos = s.onsets.find(o => o.wall === s.T0 + 8300).pos + samples(600)
  const c = analyze(s).turns[2]
  assert.deepEqual(plain(c.laterErrs.map(Math.round)), [0])
})

test('no onset ever heard: ended turns fail, an aborted one does not', () => {
  const s = goodSession()
  s.onsets.length = 0
  const rows = analyze(s).turns
  assert.equal(rows[0].status, 'fail')       // A already ended without ever being heard
  assert.equal(rows[1].status, 'fail')       // B is chained to A, which was not heard either: nothing places it
  assert(rows[1].fails.some(f => /no onset was matched to it, and it cannot be placed: it follows aaaaaaaa, which was not heard/.test(f)), rows[1].fails.join('|'))
  assert.equal(rows[2].status, 'pass')       // C was cut off by a barge-in before anything was heard
})

// --- Chained turns. The processor chains a turn (prev_turn_id) only when its first sample plays right
// after the previous turn's last one: no pause, nothing in between. Any other turn is announced
// unchained and released after min_silence_ms of quiet. A fresh onset tracker starts with every turn,
// so a chained turn that begins inside the previous one's sound announces onsets_ms[0], while the tap
// runs on across the join and finds an onset only after min_silence_ms of quiet: there may be none to
// hear. The end of the previous turn then places the turn, exactly, and nothing confirms the place.

/** goodSession() in which the tap hears no onset at the join with turn B (its own is not found). */
function sessionWithoutBsOnset() {
  const s = goodSession()
  const [heard] = s.onsets.splice(2, 1)
  assert.equal(heard.wall, s.T0 + 4000)          // that was B's first onset
  return s
}
const turnStart = (s, id) => s.log.find(r => r.data.turn_start && r.data.turn_start.turn_id === id).data.turn_start
const D = 'dddddddd-4444-4444-8444-444444444444'
/** Add turn D (not chained, onsets_ms [100]) to a session: turn_start at 5100, its onset heard at 5400. */
function addTurnD(s) {
  const start = clone(E.C_start); start.turn_start.turn_id = D; start.turn_start.onsets_ms = [100]
  const seg = clone(E.C_seg0[0]); seg.frames.turn_id = D; seg.frames.chunk_count = 1
  s.log.push({ t: s.T0 + 5100, tap: 0, size: 3000, data: start }, { t: s.T0 + 5102, tap: 0, size: 3000, data: seg })
  s.log.sort((a, b) => a.t - b.t)
  s.onsets.splice(2, 0, { pos: 480000 + 300000, wall: s.T0 + 5400, level: 0.2 })
}

test('a chained turn with no onset of its own is placed exactly from the end of the turn before it, and unverified', () => {
  const r = analyze(sessionWithoutBsOnset())
  const b = r.turns[1]
  assert.equal(b.status, 'unverified', b.fails.join('; '))       // not a pass: nothing in the audio confirms the place
  assert.deepEqual(plain(b.fails), [])
  assert.equal(b.placed, true)
  // A's first onset is at 1180 ms and A starts 180 ms earlier and is 3000 ms long, so B's first sample is at 4000 ms
  // and B's onsets_ms[0] is 0: exactly 4000 ms. There is no pause between the turns, so there is no slack.
  assert.equal(b.lead, 850)                        // turn_start at 3150
  assert.equal(b.seg0.margin, 848)                 // its last chunk at 3152
  assert.equal(b.skew, null)
  assert(b.notes.some(n => /placed from the end of the turn before it/.test(n)), b.notes.join('|'))
  assert.equal(b.seg0.complete, true)
  assert.equal(r.strays.length, 0)
  assert.deepEqual(plain(r.problems), [])
  assert(r.events.some(e => /\[bbbbbbbb\] first onset placed from the end of the turn before it, 850 ms after turn_start/.test(e.text)))
  assert.equal(r.turns[2].lead, 300)               // C keeps its own onset
})

test('the summary counts a placed turn as unverified, not as a pass, and is not green for it', () => {
  const clean = summarize(analyze(goodSession()))
  assert.equal(clean.text, '3 turns: 3 pass, 0 fail, 0 unverified, 0 pending. state degraded: 0. contract problems: 0. stray onsets: 0.')
  assert.equal(clean.tone, 'ok')
  const placed = summarize(analyze(sessionWithoutBsOnset()))
  assert.equal(placed.text, '3 turns: 2 pass, 0 fail, 1 unverified, 0 pending. state degraded: 0. contract problems: 0. stray onsets: 0.')
  assert.equal(placed.tone, 'muted')               // neither green nor red
  const s = sessionWithoutBsOnset()
  s.onsets.push({ pos: 9_999_999, wall: s.T0 + 20000, level: 0.5 })   // a stray makes it bad whatever else is true
  assert.equal(summarize(analyze(s)).tone, 'bad')
})

test('a placed chained turn keeps its later onsets: they are not taken for its first', () => {
  const s = sessionWithoutBsOnset()
  turnStart(s, B).onsets_ms = [0.0, 600.0]        // a second onset 600 ms in, which the tap does hear
  s.onsets.splice(2, 0, { pos: 480000 - samples(180) + samples(3000) + samples(600), wall: s.T0 + 4600, level: 0.1 })
  const r = analyze(s)
  const b = r.turns[1]
  assert.equal(b.placed, true)
  assert.deepEqual(plain(b.laterErrs.map(Math.round)), [0])
  assert.equal(b.status, 'unverified', b.fails.join('; '))   // still: its first onset was not heard
  assert.equal(r.strays.length, 0)
  assert.equal(r.turns[2].lead, 300)
})

test('a placed turn takes an onset near its place only, not the next turn\'s', () => {
  const s = sessionWithoutBsOnset()
  addTurnD(s)                                      // D's onset is inside B's time window, 3.4 s from B's place
  const r = analyze(s)
  const b = r.turns[1]
  const d = r.turns.find(t => t.id === D)
  assert.equal(b.placed, true)
  assert.equal(d.lead, 300)
  assert.equal(d.placed, false)                    // D's own onset, heard
  assert.equal(r.strays.length, 0)
})

test('a chained turn that cannot be placed takes no onset: a later turn keeps its own', () => {
  const s = sessionWithoutBsOnset()
  turnStart(s, B).onsets_ms = []                   // B announced no onsets: nothing to place it by
  addTurnD(s)                                      // D's onset lies inside B's time window
  const r = analyze(s)
  const d = r.turns.find(t => t.id === D)
  assert.equal(d.lead, 300)                        // D got it; the time window alone would have given it to B
  assert.deepEqual(plain(d.fails), [])
  assert.equal(d.status, 'pending')                // D has not ended yet
  assert.equal(r.turns[1].status, 'fail')
  assert.equal(r.strays.length, 0)
})

test('a chained turn that cannot be placed fails with the real reason, not that a turn was not measured', () => {
  const reasonOf = (s, i = 1) => { const row = analyze(s).turns[i]; assert.equal(row.status, 'fail', row.fails.join('; ')); return row.fails.join('; ') }
  const prefix = 'no onset was matched to it, and it cannot be placed: it '

  const noOnsets = sessionWithoutBsOnset()
  turnStart(noOnsets, B).onsets_ms = []
  assert.equal(reasonOf(noOnsets), prefix + 'announced no onsets')

  const unknown = sessionWithoutBsOnset()
  turnStart(unknown, B).prev_turn_id = 'ffffffff-6666-4666-8666-666666666666'
  assert.equal(reasonOf(unknown), prefix + 'follows ffffffff, which was never announced')

  const prevNoOnsets = sessionWithoutBsOnset()
  turnStart(prevNoOnsets, A).onsets_ms = []        // A announced none (and its two onsets are not accounted for)
  assert.equal(reasonOf(prevNoOnsets), prefix + 'follows aaaaaaaa, which announced no onsets')

  const notHeard = goodSession()
  notHeard.onsets.length = 0                       // nothing is heard at all
  assert.equal(reasonOf(notHeard), prefix + 'follows aaaaaaaa, which was not heard')

  // C ends and is chained to B, which is chained to A, which was not heard: the reason runs back along the chain
  const deep = goodSession()
  deep.onsets.length = 0
  deep.log.splice(deep.log.findIndex(r => r.data.turn_abort), 1)
  deep.log.push({ t: deep.T0 + 9000, tap: 0, size: 100, data: { tag: 'spatialreal-avatar', turn_end: { turn_id: C, pcm_ms: 2000.0, segment_count: 1 } } })
  turnStart(deep, C).prev_turn_id = B
  turnStart(deep, C).prev_turn_pcm_ms = 1500.0
  assert.equal(reasonOf(deep, 2), prefix + 'follows bbbbbbbb, which follows aaaaaaaa, which was not heard')

  for (const s of [noOnsets, unknown, prevNoOnsets, notHeard, deep]) {
    assert(!analyze(s).turns.some(row => row.fails.some(f => /not measured/.test(f))))
  }
})

test('a turn that was never announced is a failed row, and the summary counts it', () => {
  const s = goodSession()
  const ghost = 'eeeeeeee-5555-4555-8555-555555555555'
  s.log.push({ t: s.T0 + 12000, tap: 0, size: 100, data: { tag: 'spatialreal-avatar', turn_abort: { turn_id: ghost } } })
  const r = analyze(s)
  const row = r.turns.find(t => t.id === ghost)
  assert.equal(row.status, 'fail')
  assert.deepEqual(plain(row.fails), ['no turn_start'])
  assert.deepEqual(plain(summarize(r).counts), { pass: 3, fail: 1, unverified: 0, pending: 0 })
  assert.equal(summarize(r).counts.pass + summarize(r).counts.fail, r.turns.length)   // every row is counted somewhere
})

test('an unchained turn that was not heard says whether it announced onsets', () => {
  const silent = goodSession()
  silent.onsets.length = 0
  turnStart(silent, A).onsets_ms = []              // A announced none and none was heard
  const a = analyze(silent).turns[0]
  assert.equal(a.status, 'fail')
  assert(a.fails.includes('no audible onset was heard (it announced none either)'), a.fails.join('|'))

  const deaf = goodSession()
  deaf.onsets.length = 0                           // A announced onsets and none was heard
  const heardNothing = analyze(deaf).turns[0]
  assert(heardNothing.fails.includes('no audible onset was heard'), heardNothing.fails.join('|'))
})

test('an unchained turn announced before the previous turn has ended is paired by its own onset', () => {
  // When another sound sits between two turns the second is not chained, and its turn_start can arrive
  // before the first turn's turn_end: the processor holds its release until the room has been quiet.
  const s = goodSession()
  const b = turnStart(s, B)
  b.prev_turn_id = ''
  b.prev_turn_pcm_ms = 0
  s.log.find(r => r.data.turn_start && r.data.turn_start.turn_id === B).t = s.T0 + 2900
  s.log.sort((x, y) => x.t - y.t)
  const startOf = id => s.log.find(r => r.data.turn_start && r.data.turn_start.turn_id === id).t
  const endOf = id => s.log.find(r => r.data.turn_end && r.data.turn_end.turn_id === id).t
  assert(startOf(B) < endOf(A))                    // the premise: B was announced while A had not yet ended
  const r = analyze(s)
  assert.deepEqual(plain(r.problems), [])
  assert.equal(r.strays.length, 0)
  const [a, bRow] = r.turns
  assert.equal(a.status, 'pass', a.fails.join('; '))
  assert.equal(bRow.status, 'pass', bRow.fails.join('; '))
  assert.equal(bRow.lead, 1100)                    // its own onset at 4000, its turn_start at 2900
  assert.equal(bRow.placed, false)
  assert.equal(bRow.chainErr, undefined)           // not chained: nothing to compare with
  assert.deepEqual(plain(a.laterErrs.map(Math.round)), [0])   // A's later onset still belongs to A
})

test('a chained onset that is heard but off its place is paired and failed; far off it is a stray', () => {
  for (const off of [60, 150]) {
    const s = goodSession()
    s.onsets[2].pos += samples(off)
    s.onsets[2].wall += off
    const r = analyze(s)
    assert.equal(r.strays.length, 0, `${off} ms`)
    assert.equal(Math.round(r.turns[1].chainErr), off)
    assert(r.turns[1].fails.some(f => new RegExp(`chained onset is ${off} ms off`).test(f)), `${off} ms: ${r.turns[1].fails}`)
  }
  const far = goodSession()                        // 260 ms is beyond CHAIN_TOL_MS (200): not B's onset at all
  far.onsets[2].pos += samples(260)
  far.onsets[2].wall += 260
  const r = analyze(far)
  assert.equal(r.strays.length, 1)
  assert.equal(r.turns[1].placed, true)
})

test('a chain that points at itself or loops does not hang the analysis, and the reason says so', () => {
  const s = goodSession()
  s.onsets.splice(2)                               // only A's two onsets are heard, B and C are not
  turnStart(s, A).prev_turn_id = A                 // A follows itself (A is heard, so nothing recurses)
  turnStart(s, B).prev_turn_id = C                 // B follows C and C follows B: neither is heard
  turnStart(s, C).prev_turn_id = B
  const r = analyze(s)
  assert.equal(r.turns.length, 3)
  assert(r.turns[1].fails.some(f => /follows cccccccc, which follows bbbbbbbb, which is part of a loop of chained turns/.test(f)), r.turns[1].fails.join('|'))
})

test('the page\'s own notes are on the timeline', () => {
  const s = goodSession()
  s.notes = [{ t: s.T0 + 500, text: "tapping the bot's track" }, { t: s.T0 + 2000, text: "the bot's audio track was replaced" }]
  const lines = analyze(s).events.map(e => e.text)
  assert(lines.includes("page: tapping the bot's track"))
  assert(lines.indexOf("page: the bot's audio track was replaced") > lines.findIndex(l => /\[aaaaaaaa\] turn_start/.test(l)))
})

test('detector: finds the onsets the processor\'s OnsetTracker finds, block by block', () => {
  const { pcm, rate, threshold, min_silence_ms: minSilenceMs, onsets } = FIXTURE.audio
  const bytes = Buffer.from(pcm, 'base64')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length)
  const ints = Int16Array.from({ length: bytes.length / 2 }, (_, i) => view.getInt16(i * 2, true))
  const detector = new OnsetDetector(rate, threshold, minSilenceMs)
  const found = []
  for (let at = 0; at < ints.length; at += 128) {
    const block = Float32Array.from(ints.subarray(at, at + 128), v => v / 32768)
    for (const i of detector.push(block)) found.push(at + i)
  }
  assert(onsets.length >= 8, `the fixture should have several onsets, has ${onsets.length}`)
  assert.deepEqual(found, onsets)
})

// ------------------------------------------------------------------ the turn envelope in the analysis

/** The session of goodSession() with turn A's envelope: E.A_env (2 560 values in two messages). */
function withEnvelope(s, chunks = E.A_env) {
  const at = s.log.findIndex(r => r.data.turn_start && r.data.turn_start.turn_id === A)
  chunks.forEach((c, i) => s.log.splice(at + 1 + i, 0, { t: s.log[at].t + 1, tap: 0, size: 3400, data: clone(c) }))
  return s
}

test('turn_envelope: whole and in order is no problem; its keys, hop, coverage and timing are checked', () => {
  const ok = analyze(withEnvelope(goodSession()))
  assert.deepEqual(plain(ok.problems), [])
  assert(ok.events.some(e => /\[aaaaaaaa\] turn_envelope {2}2 messages, 2560 values \(12\.8 s\)/.test(e.text)), ok.events.map(e => e.text).join('\n'))
  // Out of order is fine: the values are placed by first_index.
  assert.deepEqual(plain(analyze(withEnvelope(goodSession(), [E.A_env[1], E.A_env[0]])).problems), [])
  const problems = chunks => analyze(withEnvelope(goodSession(), chunks)).problems.map(p => p.text).join('\n')
  assert.match(problems([E.A_env[1]]), /turn_envelope leaves 2400 of indices 0\.\.2559 uncovered at turn_end/)
  assert.match(problems([E.A_env[0], E.A_env[0]]), /turn_envelope covers 2400 indices more than once/)
  const hop = clone(E.A_env[0]); hop.turn_envelope.hop_ms = 10
  assert.match(problems([hop, E.A_env[1]]), /hop_ms 10, window_ms 10; 5 and 10 expected/)
  const missing = clone(E.A_env[0]); delete missing.turn_envelope.window_ms
  assert.match(problems([missing, E.A_env[1]]), /turn_envelope\.window_ms is missing/)
  const late = goodSession()
  late.log.push({ t: late.T0 + 9500, tap: 0, size: 300, data: clone(E.C_env[0]) })
  assert.match(analyze(late).problems.map(p => p.text).join('\n'), /cccccccc: turn_envelope after turn_abort/)
  const noGap = goodSession()
  delete turnStart(noGap, A).gap_ms
  assert.match(analyze(noGap).problems.map(p => p.text).join('\n'), /turn_start\.gap_ms is missing/)
})

// ------------------------------------------------------------------ sync.js against the oracle

const sync = await import(pathToFileURL(path.join(path.dirname(path.resolve(process.argv[2])), 'sync.js')).href)
const ORACLE = JSON.parse(fs.readFileSync(new URL('./fixtures/sync/oracle.json', import.meta.url), 'utf8'))
const ORACLE_PCM = (() => {
  const bytes = Buffer.from(ORACLE.pcm_int16_le, 'base64')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length)
  return Int16Array.from({ length: bytes.length / 2 }, (_, i) => view.getInt16(i * 2, true))
})()
const oracleNumber = x => (x === 'Infinity' ? Infinity : x)

/** The turn's source of an oracle case (make_oracle_fixture.build_source): the samples, tiled, or held. */
function oracleSource(c) {
  const spec = c.source
  if (!spec) return ORACLE_PCM
  if (spec.kind === 'repeat') return Int16Array.from({ length: spec.total }, (_, n) => ORACLE_PCM[spec.from + (n % spec.length)])
  if (spec.kind === 'hold') return Int16Array.from({ length: spec.total }, (_, n) => ORACLE_PCM[Math.floor(n / spec.factor)])
  throw new Error(`unknown source ${spec.kind}`)
}

/** The case's reference: the oracle's envelope_bytes of its source, at its rate. */
const oracleReference = c => sync.envelopeBytes(oracleSource(c), c.rate ?? ORACLE.rate)

/** make_oracle_fixture.lcg_noise: (state >> 16) % (2A + 1) - A, the 32-bit LCG state updated first. */
function lcgNoise(count, amplitude, seed) {
  const out = new Int32Array(count)
  let state = seed >>> 0
  for (let i = 0; i < count; i++) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0
    out[i] = ((state >>> 16) % (2 * amplitude + 1)) - amplitude
  }
  return out
}

/** The received track of an oracle case, rebuilt from the same samples (make_oracle_fixture.build_received). */
function oracleReceived(c) {
  const source = oracleSource(c)
  const src = c.silent ? new Int16Array(source.length) : c.reverse ? source.slice().reverse() : source
  const out = new Int16Array(c.samples)
  const bounds = c.segments.map(g => g.from).concat([c.samples])
  c.segments.forEach((g, i) => {
    const own = g.reverse ? src.slice().reverse() : src
    for (let n = g.from; n < bounds[i + 1]; n++) {
      const k = n - g.lead
      if (k >= 0 && k < own.length) out[n] = own[k] >> (g.shift ?? 0)
    }
  })
  if (c.noise) {
    const noise = lcgNoise(c.samples, c.noise.amplitude, c.noise.seed)
    for (let n = 0; n < c.samples; n++) out[n] = Math.max(-32768, Math.min(32767, out[n] + noise[n]))
  }
  return out
}

/** A case's params in the port's names (the oracle's are snake_case: min_peak -> minPeak). */
const oracleParams = c => Object.fromEntries(Object.entries(c.params ?? {}).map(([k, v]) => [k.replace(/_([a-z])/g, (_, ch) => ch.toUpperCase()), v]))

/** The received envelope (dB) of an oracle case: given, or measured on the rebuilt track. */
const oracleReceivedDb = (c, opts) => (c.received_db ? Float64Array.from(c.received_db) : sync.receivedEnvelope(oracleReceived(c), c.rate ?? ORACLE.rate, opts))

test('sync: envelopeBytes equals the oracle\'s bytes at every rate, from int16 and from Web Audio floats', () => {
  for (const [rate, b64] of Object.entries(ORACLE.envelope_bytes)) {
    const expected = Buffer.from(b64, 'base64')
    assert.deepEqual(Buffer.from(sync.envelopeBytes(ORACLE_PCM, Number(rate))), expected, `${rate} Hz`)
    const floats = Float32Array.from(ORACLE_PCM, v => v / 32768)
    assert.deepEqual(Buffer.from(sync.envelopeBytes(floats, Number(rate))), expected, `${rate} Hz, floats`)
  }
  assert.equal(sync.envelopeBytes(new Int16Array(24000), 24000).length, 199)   // 1 s: fully-inside windows only
  assert.deepEqual([-200, -96, -80, -48, -16, 0, 12].map(sync.quantise), [0, 0, 43, 128, 213, 255, 255])
})

test('sync: the received envelope without the high-pass equals the oracle\'s envelope_db', () => {
  const c = ORACLE.cases.find(k => k.name === 'shifted')
  const got = sync.receivedEnvelope(oracleReceived(c), ORACLE.rate, { highPassHz: null })
  assert.equal(got.length, c.received_db_no_hp.length)
  const worst = c.received_db_no_hp.reduce((m, v, i) => Math.max(m, Math.abs(v - got[i])), 0)
  assert(worst < 1e-9, `max |dB difference| ${worst}`)
})

test('sync: align equals the oracle on every case (high-pass off): offset within 0.1 ms, checks within 1e-6', () => {
  for (const c of ORACLE.cases) {
    const o = c.oracle_no_hp
    const r = sync.align(oracleReceivedDb(c, { highPassHz: null }), oracleReference(c), c.prior_index, oracleParams(c))
    const k = r.checks
    if (c.gate_only) {
      // A synthetic window that only pins the speech gate: its correlation curve is flat enough
      // that the peak's lag is decided by rounding, so only the gate and the outcome are compared.
      assert.deepEqual([k.speech, k.windowOk, k.accepted], [o.speech, o.window_ok, o.accepted], c.name)
      continue
    }
    if (o.offset_ms === null) assert.equal(r.offsetMs, null, c.name)
    else assert(Math.abs(r.offsetMs - o.offset_ms) <= 0.1, `${c.name}: offset ${r.offsetMs} vs ${o.offset_ms}`)
    assert.deepEqual(
      [k.accepted, k.speech, k.windowOk, k.edge, k.peakOk, k.ratioOk, k.sharpnessOk, k.halvesOk, k.interiorOk],
      [o.accepted, o.speech, o.window_ok, o.edge, o.peak_ok, o.ratio_ok, o.sharpness_ok, o.halves_ok, o.interior_ok],
      c.name,
    )
    for (const [mine, theirs] of [['peak', 'peak'], ['ratio', 'ratio'], ['sharpness', 'sharpness'], ['halfDeviationMs', 'half_deviation_ms'], ['candidateMs', 'candidate_ms']]) {
      const want = oracleNumber(o[theirs])
      if (want === null || want === Infinity) assert.equal(k[mine], want, `${c.name} ${mine}`)
      else assert(Math.abs(k[mine] - want) <= 1e-6, `${c.name} ${mine}: ${k[mine]} vs ${want}`)
    }
  }
  // The cases cover what they claim to: each threshold and branch decides at least one of them.
  const byName = Object.fromEntries(ORACLE.cases.map(c => [c.name, c.oracle_no_hp]))
  const failing = o => ['peak_ok', 'ratio_ok', 'sharpness_ok', 'halves_ok', 'interior_ok'].filter(k => !o[k])
  assert(byName.shifted.accepted && !byName['halves-disagree'].halves_ok && !byName['other-audio'].accepted)
  assert(byName.edge.edge && !byName.silent.speech && !byName.short.window_ok)
  assert(byName['room-noise'].accepted && byName['shifted-48k'].accepted)
  assert.deepEqual(failing(byName['ratio-only']), ['ratio_ok'])
  assert.deepEqual(failing(byName['sharpness-only']), ['sharpness_ok'])
  assert.equal(byName['late-until-1.85'].half_deviation_ms, 'Infinity')   // a half peaked at the edge of its +/- 40 ms
  assert.equal(byName['newer-half-jump'].half_deviation_ms, 'Infinity')
  assert.deepEqual(failing(byName['older-half-only']), ['halves_ok'])
  assert(byName['room-noise-long'].accepted)
  assert(!byName['gate-1.5'].speech && !byName['gate-1.999'].speech && byName['gate-2.0'].speech)
})

test('sync: which half decides: older-half-only by its older half alone, the others by the newer half', () => {
  const halves = name => ORACLE.cases.find(k => k.name === name).oracle_half_deviations_ms.map(oracleNumber)
  const [older, newer] = halves('older-half-only')
  assert(older > 10 && newer <= 10, `older ${older}, newer ${newer}`)
  const c = ORACLE.cases.find(k => k.name === 'older-half-only')
  const k = sync.align(oracleReceivedDb(c, { highPassHz: null }), oracleReference(c), c.prior_index, oracleParams(c)).checks
  assert(!k.halvesOk && k.peakOk && k.ratioOk && k.sharpnessOk && k.interiorOk)
  assert(Math.abs(k.halfDeviationMs - older) <= 1e-6, `${k.halfDeviationMs} vs the older half's ${older}`)
  for (const name of ['late-until-1.85', 'newer-half-jump']) {
    const [o, n] = halves(name)
    assert(n === Infinity && o <= 40, `${name}: older ${o}, newer ${n}`)
  }
})

test('sync: the room-noise cases put the floor above -70 dB, from the 3 s history', () => {
  for (const name of ['room-noise', 'room-noise-long']) {
    const c = ORACLE.cases.find(k => k.name === name)
    const env = oracleReceivedDb(c, { highPassHz: null })
    const floor = sync.align(env, oracleReference(c), c.prior_index).checks.floor
    const history = Array.from(env.subarray(Math.max(0, env.length - 600)))
    assert(floor > -60, `${name}: floor ${floor}`)
    assert.equal(floor, sync.percentile(history, 5) + 6, name)
    // The 5th percentile interpolates (numpy's linear rule), and the last 1 s alone would give another floor.
    assert(!history.slice().sort((a, b) => a - b).includes(sync.percentile(history, 5)), `${name}: the 5th percentile falls between two values`)
    assert.notEqual(floor, sync.percentile(history.slice(-200), 5) + 6, `${name}: 1 s of history would give another floor`)
    if (name === 'room-noise-long') {
      // 3.4 s of track: a 4 s history would be all of it, and gives another floor.
      assert(env.length > 600, 'longer than the 3 s history')
      assert.notEqual(floor, sync.percentile(Array.from(env.subarray(0)), 5) + 6, '4 s of history would give another floor')
    }
  }
})

test('sync: the rule\'s fixed terms and the cadence are the design\'s values', () => {
  assert.deepEqual({ ...sync.DEFAULTS }, {
    hopMs: 5, windowMs: 1000, searchMs: 300, halfSearchMs: 40, sharpAtMs: 20,
    floorHistoryMs: 3000, floorPercentile: 5, floorMarginDb: 6, floorMinDb: -70, gateStdDb: 2,
    minPeak: 0.95, minRatio: 1.2, minSharpness: 0.03, halfTolMs: 10, interpolate: true,
  })
  assert.deepEqual({ ...sync.CADENCE }, { beforeLockMs: 50, afterLockMs: 200, priorLeadMs: 200, widenAfterTicks: 10, wideSearchMs: 600 })
  assert.equal(sync.HIGH_PASS_HZ, 30)
  assert(Object.isFrozen(sync.DEFAULTS) && Object.isFrozen(sync.CADENCE))
})

test('sync: the received hop is the oracle\'s round(0.005 * rate), half to even', () => {
  for (const [rate, hop] of Object.entries(ORACLE.received_hops)) assert.equal(new sync.ReceivedEnvelope(Number(rate)).hop, hop, `${rate} Hz`)
})

test('sync: the high-pass is a first-order y = a (y\' + x - x\'), a = RC / (RC + dt) at the track\'s rate, 30 Hz', () => {
  const rc = 1 / (2 * Math.PI * 30)
  assert.equal(sync.HIGH_PASS_HZ, 30)
  assert.equal(sync.highPassCoefficient(30, 48000), rc / (rc + 1 / 48000))
  for (const rate of [24000, 48000]) assert(Math.abs(sync.highPassCoefficient(30, rate) - 1 / (1 + (2 * Math.PI * 30) / rate)) < 1e-15)
  // A step of 0.5 (Web Audio scale): the filter's output is 0.5 * 32768 * a^(n + 1); every
  // frame's dB must be that of the same samples measured by hand.
  for (const rate of [24000, 48000]) {
    const a = rc / (rc + 1 / rate)
    const hop = new sync.ReceivedEnvelope(rate).hop
    const env = sync.receivedEnvelope(new Float32Array(rate / 10).fill(0.5), rate)
    for (let g = 0; g < env.length; g++) {
      let sum = 0
      for (let n = g * hop; n < g * hop + 2 * hop; n++) sum += (0.5 * 32768 * a ** (n + 1)) ** 2
      const want = 10 * Math.log10(sum / (2 * hop) / (32768 * 32768) + 1e-10)
      assert(Math.abs(env[g] - want) < 1e-9, `${rate} Hz frame ${g}: ${env[g]} vs ${want}`)
    }
  }
})

test('sync: Web Audio floats are measured x 32768 (a level on a byte boundary tells it from 32767)', () => {
  // A constant level 0.0001 dB above the boundary between bytes 199 and 200; x 32767 would put it
  // 0.00027 dB lower, under the boundary.
  const boundaryDb = (199.5 * 96) / 255 - 96
  const level = Math.fround(10 ** ((boundaryDb + 0.0001) / 20))
  const bytes = sync.envelopeBytes(new Float32Array(4800).fill(level), 48000)
  assert(bytes.every(b => b === 200), `${bytes[0]}`)
  assert.equal(sync.quantise(20 * Math.log10((level * 32767) / 32768)), 199)
})

test('sync: with the causal 30 Hz high-pass the offset stays within 0.5 ms of the oracle\'s (zero-phase) one', () => {
  for (const c of ORACLE.cases.filter(k => k.oracle_hp)) {
    const o = c.oracle_hp
    const r = sync.align(oracleReceivedDb(c), oracleReference(c), c.prior_index, oracleParams(c))
    assert.equal(r.checks.accepted, o.accepted, c.name)
    if (o.offset_ms !== null) assert(Math.abs(r.offsetMs - o.offset_ms) <= 0.5, `${c.name}: ${r.offsetMs} vs ${o.offset_ms}`)
  }
})

// ------------------------------------------------------------------ sync.js: reference and splice

const b64 = bytes => Buffer.from(bytes).toString('base64')

test('sync: the reference is reassembled by first_index; holes are counted, values in arrival order do not matter', () => {
  const values = Uint8Array.from({ length: 30 }, (_, i) => i + 1)
  const chunks = [{ first_index: 20, values: b64(values.subarray(20)) }, { first_index: 0, values: b64(values.subarray(0, 20)) }]
  const whole = sync.reassembleEnvelope(chunks)
  assert.deepEqual(Buffer.from(whole.values), Buffer.from(values))
  assert.equal(whole.missing, 0)
  const holed = sync.reassembleEnvelope([chunks[0]])
  assert.equal(holed.values.length, 30)
  assert.equal(holed.missing, 20)
  assert.equal(holed.values[0], 0)
  // An empty message covers nothing: it does not lengthen the reference.
  assert.equal(sync.reassembleEnvelope([...chunks, { first_index: 40, values: '' }]).values.length, 30)
})

test('sync: splice rule: any gap_ms > 0 exactly, never on 0 unchained, never onto an aborted turn, chained at prev_turn_pcm_ms', () => {
  const prev = { id: 'p', endPcmMs: 1500, aborted: false }
  assert.deepEqual(sync.splicePlan({ prevTurnId: '', prevTurnPcmMs: 0, gapMs: 200 }, prev), { kind: 'gap', from: 'p', joinMs: 1700 })
  assert.deepEqual(sync.splicePlan({ prevTurnId: '', prevTurnPcmMs: 0, gapMs: 1000 }, prev), { kind: 'gap', from: 'p', joinMs: 2500 })
  assert.equal(sync.splicePlan({ prevTurnId: '', prevTurnPcmMs: 0, gapMs: 0 }, prev), null)
  assert.equal(sync.splicePlan({ prevTurnId: '', prevTurnPcmMs: 0, gapMs: 200 }, { ...prev, aborted: true }), null)
  assert.equal(sync.splicePlan({ prevTurnId: '', prevTurnPcmMs: 0, gapMs: 200 }, { ...prev, endPcmMs: null }), null)
  assert.deepEqual(sync.splicePlan({ prevTurnId: 'p', prevTurnPcmMs: 1480, gapMs: 0 }, { ...prev, endPcmMs: null }), { kind: 'chained', from: 'p', joinMs: 1480 })
  assert.equal(sync.splicePlan({ prevTurnId: 'p', prevTurnPcmMs: 1480, gapMs: 0 }, { ...prev, aborted: true }), null)
  // A chained turn joins at its own prev_turn_pcm_ms, not at the turn_end of the turn before it.
  assert.deepEqual(sync.splicePlan({ prevTurnId: 'p', prevTurnPcmMs: 1480, gapMs: 0 }, prev), { kind: 'chained', from: 'p', joinMs: 1480 })
  assert.equal(sync.splicePlan({ prevTurnId: '', prevTurnPcmMs: 0, gapMs: 200 }, null), null)
  // previous envelope, then floor up to the join, then this turn's
  const { values, ownOffset } = sync.spliceReference(Uint8Array.of(9, 9, 9), Uint8Array.of(7, 7), 30)
  assert.equal(ownOffset, 6)
  assert.deepEqual([...values], [9, 9, 9, 0, 0, 0, 7, 7])
  // A join off the 5 ms grid rounds to the nearest index: 88 ms is 17.6 hops -> 18, 87 ms is 17.4 -> 17.
  assert.equal(sync.spliceReference(Uint8Array.of(9), Uint8Array.of(7), 88).ownOffset, 18)
  assert.equal(sync.spliceReference(Uint8Array.of(9), Uint8Array.of(7), 87).ownOffset, 17)
})

// ------------------------------------------------------------------ sync.js: the live tracker

const SR = ORACLE.rate
const TURN_X = 'xxxxxxxx-0000-4000-8000-000000000000'
const TURN_Y = 'yyyyyyyy-0000-4000-8000-000000000000'
const ms = v => Math.round((v * SR) / 1000)

/** turn_start + turn_envelope messages of a turn whose audio is `pcm` (the processor's view), via the package's builders' shape. */
function turnMessages(id, pcm, { prev = '', prevPcmMs = 0, gapMs = 0, split = 200, endPcmMs = (pcm.length * 1000) / SR } = {}) {
  const env = sync.envelopeBytes(pcm, SR)
  const start = clone(E.A_start)
  Object.assign(start.turn_start, { turn_id: id, sample_rate: SR, onsets_ms: [0], prev_turn_id: prev, prev_turn_pcm_ms: prevPcmMs, gap_ms: gapMs })
  const chunk = (first, bytes) => ({ tag: 'spatialreal-avatar', turn_envelope: { turn_id: id, first_index: first, hop_ms: 5, window_ms: 10, values: b64(bytes) } })
  // Out of order on purpose: the tail first.
  return { start, envelopes: [chunk(split, env.subarray(split)), chunk(0, env.subarray(0, split))], end: { tag: 'spatialreal-avatar', turn_end: { turn_id: id, pcm_ms: endPcmMs, segment_count: 1 } }, abort: { tag: 'spatialreal-avatar', turn_abort: { turn_id: id } } }
}

/**
 * The accepted ticks' errors against the true first sample, ms. The port's precision is pinned
 * against the oracle above; what the tracker adds is the rule's own error on a live window (a lock
 * after a few ms of speech is a few ms off, design §3.2: p95 <= 5 ms), so the bound here is the
 * design's, not the port's.
 */
function lockErrors(ticks, truth) {
  const errs = ticks.filter(k => k.accepted).map(k => Math.abs(k.first_sample_pos - truth) * 1000 / SR)
  assert(errs.length > 0, 'never locked')
  const sorted = errs.slice().sort((a, b) => a - b)
  assert(sorted[sorted.length - 1] <= 10, `an accepted tick ${sorted[sorted.length - 1]} ms off`)
  assert(sorted[Math.floor(sorted.length / 2)] <= 1, `median error ${sorted[Math.floor(sorted.length / 2)]} ms`)
  return errs
}

/** Where runTracker delivers a message due at `pos`: the start of the first block at or after it. */
const delivered = pos => Math.ceil(pos / 128) * 128

/** Run a tracker over `track` (Int16Array) in 128-sample Web Audio blocks, delivering each scheduled message before the first block at or after its `pos`. */
function runTracker(track, schedule, opts) {
  const tracker = new sync.SyncTracker(SR, opts)
  const due = [...schedule].sort((a, b) => a.pos - b.pos)
  let next = 0
  for (let at = 0; at < track.length; at += 128) {
    while (next < due.length && due[next].pos <= at) {
      const item = due[next++]
      if (item.data) tracker.message(item.data, at)
      if (item.onsetPrior) tracker.setOnsetPrior(item.onsetPrior.id, item.onsetPrior.pos)
    }
    tracker.audio(Float32Array.from(track.subarray(at, at + 128), v => v / 32768))
  }
  return tracker
}

function placeAudio(length, pieces) {
  const track = new Int16Array(length)
  for (const { at, pcm } of pieces) track.set(pcm.subarray(0, Math.max(0, Math.min(pcm.length, length - at))), at)
  return track
}

test('tracker: locks on the turn, 50 ms ticks until the first lock and 200 ms after, the rule\'s error', () => {
  const m = turnMessages(TURN_X, ORACLE_PCM)
  const arrival = ms(1000)
  const first = arrival + ms(153.25)              // the audio path lags the message by 153.25 ms
  const track = placeAudio(ms(4500), [{ at: first, pcm: ORACLE_PCM }])
  const tracker = runTracker(track, [
    { pos: arrival, data: m.start }, ...m.envelopes.map(d => ({ pos: arrival, data: d })),
    { pos: first + ORACLE_PCM.length, data: m.end },
  ])
  const t = tracker.summary()[TURN_X]
  assert.equal(t.referenceLength, 399)
  assert.equal(t.splice, null)
  const lock = t.ticks.findIndex(k => k.accepted)
  assert(lock >= 0, 'never locked')
  lockErrors(t.ticks, first)
  // A tick is align() on the last 3 s of the received envelope at the prior: the same call by hand.
  const k = t.ticks[lock]
  const env = sync.receivedEnvelope(track.subarray(0, k.pos), SR)
  const r = sync.align(env, sync.envelopeBytes(ORACLE_PCM, SR), k.prior_pos / tracker.hop)
  assert(Math.abs((r.offsetMs / 5) * tracker.hop - k.first_sample_pos) < 1e-6, `${r.offsetMs} vs ${k.first_sample_pos}`)
  assert.equal(k.prior_pos, delivered(arrival) + ms(200))
  const gaps = t.ticks.slice(1).map((k, i) => k.pos - t.ticks[i].pos)
  gaps.slice(0, lock).forEach(g => assert(g >= ms(50) && g < ms(50) + 128, `before the lock: ${g}`))
  gaps.slice(lock).forEach(g => assert(g >= ms(200) && g < ms(200) + 128, `after the lock: ${g}`))
  assert(t.ticks.every(k => k.search_ms === 300 && k.prior_pos !== null))
  // After a lock the prior is the last accepted position, not the arrival prior.
  let last = null
  for (const k of t.ticks) {
    if (last !== null) assert.equal(k.prior_pos, last, 'the prior after a lock is the last accepted position')
    else assert.equal(k.prior_pos, delivered(arrival) + ms(200))
    if (k.accepted) last = k.first_sample_pos
  }
  assert(t.ticks.slice(lock + 1).some(k => k.prior_pos !== t.ticks[lock].first_sample_pos), 'the prior moves with each lock')
  // The ticks stop once the 1 s window is past the turn's end.
  assert(t.ticks[t.ticks.length - 1].pos <= first + ORACLE_PCM.length + ms(1000) + ms(200) + 128)
  assert(tracker.ticks === t.ticks.length)
  const exported = tracker.export().turns[0]
  assert.equal(exported.reference, b64(sync.envelopeBytes(ORACLE_PCM, SR)))
  assert.equal(exported.reference_missing, 0)
  assert.equal(exported.start_tap, delivered(arrival))
  assert.equal(exported.ticks.length, t.ticks.length)
})

test('tracker: no lock inside +/- 300 ms of the prior; after 10 held ticks the search widens to +/- 600 ms and locks', () => {
  const m = turnMessages(TURN_X, ORACLE_PCM)
  const arrival = ms(1000)
  const first = arrival + ms(650)                 // 450 ms past the prior (arrival + 200 ms)
  const track = placeAudio(ms(4500), [{ at: first, pcm: ORACLE_PCM }])
  const tracker = runTracker(track, [{ pos: arrival, data: m.start }, ...m.envelopes.map(d => ({ pos: arrival, data: d }))])
  const ticks = tracker.summary()[TURN_X].ticks
  const lock = ticks.findIndex(k => k.accepted)
  assert(lock >= 0, 'never locked')
  const heldBefore = ticks.slice(0, lock).filter(k => k.gate).length
  assert(heldBefore >= 10, `locked after only ${heldBefore} held ticks`)
  const firstWide = ticks.findIndex(k => k.search_ms === 600)
  assert.equal(ticks.slice(0, firstWide).filter(k => k.gate).length, 10)
  assert(ticks.slice(0, firstWide).every(k => k.search_ms === 300 && !k.accepted))
  lockErrors(ticks, first)
  // Re-centred on the onset-detected position when there is one: the same turn with an onset
  // prior locks at once (it is the prior from the start), at the same place.
  const withPrior = runTracker(track, [{ pos: arrival, data: m.start }, ...m.envelopes.map(d => ({ pos: arrival, data: d })), { pos: arrival, onsetPrior: { id: TURN_X, pos: first + ms(30) } }])
  const k2 = withPrior.summary()[TURN_X].ticks
  const lock2 = k2.findIndex(k => k.accepted)
  assert(lock2 >= 0 && lock2 < lock && k2.every(k => k.search_ms === 300))
  lockErrors(k2, first)
  assert.equal(k2[0].prior_pos, first + ms(30))
})

test('tracker: a chained turn is spliced onto the turn before it and locks at prev_turn_pcm_ms after it', () => {
  const half = ms(1000)
  const pcmX = ORACLE_PCM.subarray(0, half)
  const pcmY = ORACLE_PCM.subarray(half)
  const mx = turnMessages(TURN_X, pcmX)
  const my = turnMessages(TURN_Y, pcmY, { prev: TURN_X, prevPcmMs: 1000 })
  const arrival = ms(1000)
  const first = arrival + ms(140)
  const track = placeAudio(ms(4500), [{ at: first, pcm: ORACLE_PCM }])  // X then Y, back to back
  const tracker = runTracker(track, [
    { pos: arrival, data: mx.start }, ...mx.envelopes.map(d => ({ pos: arrival, data: d })),
    { pos: arrival + ms(600), data: my.start }, ...my.envelopes.map(d => ({ pos: arrival + ms(600), data: d })),
    { pos: first + half, data: mx.end },
  ])
  const y = tracker.summary()[TURN_Y]
  assert.deepEqual(plain(y.splice), { kind: 'chained', from: TURN_X, joinMs: 1000, ownOffset: 200 })
  lockErrors(y.ticks, first + half)
  const exported = tracker.export().turns.find(t => t.turn_id === TURN_Y)
  assert.equal(exported.splice.own_offset, 200)
  assert.equal(exported.splice.values, b64(sync.spliceReference(sync.envelopeBytes(pcmX, SR), sync.envelopeBytes(pcmY, SR), 1000).values))
})

/**
 * Turn X (the first `xMs` of the samples), then turn Y (the rest) `gapMs` after X's last sample.
 * A chained Y is announced while X still plays and, as on the page, gets the chain's place as its
 * onset prior (analyze() places a chained turn from the end of the turn before it).
 */
function twoTurns({ xMs, gapMs, y = {}, xEnd = {}, abortX = false, endThenAbortX = false }) {
  const cut = ms(xMs)
  const pcmX = ORACLE_PCM.subarray(0, cut)
  const pcmY = ORACLE_PCM.subarray(cut)
  const mx = turnMessages(TURN_X, pcmX, xEnd)
  const my = turnMessages(TURN_Y, pcmY, y)
  const arrival = ms(1000)
  const first = arrival + ms(140)
  const yFirst = first + cut + ms(gapMs)
  const track = placeAudio(ms(5000), [{ at: first, pcm: pcmX }, { at: yFirst, pcm: pcmY }])
  const yArrival = y.prev ? arrival + ms(600) : yFirst - ms(140)
  const tracker = runTracker(track, [
    { pos: arrival, data: mx.start }, ...mx.envelopes.map(d => ({ pos: arrival, data: d })),
    { pos: first + cut, data: abortX ? mx.abort : mx.end },
    ...(endThenAbortX ? [{ pos: first + cut + 1, data: mx.abort }] : []),
    { pos: yArrival, data: my.start }, ...my.envelopes.map(d => ({ pos: yArrival, data: d })),
    ...(y.prev ? [{ pos: yArrival, onsetPrior: { id: TURN_Y, pos: yFirst } }] : []),
  ])
  return { tracker, yFirst, pcmX, pcmY }
}

test('tracker: an unchained turn with gap_ms > 0 is spliced onto the turn announced before it, gap_ms of floor in between', () => {
  const { tracker, yFirst, pcmX, pcmY } = twoTurns({ xMs: 1000, gapMs: 200, y: { gapMs: 200 } })
  const y = tracker.summary()[TURN_Y]
  assert.deepEqual(plain(y.splice), { kind: 'gap', from: TURN_X, joinMs: 1200, ownOffset: 240 })
  lockErrors(y.ticks, yFirst)
  const exported = tracker.export().turns.find(t => t.turn_id === TURN_Y)
  assert.equal(exported.splice.values, b64(sync.spliceReference(sync.envelopeBytes(pcmX, SR), sync.envelopeBytes(pcmY, SR), 1200).values))
})

test('tracker: never spliced on gap_ms 0 for an unchained turn, nor onto an aborted turn', () => {
  const zero = twoTurns({ xMs: 1000, gapMs: 200, y: { gapMs: 0 } }).tracker.summary()[TURN_Y]
  assert.equal(zero.splice, null)
  assert(zero.ticks.length > 0)
  const aborted = twoTurns({ xMs: 1000, gapMs: 200, y: { gapMs: 200 }, abortX: true }).tracker.summary()[TURN_Y]
  assert.equal(aborted.splice, null)
  assert(aborted.ticks.length > 0)
  // Where the abort flag alone decides: X has its turn_end (pcm_ms known) and then a turn_abort,
  // and a chained Y, whose join does not need X's turn_end at all.
  const endedThenAborted = twoTurns({ xMs: 1000, gapMs: 200, y: { gapMs: 200 }, endThenAbortX: true }).tracker.summary()[TURN_Y]
  assert.equal(endedThenAborted.splice, null)
  const chainedAfterAbort = twoTurns({ xMs: 1000, gapMs: 0, y: { prev: TURN_X, prevPcmMs: 1000 }, abortX: true }).tracker.summary()[TURN_Y]
  assert.equal(chainedAfterAbort.splice, null)
  assert(chainedAfterAbort.ticks.length > 0)
  // ... and the same two without the abort are spliced.
  assert.equal(twoTurns({ xMs: 1000, gapMs: 200, y: { gapMs: 200 } }).tracker.summary()[TURN_Y].splice.kind, 'gap')
  assert.equal(twoTurns({ xMs: 1000, gapMs: 0, y: { prev: TURN_X, prevPcmMs: 1000 } }).tracker.summary()[TURN_Y].splice.kind, 'chained')
})

test('tracker: a chained turn joins at its prev_turn_pcm_ms (1498 ms, off the grid: index 300), not at the turn_end before it (1500 ms)', () => {
  const { tracker, yFirst } = twoTurns({ xMs: 1498, gapMs: 0, y: { prev: TURN_X, prevPcmMs: 1498 }, xEnd: { endPcmMs: 1500 } })
  const y = tracker.summary()[TURN_Y]
  assert.deepEqual(plain(y.splice), { kind: 'chained', from: TURN_X, joinMs: 1498, ownOffset: 300 })
  // The join rounds to the 5 ms grid, so the spliced reference holds Y 2 ms later than it plays:
  // the lock, driven mostly by X's audio in the window, puts Y's first sample there.
  lockErrors(y.ticks, yFirst + ms(2))
})

test('tracker: an onset prior far from the arrival prior is the prior from the first tick, and stays the centre once the search widens', () => {
  // The onset rule's place is 450 ms past the truth (and 650 ms past the arrival prior): the normal
  // search (+/- 300 ms around it) misses the audio, the widened one (+/- 600 ms around it) finds it.
  const m = turnMessages(TURN_X, ORACLE_PCM)
  const arrival = ms(1000)
  const first = arrival + ms(400)
  const onsetPrior = first + ms(450)                  // outside +/- 300 of the truth, inside +/- 600
  const track = placeAudio(ms(4500), [{ at: first, pcm: ORACLE_PCM }])
  const ticks = runTracker(track, [
    { pos: arrival, data: m.start }, ...m.envelopes.map(d => ({ pos: arrival, data: d })),
    { pos: arrival, onsetPrior: { id: TURN_X, pos: onsetPrior } },
  ]).summary()[TURN_X].ticks
  const lock = ticks.findIndex(k => k.accepted)
  assert(lock >= 0, 'never locked')
  ticks.slice(0, lock + 1).forEach(k => assert.equal(k.prior_pos, onsetPrior))
  assert.equal(ticks[lock].search_ms, 600)
  assert.equal(ticks.slice(0, lock).filter(k => k.gate).length, 10)
  lockErrors(ticks, first)
})

test('the table: the header has the alignment columns and every row has one cell per header', () => {
  const head = html.match(/<thead>([\s\S]*?)<\/thead>/)[1]
  const titles = [...head.matchAll(/<th[^>]*>([\s\S]*?)<\/th>/g)].map(m => m[1].replace(/&Delta;/g, 'Δ').trim())
  for (const name of ['first lock ms', 'offset ms', 'p95 err ms', 'held / acc', 'peak / ratio / sharp / halves']) assert(titles.includes(name), `no "${name}" column: ${titles}`)
  const s = goodSession()
  const o = 480000 - samples(180)                  // A's onset prior: its first onset heard, minus onsets_ms[0]
  s.log.find(r => r.data.turn_start && r.data.turn_start.turn_id === A).tap = o - samples(150)
  const tick = (pos, accepted, firstSample, checks) => ({ pos, gate: true, accepted, first_sample_pos: accepted ? firstSample : null, checks })
  const good = { speech: true, peak: 0.991, ratio: 3.04, sharpness: 0.224, half_deviation_ms: 0.52, edge: false, peak_ok: true, ratio_ok: true, sharpness_ok: true, halves_ok: true, interior_ok: true }
  const bad = { ...good, peak: 0.9, peak_ok: false, ratio: 'Infinity' }
  // A locks on its onset prior, then 1 ms either side; its speech starts onsets_ms[0] = 180 ms after it.
  const speech = o + samples(180)
  s.sync = {
    [A]: { referenceLength: 600, splice: null, ticks: [
      { pos: speech - 2400, gate: false, accepted: false, first_sample_pos: null, checks: { speech: false } },
      tick(speech, false, null, bad),
      tick(speech + samples(120), true, o, good),
      tick(speech + samples(320), true, o + samples(1), good),
      tick(speech + samples(520), true, o - samples(1), good),
    ] },
    [B]: { referenceLength: 300, splice: { kind: 'chained' }, ticks: [tick(200000, false, null, bad)] },
  }
  const r = analyze(s)
  const [a, b, c] = r.turns
  assert.equal(a.onsetPriorPos, o)
  assert.equal(a.sync.held, 1); assert.equal(a.sync.accepted, 3)
  assert.equal(a.sync.firstLockMs, 120)
  assert.equal(a.sync.offsetMs, 149)              // the last accepted tick: 1 ms before the prior, which is 150 ms after turn_start
  assert.equal(a.sync.p95ErrMs, 1)                // |errors| 0, 1, 1 ms
  const cells = api.rowCells(a).map(x => x.text)
  assert.equal(cells.length, titles.length)
  const col = name => cells[titles.indexOf(name)]
  assert.equal(col('first lock ms'), '120')
  assert.equal(col('offset ms'), '+149.0')
  assert.equal(col('p95 err ms'), '1.0')
  assert.equal(col('held / acc'), '1 / 3')
  assert.equal(col('peak / ratio / sharp / halves'), '0.991 / 3.04 / 0.224 / 0.5')
  const bcells = api.rowCells(b).map(x => x.text)
  assert.equal(bcells[titles.indexOf('held / acc')], '1 / 0 *')
  assert.equal(bcells[titles.indexOf('first lock ms')], 'none')
  assert.equal(bcells[titles.indexOf('peak / ratio / sharp / halves')], '0.900! / inf / 0.224 / 0.5')
  const ccells = api.rowCells(c).map(x => x.text)
  assert.equal(ccells.length, titles.length)
  assert.equal(ccells[titles.indexOf('offset ms')], '-')
})

console.log(n + ' passed')
