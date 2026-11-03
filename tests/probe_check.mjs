// Runs the analysis code of examples/probe/index.html (its <script id="analysis"> block) under
// Node. tests/test_probe_page.py runs it: usage `node probe_check.mjs <page> <fixture.json>`, where
// the fixture holds envelope messages built by pipecat_spatialreal.envelope and random audio with
// the onsets the package's OnsetTracker finds in it.
import fs from 'node:fs'
import vm from 'node:vm'
import assert from 'node:assert/strict'

const html = fs.readFileSync(process.argv[2], 'utf8')
const match = html.match(/<script id="analysis">([\s\S]*?)<\/script>/)
assert(match, 'the analysis script is not in the page')
const sandbox = { atob: globalThis.atob }
vm.createContext(sandbox)
const api = vm.runInContext(match[1] + '\n;({ analyze, summarize, OnsetDetector, toPcm16, reassemble, decodeBase64 })', sandbox)
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
console.log(n + ' passed')
