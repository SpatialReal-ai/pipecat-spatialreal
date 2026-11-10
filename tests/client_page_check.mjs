// Runs the module script of examples/client/index.html under Node and drives it the way a browser
// would: press Start, let the bot's callbacks fire, press Stop. Every module the page imports is a
// recording fake, and so are fetch, the timers and the few elements the page touches, so nothing
// leaves the machine. tests/test_client_page.py runs it: `node client_page_check.mjs <page> <dir>`,
// where <dir> is an empty directory for the rewritten page and its fakes.
import fs from 'node:fs'
import path from 'node:path'
import assert from 'node:assert/strict'
import { pathToFileURL } from 'node:url'

const [PAGE, DIR] = process.argv.slice(2)
const html = fs.readFileSync(PAGE, 'utf8')
const script = html.match(/<script type="module">([\s\S]*?)<\/script>/)
assert(script, 'the page has no module script')

// --- the page, with each import pointed at a fake

const FAKES = {
  'https://esm.sh/@pipecat-ai/client-js@': 'client-js.mjs',
  'https://esm.sh/@pipecat-ai/daily-transport@': 'daily-transport.mjs',
  'https://esm.sh/@pipecat-ai/livekit-transport@': 'livekit-transport.mjs',
  'https://cdn.jsdelivr.net/npm/@spatialreal/web-sdk@': 'spatialreal-web-sdk.mjs',
}
const source = script[1].replace(/(\bfrom\s*|\bimport\s*\()'([^']+)'/g, (all, head, specifier) => {
  const fake = Object.entries(FAKES).find(([prefix]) => specifier.startsWith(prefix))
  assert(fake, `the page imports ${specifier}, which has no fake here`)
  return `${head}'./fakes/${fake[1]}'`
})
fs.mkdirSync(path.join(DIR, 'fakes'), { recursive: true })
fs.writeFileSync(path.join(DIR, 'package.json'), '{"type": "module"}\n')
fs.writeFileSync(path.join(DIR, 'page.mjs'), source)
// The fakes record into globalThis.world, the scenario that is running.
fs.writeFileSync(path.join(DIR, 'fakes', 'client-js.mjs'), `
export class PipecatClient {
  constructor(options) { this.options = options; this.disconnects = 0; globalThis.world.clients.push(this) }
  connect(params) { this.params = params; return globalThis.world.connect(this) }
  async disconnect() {
    this.disconnects++
    await null
    this.options.callbacks.onDisconnected?.() // as pipecat-client-js does once the transport has left
  }
}
`)
fs.writeFileSync(path.join(DIR, 'fakes', 'daily-transport.mjs'), `export class DailyTransport { constructor() { this.kind = 'daily' } }\n`)
fs.writeFileSync(path.join(DIR, 'fakes', 'livekit-transport.mjs'), `export class LiveKitTransport { constructor() { this.kind = 'livekit' } }\n`)
fs.writeFileSync(path.join(DIR, 'fakes', 'spatialreal-web-sdk.mjs'), `
export function createUnlockedAudioContext() { return globalThis.world.audioContext() }
export class SpatialReal {
  constructor(config) { globalThis.world.configs.push(config) }
  createSession(options) { return globalThis.world.createSession(options) }
}
`)

// --- a world for one scenario: the browser around the page, and the bot's server

const [realSetTimeout, realClearTimeout] = [globalThis.setTimeout, globalThis.clearTimeout]
const idleTurns = async () => { for (let i = 0; i < 50; i++) await new Promise(resolve => setImmediate(resolve)) }
const abortError = () => Object.assign(new Error('The operation was aborted.'), { name: 'AbortError' })

const SESSION_ID = 'abcdef12-3456-4789-8abc-def012345678'
const TOKEN_REPLY = { session_token: 'tok-1', avatar_id: 'avatar-1', app_id: 'app-1', endpoints: {} }
const START_REPLY = {
  daily: { dailyRoom: 'https://example.daily.co/room', dailyToken: 'room-token', sessionId: SESSION_ID },
  livekit: { url: 'wss://livekit.example', token: 'user-token', sessionId: SESSION_ID },
}

/**
 * routes: path -> { status, body } | 'hang' (never answers; rejects when its signal aborts).
 * connect: 'ready' (the bot gets ready at once) or 'wait' (until world.ready() is called).
 */
function makeWorld({ search = '', routes = {}, connect = 'wait' } = {}) {
  const transport = new URLSearchParams(search).get('transport') === 'livekit' ? 'livekit' : 'daily'
  const answers = { '/spatialreal-token': { status: 200, body: TOKEN_REPLY }, '/start': { status: 200, body: START_REPLY[transport] }, ...routes }
  const elements = new Map()
  for (const [, id, attributes] of html.matchAll(/<(?:button|div|span|pre)\s+id="(\w+)"([^>]*)>/g)) {
    elements.set(id, { id, textContent: '', disabled: /\bdisabled\b/.test(attributes), onclick: null, scrollTop: 0, scrollHeight: 0 })
  }
  let readyNow = null
  const world = {
    search, fetches: [], clients: [], configs: [], sessionOptions: [], sessions: [], contexts: [], timers: [],
    el: id => { assert(elements.has(id), `the page has no #${id}`); return elements.get(id) },
    status: () => elements.get('status').textContent,
    log: () => elements.get('log').textContent,
    ready: () => { assert(readyNow, 'nobody waits for the bot'); readyNow() },
    /** Let every timer the page set run out, as if the time had passed. */
    expire: () => { for (const timer of world.timers.filter(t => !t.done)) { timer.done = true; timer.fn() } },
    audioContext() {
      const context = { closes: 0, async close() { this.closes++ } }
      world.contexts.push(context)
      return context
    },
    createSession(options) {
      world.sessionOptions.push(options)
      const session = {
        handlers: {}, starts: 0, disposals: 0, tracks: [], messages: [],
        on(event, handler) { (this.handlers[event] ??= []).push(handler) },
        async start() { this.starts++ },
        setBotAudioTrack(track) { this.tracks.push(track) },
        handleServerMessage(data) { this.messages.push(data); return true },
        async dispose() { this.disposals++ },
      }
      world.sessions.push(session)
      return Promise.resolve(session)
    },
    connect(client) {
      if (connect === 'ready') return Promise.resolve()
      return new Promise(resolve => { readyNow = resolve })
    },
  }
  const fetch = (url, init = {}) => {
    const call = { url: String(url), method: init.method ?? 'GET', headers: init.headers ?? {}, body: init.body, signal: init.signal }
    world.fetches.push(call)
    const answer = answers[call.url]
    assert(answer, `the page asked for ${call.url}, which nothing serves here`)
    return new Promise((resolve, reject) => {
      if (call.signal?.aborted) return reject(abortError())
      call.signal?.addEventListener('abort', () => reject(abortError()), { once: true })
      if (answer === 'hang') return
      resolve({ ok: answer.status < 400, status: answer.status, json: async () => answer.body })
    })
  }
  return { world, globals: {
    world,
    document: { getElementById: id => elements.get(id) ?? null },
    location: { search },
    fetch,
    setTimeout: (fn, ms) => { const timer = { fn, ms, done: false }; world.timers.push(timer); return timer },
    clearTimeout: timer => { if (timer) timer.done = true },
  } }
}

let loads = 0
/** A fresh copy of the page in a fresh world: the module is evaluated again for every scenario. */
async function open(options) {
  const { world, globals } = makeWorld(options)
  Object.assign(globalThis, globals)
  await import(`${pathToFileURL(path.join(DIR, 'page.mjs')).href}?load=${++loads}`)
  return world
}

/** Press a button: run its click handler, without waiting for what it starts. */
function press(world, id) {
  const button = world.el(id)
  assert.equal(button.disabled, false, `#${id} is disabled`)
  return button.onclick()
}

let passed = 0
async function test(name, fn) {
  let timer
  try {
    const late = new Promise((_, reject) => { timer = realSetTimeout(() => reject(new Error('timed out: the page never got there')), 5000) })
    await Promise.race([fn(), late])
    console.log('ok   ', name)
    passed++
  } catch (e) {
    console.log('FAIL ', name, '\n', e.stack ?? e.message)
    process.exitCode = 1
  } finally {
    realClearTimeout(timer)
  }
}

const botAudio = { kind: 'audio', id: 'bot-audio' }
const DAILY_BOT = { id: 'b0', name: 'Pipecat Bot', local: false }
const LIVEKIT_BOT = { id: 'Pipecat Agent-abcdef12', name: 'Pipecat Agent-abcdef12', local: false }

/** Start, wait until the page waits for the bot, and return the world with its one client. */
async function startedUntilConnect(options) {
  const world = await open(options)
  const run = press(world, 'start')
  await idleTurns()
  assert.equal(world.clients.length, 1, `no client yet; the page says: ${world.status()}`)
  return { world, run, client: world.clients[0], session: world.sessions[0] }
}

// --- the scenarios

await test('Daily: Start gets a token, asks /start for a new Daily room and joins it', async () => {
  const { world, run, client, session } = await startedUntilConnect({})
  assert.deepEqual(world.fetches.map(f => [f.method, f.url]), [['GET', '/spatialreal-token'], ['POST', '/start']])
  const start = world.fetches[1]
  assert.deepEqual(JSON.parse(start.body), { transport: 'daily', createDailyRoom: true }) // R18
  assert.equal(start.headers['Content-Type'], 'application/json')
  assert.equal(client.options.transport.kind, 'daily')
  assert.equal(client.options.enableMic, true)
  assert.deepEqual(client.params, { url: 'https://example.daily.co/room', token: 'room-token' })
  world.ready()
  await run
  assert.equal(world.el('stop').disabled, false)
  assert.equal(world.el('start').disabled, true)
  assert.equal(session.disposals, 0)
})

await test('the avatar session: the token, the avatar, the container, the context, and its three events', async () => {
  const { world, run, session } = await startedUntilConnect({})
  assert.deepEqual(world.configs, [{ appId: 'app-1', endpoints: {} }])
  const [options] = world.sessionOptions
  assert.equal(options.avatarId, 'avatar-1')
  assert.equal(options.credential, 'tok-1')
  assert.equal(options.container, world.el('avatar'))
  assert.deepEqual(options.pipecat, {})
  assert.equal(options.audioContext, world.contexts[0])
  assert.deepEqual(Object.keys(session.handlers).sort(), ['degraded', 'error', 'turn'])
  for (const handlers of Object.values(session.handlers)) assert.equal(handlers.length, 1)
  assert.equal(session.starts, 1)
  // the page shows what the session reports
  session.handlers.turn[0]({ phase: 'begin', turnId: 'aaaaaaaa-1111' })
  assert.match(world.status(), /turn begin aaaaaaaa/)
  session.handlers.degraded[0]({ reason: 'no-envelope', turnId: 'bbbbbbbb-2222' })
  assert.match(world.status(), /degraded.*no-envelope.*bbbbbbbb/)
  session.handlers.error[0]({ error: new Error('boom') })
  assert.match(world.status(), /avatar error: boom/)
  world.ready()
  await run
})

await test('the deployment the token route names goes to new SpatialReal()', async () => {
  const endpoints = { api: 'https://api.deployment.example' }
  const { world, run } = await startedUntilConnect({ routes: { '/spatialreal-token': { status: 200, body: { ...TOKEN_REPLY, endpoints } } } })
  assert.deepEqual(world.configs, [{ appId: 'app-1', endpoints }])
  world.ready()
  await run
})

await test('Daily: only the bot\'s audio reaches the SDK; every server message does', async () => {
  const { world, run, client, session } = await startedUntilConnect({})
  const { onTrackStarted, onServerMessage } = client.options.callbacks
  onTrackStarted({ kind: 'audio', id: 'other' }, { id: 'u2', name: 'Someone else', local: false })
  onTrackStarted({ kind: 'audio', id: 'mic' }, { id: 'me', name: '', local: true })
  onTrackStarted({ kind: 'video', id: 'bot-video' }, DAILY_BOT)
  onTrackStarted({ kind: 'audio', id: 'viewer' }, { id: 'p', name: 'Another viewer', local: false })
  assert.deepEqual(session.tracks, [])
  onTrackStarted(botAudio, DAILY_BOT)
  assert.deepEqual(session.tracks, [botAudio])
  const message = { tag: 'spatialreal-avatar', turn_start: { turn_id: 't' } }
  onServerMessage(message)
  onServerMessage({ some: 'other server message' })
  assert.equal(session.messages[0], message)
  assert.equal(session.messages.length, 2)
  world.ready()
  await run
})

await test('LiveKit: /start asks for LiveKit, and the bot is the agent of this session', async () => {
  const { world, run, client, session } = await startedUntilConnect({ search: '?transport=livekit' })
  assert.deepEqual(JSON.parse(world.fetches[1].body), { transport: 'livekit' })
  assert.equal(client.options.transport.kind, 'livekit')
  assert.deepEqual(client.params, { url: 'wss://livekit.example', token: 'user-token' })
  const { onTrackStarted } = client.options.callbacks
  onTrackStarted({ kind: 'audio', id: 'other bot' }, { id: 'Pipecat Agent-99999999', name: 'Pipecat Agent-99999999', local: false })
  onTrackStarted({ kind: 'audio', id: 'user' }, { id: 'User-abcdef12', name: 'User-abcdef12', local: false })
  onTrackStarted({ kind: 'audio', id: 'named' }, { id: 'x', name: 'Pipecat Bot', local: false })
  assert.deepEqual(session.tracks, [])
  onTrackStarted(botAudio, LIVEKIT_BOT)
  assert.deepEqual(session.tracks, [botAudio])
  world.ready()
  await run
})

await test('Stop works while the bot starts, and lets go of everything the run made', async () => {
  const { world, run, client, session } = await startedUntilConnect({})
  assert.equal(world.el('stop').disabled, false, 'Stop must be usable while the page waits for the bot')
  await press(world, 'stop')
  await run
  assert.equal(client.disconnects, 1)
  assert.equal(session.disposals, 1)
  assert.equal(world.contexts[0].closes, 1)
  assert.equal(world.el('start').disabled, false)
  assert.equal(world.el('stop').disabled, true)
  assert.equal(world.status(), 'stopped')
  assert.doesNotMatch(world.log(), /failed|disconnected/)
})

await test('Stop while the token is still on its way: the request is cancelled and nothing is left open', async () => {
  const world = await open({ routes: { '/spatialreal-token': 'hang' } })
  const run = press(world, 'start')
  await idleTurns()
  assert.equal(world.el('stop').disabled, false)
  await press(world, 'stop')
  await run
  assert.equal(world.fetches[0].signal.aborted, true)
  assert.equal(world.sessions.length, 0)
  assert.equal(world.contexts[0].closes, 1)
  assert.equal(world.el('start').disabled, false)
  assert.equal(world.status(), 'stopped')
})

await test('a bot that never gets ready: Start gives up after a bounded wait and says so', async () => {
  const { world, run, client, session } = await startedUntilConnect({})
  const waits = world.timers.filter(t => !t.done)
  assert.equal(waits.length, 1, 'one bounded wait for the bot')
  assert(waits[0].ms > 0 && waits[0].ms <= 120000, `the wait for the bot is ${waits[0].ms} ms`)
  world.expire()
  await run
  assert.match(world.status(), /^failed: .*not ready.*terminal/)
  assert.equal(client.disconnects, 1)
  assert.equal(session.disposals, 1)
  assert.equal(world.contexts[0].closes, 1)
  assert.equal(world.el('start').disabled, false)
  assert.equal(world.el('stop').disabled, true)
})

await test('/start that never answers: Start gives up after a bounded wait and says so', async () => {
  const world = await open({ routes: { '/start': 'hang' } })
  const run = press(world, 'start')
  await idleTurns()
  const waits = world.timers.filter(t => !t.done)
  assert.equal(waits.length, 1)
  assert(waits[0].ms > 0 && waits[0].ms <= 120000, `the wait for /start is ${waits[0].ms} ms`)
  world.expire()
  await run
  assert.match(world.status(), /^failed: \/start did not answer/)
  assert.equal(world.fetches[1].signal.aborted, true)
  assert.equal(world.sessions[0].disposals, 1)
  assert.equal(world.contexts[0].closes, 1)
  assert.equal(world.el('start').disabled, false)
})

await test('an error from a route is shown with its detail', async () => {
  for (const [route, status] of [['/spatialreal-token', 403], ['/start', 500]]) {
    const world = await open({ routes: { [route]: { status, body: { detail: `no (${status})` } } } })
    await press(world, 'start')
    assert.equal(world.status(), `failed: ${route} answered ${status}: no (${status})`)
    assert.equal(world.el('start').disabled, false)
    assert.equal(world.el('stop').disabled, true)
    assert.equal(world.contexts[0].closes, 1)
  }
})

await test('/start with no room in its reply fails at once instead of waiting', async () => {
  const world = await open({ routes: { '/start': { status: 200, body: null } } })
  await press(world, 'start')
  assert.match(world.status(), /^failed: \/start returned no room/)
  assert.equal(world.clients.length, 0)
})

await test('the bot leaving ends the run; a Stop does not report it as a disconnect', async () => {
  const { world, run, client, session } = await startedUntilConnect({ connect: 'ready' })
  await run
  client.options.callbacks.onDisconnected()
  await idleTurns()
  assert.match(world.log(), /disconnected/)
  assert.equal(client.disconnects, 1)
  assert.equal(session.disposals, 1)
  assert.equal(world.el('start').disabled, false)
  assert.equal(world.el('stop').disabled, true)
})

await test('Start after Stop makes a new run with its own context, session and client', async () => {
  const { world, run } = await startedUntilConnect({ connect: 'ready' })
  await run
  await press(world, 'stop')
  const again = press(world, 'start')
  await idleTurns()
  await again
  assert.equal(world.contexts.length, 2)
  assert.equal(world.sessions.length, 2)
  assert.equal(world.clients.length, 2)
  assert.equal(world.contexts[0].closes, 1)
  assert.equal(world.contexts[1].closes, 0)
  assert.equal(world.el('stop').disabled, false)
})

assert(passed > 0, 'no scenario ran')
console.log(`${passed} passed`)
