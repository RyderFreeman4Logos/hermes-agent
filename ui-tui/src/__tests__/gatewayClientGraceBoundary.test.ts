import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ChildProcess } from 'node:child_process'

// A startup that never emits gateway.ready must reap the child it spawned.
// HERMES_TUI_STARTUP_TIMEOUT_MS is read at import and floored at 5000ms, so
// the fixture owns that value before the client module loads.

const TIMEOUT_ENV = 'HERMES_TUI_STARTUP_TIMEOUT_MS'
const GRACE_ENV = 'HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S'
const previousTimeout = process.env[TIMEOUT_ENV]
const previousGrace = process.env[GRACE_ENV]

delete process.env[TIMEOUT_ENV]
process.env[TIMEOUT_ENV] = '5000'
delete process.env[GRACE_ENV]
vi.resetModules()

const children: ChildProcess[] = []

vi.mock('node:child_process', async () => {
  const actual = await vi.importActual<typeof import('node:child_process')>('node:child_process')

  return {
    ...actual,
    spawn: (...args: Parameters<typeof actual.spawn>) => {
      const child = actual.spawn(...args)

      children.push(child)

      return child
    }
  }
})

const realSetTimeout = globalThis.setTimeout
const killDelays: number[] = []

vi.spyOn(globalThis, 'setTimeout').mockImplementation((fn: TimerHandler, ms?: number, ...rest: unknown[]) => {
  if (typeof ms === 'number') killDelays.push(ms)
  return realSetTimeout(fn as (...args: unknown[]) => void, ms as number, ...(rest as []))
})

const { GatewayClient } = await import('../gatewayClient.js')

const clients: InstanceType<typeof GatewayClient>[] = []

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0)

    return true
  } catch {
    return false
  }
}

function startTime(pid: number): string | null {
  try {
    return readFileSync(`/proc/${pid}/stat`, 'utf8').split(')').slice(1).join(')').trim().split(/\s+/)[19] ?? null
  } catch {
    return null
  }
}

function sameProcess(pid: number, started: string | null): boolean {
  return started !== null && startTime(pid) === started && alive(pid)
}

async function waitFor(predicate: () => boolean, ms: number) {
  const deadline = Date.now() + ms

  while (Date.now() < deadline && !predicate()) {
    await new Promise(resolve => setTimeout(resolve, 25))
  }
}

function destroyPipes(child: ChildProcess) {
  child.stdin?.destroy()
  child.stdout?.destroy()
  child.stderr?.destroy()
}

async function closeFixture(child: ChildProcess) {
  const pid = child.pid ?? 0
  const started = pid > 0 ? startTime(pid) : null

  destroyPipes(child)

  if (sameProcess(pid, started)) {
    try {
      child.kill('SIGKILL')
    } catch {
      // already gone
    }
  }

  await Promise.race([
    new Promise(resolve => child.once('close', resolve)),
    new Promise(resolve => setTimeout(resolve, 2000))
  ])
  expect(child.exitCode !== null || child.signalCode !== null).toBe(true)
  expect(sameProcess(pid, started)).toBe(false)
}

afterEach(async () => {
  if (previousTimeout === undefined) {
    delete process.env[TIMEOUT_ENV]
  } else {
    process.env[TIMEOUT_ENV] = previousTimeout
  }

  if (previousGrace === undefined) {
    delete process.env[GRACE_ENV]
  } else {
    process.env[GRACE_ENV] = previousGrace
  }

  for (const client of clients.splice(0)) {
    client.kill('test-cleanup')
  }

  for (const child of children.splice(0)) {
    await closeFixture(child)
  }
})

function hangPython(root: string) {
  const python = join(root, 'hang-python')

  writeFileSync(
    python,
    `#!/usr/bin/env node
process.on('SIGTERM', () => {})
process.stdout.write(JSON.stringify({jsonrpc:'2.0',method:'event',params:{type:'gateway.ready',payload:{}}}) + '\\n')
setTimeout(() => {}, 30000)
`
  )
  chmodSync(python, 0o755)

  return python
}

function isolateSpawnEnv(python: string, root: string) {
  const previous = {
    python: process.env.HERMES_PYTHON,
    root: process.env.HERMES_PYTHON_SRC_ROOT,
    url: process.env.HERMES_TUI_GATEWAY_URL,
    sidecar: process.env.HERMES_TUI_SIDECAR_URL,
    grace: process.env[GRACE_ENV]
  }

  delete process.env.HERMES_TUI_GATEWAY_URL
  delete process.env.HERMES_TUI_SIDECAR_URL
  process.env.HERMES_PYTHON = python
  process.env.HERMES_PYTHON_SRC_ROOT = root

  return () => {
    if (previous.python === undefined) delete process.env.HERMES_PYTHON
    else process.env.HERMES_PYTHON = previous.python
    if (previous.root === undefined) delete process.env.HERMES_PYTHON_SRC_ROOT
    else process.env.HERMES_PYTHON_SRC_ROOT = previous.root
    if (previous.url === undefined) delete process.env.HERMES_TUI_GATEWAY_URL
    else process.env.HERMES_TUI_GATEWAY_URL = previous.url
    if (previous.sidecar === undefined) delete process.env.HERMES_TUI_SIDECAR_URL
    else process.env.HERMES_TUI_SIDECAR_URL = previous.sidecar
    if (previous.grace === undefined) delete process.env[GRACE_ENV]
    else process.env[GRACE_ENV] = previous.grace
  }
}

const MAX_TIMER_MS = 2_147_483_647
const SLACK_MS = 50

function script() {
  return `#!/usr/bin/env node
process.on('SIGTERM', () => {})
process.stdout.write(JSON.stringify({jsonrpc:'2.0',method:'event',params:{type:'gateway.ready',payload:{}}}) + '\\n')
setTimeout(() => {}, 30000)
`
}

function retireDelay(raw: string): Promise<number> {
  const root = mkdtempSync(join(tmpdir(), 'hermes-grace-boundary-'))
  const python = join(root, 'gateway')

  writeFileSync(python, script())
  chmodSync(python, 0o755)

  const restore = isolateSpawnEnv(python, root)

  process.env[GRACE_ENV] = raw

  const events: string[] = []
  const gw = new GatewayClient()

  clients.push(gw)
  gw.on('event', (ev: { type: string }) => events.push(ev.type))
  gw.drain()

  const before = killDelays.length

  return (async () => {
    try {
      await new Promise(resolve => setTimeout(resolve, 1))
      gw.start()
      await waitFor(() => events.includes('gateway.ready'), 4000)
      expect(events).toContain('gateway.ready')
      gw.kill('grace-boundary')
      await waitFor(() => killDelays.slice(before).some(ms => ms !== 5000), 1000)
      const delays = killDelays.slice(before).filter(ms => ms !== 5000)
      expect(delays.length).toBeGreaterThan(0)
      return delays.at(-1) as number
    } finally {
      restore()
    }
  })()
}

describe('owned shutdown grace final delay', () => {
  it('keeps a finite backend grace plus slack inside the Node timer ceiling', async () => {
    expect(await retireDelay(String((MAX_TIMER_MS - SLACK_MS) / 1000))).toBe(MAX_TIMER_MS)
    expect(await retireDelay(String(MAX_TIMER_MS / 1000))).toBe(MAX_TIMER_MS)
    expect(await retireDelay('2147483.647')).toBe(MAX_TIMER_MS)
  }, 10_000)

  it('does not turn a grace above the ceiling into a 1ms kill', async () => {
    expect(await retireDelay('2147483.648')).toBe(MAX_TIMER_MS)
    expect(await retireDelay('Infinity')).toBe(MAX_TIMER_MS)
  }, 10_000)

  it('uses the Python float grammar the backend uses', async () => {
    expect(await retireDelay('2_0e-1')).toBe(2000 + SLACK_MS)
    expect(await retireDelay('0x2')).toBe(1000 + SLACK_MS)
    expect(await retireDelay('')).toBe(1000 + SLACK_MS)
    expect(await retireDelay('nope')).toBe(1000 + SLACK_MS)
    expect(await retireDelay('nan')).toBe(1000 + SLACK_MS)
    expect(await retireDelay('-1')).toBe(1000 + SLACK_MS)
    expect(await retireDelay('-0')).toBe(1000 + SLACK_MS)
  }, 20_000)
})
