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

describe('GatewayClient startup timeout cleanup', () => {
  it('reaps a spawned gateway that never becomes ready', async () => {
    expect(process.env[TIMEOUT_ENV]).toBe('5000')

    const root = mkdtempSync(join(tmpdir(), 'hermes-startup-timeout-'))
    const restore = isolateSpawnEnv(hangPython(root), root)
    const events: string[] = []
    const gw = new GatewayClient()

    clients.push(gw)
    gw.on('event', (ev: { type: string }) => events.push(ev.type))
    gw.drain()
    await new Promise(resolve => setTimeout(resolve, 20))

    try {
      gw.start()
      await waitFor(() => events.includes('gateway.start_timeout'), 8000)

      const child = children.at(-1)
      const pid = child?.pid ?? 0
      const started = startTime(pid)

      await new Promise(resolve => setTimeout(resolve, 2000))
      expect(events).toContain('gateway.start_timeout')
      expect(events).not.toContain('gateway.reconnecting')
      expect(children).toEqual([child])
      expect(pid).toBeGreaterThan(0)
      expect(sameProcess(pid, started)).toBe(false)
      expect(child?.exitCode !== null || child?.signalCode !== null).toBe(true)
    } finally {
      restore()
    }
  }, 15_000)

  it('does not publish a late ready from a TERM-resistant child, and that child is dead', async () => {
    const root = mkdtempSync(join(tmpdir(), 'hermes-startup-resistant-'))
    const python = join(root, 'gateway')

    writeFileSync(
      python,
      `#!/usr/bin/env node
process.on('SIGTERM', () => {
  process.stdout.write(JSON.stringify({jsonrpc:'2.0',method:'event',params:{type:'gateway.ready',payload:{}}}) + '\\n')
})
setTimeout(() => process.exit(0), 25000)
`
    )
    chmodSync(python, 0o755)

    const restore = isolateSpawnEnv(python, root)
    const events: string[] = []
    let exits = 0
    const gw = new GatewayClient()

    clients.push(gw)
    gw.on('event', (ev: { type: string }) => events.push(ev.type))
    gw.on('exit', () => {
      exits += 1
      gw.start()
    })
    gw.drain()
    await new Promise(resolve => setTimeout(resolve, 1))

    try {
      gw.start()
      await waitFor(() => events.includes('gateway.start_timeout'), 8000)
      await waitFor(() => {
        const child = children[0]

        return Boolean(child && (child.exitCode !== null || child.signalCode !== null))
      }, 4000)

      const child = children[0]

      expect(events).toContain('gateway.start_timeout')
      expect(events).not.toContain('gateway.ready')
      expect(events).not.toContain('gateway.reconnecting')
      expect(exits).toBe(1)
      expect(children).toHaveLength(1)
      expect(child.exitCode !== null || child.signalCode !== null).toBe(true)
      expect(alive(child.pid ?? 0)).toBe(false)
    } finally {
      restore()
    }
  }, 15_000)

  it('public kill of a live child waits for the spawned shutdown grace', async () => {
    const root = mkdtempSync(join(tmpdir(), 'hermes-shutdown-grace-'))
    const python = join(root, 'gateway')

    writeFileSync(
      python,
      `#!/usr/bin/env node
process.on('SIGTERM', () => {})
process.stdout.write(JSON.stringify({jsonrpc:'2.0',method:'event',params:{type:'gateway.ready',payload:{}}}) + '\\n')
setTimeout(() => {}, 30000)
`
    )
    chmodSync(python, 0o755)

    const restore = isolateSpawnEnv(python, root)
    process.env[GRACE_ENV] = '2'
    const events: string[] = []
    const gw = new GatewayClient()

    clients.push(gw)
    gw.on('event', (ev: { type: string }) => events.push(ev.type))
    gw.drain()
    await new Promise(resolve => setTimeout(resolve, 1))

    try {
      gw.start()
      await waitFor(() => events.includes('gateway.ready'), 4000)
      expect(events).toContain('gateway.ready')

      const child = children[0]
      const pid = child.pid ?? 0
      const started = startTime(pid)

      gw.kill('real-backend-shutdown')
      await new Promise(resolve => setTimeout(resolve, 1500))
      expect(sameProcess(pid, started)).toBe(true)
      expect(child.signalCode).toBeNull()
      await waitFor(() => child.signalCode === 'SIGKILL', 2000)
      expect(child.signalCode).toBe('SIGKILL')
      expect(sameProcess(pid, started)).toBe(false)
    } finally {
      restore()
    }
  }, 15_000)
})
