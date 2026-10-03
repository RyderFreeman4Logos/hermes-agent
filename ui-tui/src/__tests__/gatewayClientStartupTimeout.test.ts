import { chmodSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ChildProcess } from 'node:child_process'

// A startup that never emits gateway.ready must reap the child it spawned.
// HERMES_TUI_STARTUP_TIMEOUT_MS is read at import and floored at 5000ms, so
// the fixture owns that value before the client module loads.

const TIMEOUT_ENV = 'HERMES_TUI_STARTUP_TIMEOUT_MS'
const previousTimeout = process.env[TIMEOUT_ENV]

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

async function waitFor(predicate: () => boolean, ms: number) {
  const deadline = Date.now() + ms

  while (Date.now() < deadline && !predicate()) {
    await new Promise(resolve => setTimeout(resolve, 25))
  }
}

afterEach(async () => {
  if (previousTimeout === undefined) {
    delete process.env[TIMEOUT_ENV]
  } else {
    process.env[TIMEOUT_ENV] = previousTimeout
  }

  for (const client of clients.splice(0)) {
    client.kill('test-cleanup')
  }

  for (const child of children.splice(0)) {
    if (child.exitCode !== null || child.signalCode !== null || !child.pid) {
      continue
    }

    try {
      child.kill('SIGKILL')
    } catch {
      // already gone
    }

    await Promise.race([
      new Promise(resolve => child.once('close', resolve)),
      new Promise(resolve => setTimeout(resolve, 2000))
    ])
  }
})

function hangPython(root: string) {
  const python = join(root, 'hang-python')

  writeFileSync(python, '#!/bin/sh\nwhile true; do sleep 30; done\n')
  chmodSync(python, 0o755)

  return python
}

function isolateSpawnEnv(python: string, root: string) {
  const previous = {
    python: process.env.HERMES_PYTHON,
    root: process.env.HERMES_PYTHON_SRC_ROOT,
    url: process.env.HERMES_TUI_GATEWAY_URL,
    sidecar: process.env.HERMES_TUI_SIDECAR_URL
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

      await new Promise(resolve => setTimeout(resolve, 2000))
      expect(events).toContain('gateway.start_timeout')
      expect(events).not.toContain('gateway.reconnecting')
      expect(children).toEqual([child])
      expect(pid).toBeGreaterThan(0)
      expect(alive(pid)).toBe(false)
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
})
