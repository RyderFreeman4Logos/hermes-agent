import { chmodSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

// Issue #352: a startup that never emits gateway.ready must reap the child it
// spawned. The timer only warns today, so the process stays alive and a later
// parent death orphans it. HERMES_TUI_STARTUP_TIMEOUT_MS is read at import and
// floored at 5000ms, so this file must be launched with that env already set.

const spawned: number[] = []

vi.mock('node:child_process', async () => {
  const actual = await vi.importActual<typeof import('node:child_process')>('node:child_process')

  return {
    ...actual,
    spawn: (...args: Parameters<typeof actual.spawn>) => {
      const child = actual.spawn(...args)

      if (child.pid) {
        spawned.push(child.pid)
      }

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

afterEach(() => {
  for (const client of clients.splice(0)) {
    client.kill('test-cleanup')
  }

  for (const pid of spawned.splice(0)) {
    if (!alive(pid)) {
      continue
    }

    try {
      process.kill(pid, 'SIGKILL')
    } catch {
      // already gone
    }
  }
})

describe('GatewayClient startup timeout cleanup', () => {
  it('reaps a spawned gateway that never becomes ready', async () => {
    const root = mkdtempSync(join(tmpdir(), 'hermes-352-'))
    const python = join(root, 'hang-python')

    writeFileSync(python, "#!/bin/sh\nexec python3 -c 'import time\nwhile True:\n time.sleep(30)'\n")
    chmodSync(python, 0o755)

    const previous = {
      python: process.env.HERMES_PYTHON,
      root: process.env.HERMES_PYTHON_SRC_ROOT,
      url: process.env.HERMES_TUI_GATEWAY_URL
    }

    process.env.HERMES_PYTHON = python
    process.env.HERMES_PYTHON_SRC_ROOT = root
    delete process.env.HERMES_TUI_GATEWAY_URL

    const events: string[] = []
    const gw = new GatewayClient()

    clients.push(gw)
    gw.on('event', (ev: { type: string }) => events.push(ev.type))
    gw.drain()
    await new Promise(resolve => setTimeout(resolve, 20))

    try {
      gw.start()

      const deadline = Date.now() + 8000

      while (Date.now() < deadline && !events.includes('gateway.start_timeout')) {
        await new Promise(resolve => setTimeout(resolve, 50))
      }

      const pid = spawned.at(-1) ?? 0

      await new Promise(resolve => setTimeout(resolve, 2000))
      expect(events).toContain('gateway.start_timeout')
      expect(events).not.toContain('gateway.reconnecting')
      expect(spawned).toEqual([pid])
      expect(pid).toBeGreaterThan(0)
      expect(alive(pid)).toBe(false)
    } finally {
      if (previous.python === undefined) delete process.env.HERMES_PYTHON
      else process.env.HERMES_PYTHON = previous.python
      if (previous.root === undefined) delete process.env.HERMES_PYTHON_SRC_ROOT
      else process.env.HERMES_PYTHON_SRC_ROOT = previous.root
      if (previous.url === undefined) delete process.env.HERMES_TUI_GATEWAY_URL
      else process.env.HERMES_TUI_GATEWAY_URL = previous.url
    }
  }, 15_000)
})
