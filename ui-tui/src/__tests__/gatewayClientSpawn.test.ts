import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { GatewayClient } from '../gatewayClient.js'

const ENV_KEYS = [
  'HERMES_CWD',
  'HERMES_PYTHON',
  'HERMES_PYTHON_SRC_ROOT',
  'HERMES_TUI_GATEWAY_URL',
  'HERMES_TUI_SIDECAR_URL'
] as const

const gatewayEntrypoint = (origin: string) => `
import json
import sys

event = {
    "jsonrpc": "2.0",
    "method": "event",
    "params": {"type": "gateway.ready", "payload": {"origin": "${origin}"}},
}
print(json.dumps(event), flush=True)
for _line in sys.stdin:
    pass
`

async function writeGatewayPackage(root: string, origin: string): Promise<void> {
  const packageDir = join(root, 'tui_gateway')

  await mkdir(packageDir, { recursive: true })
  await writeFile(join(packageDir, '__init__.py'), '')
  await writeFile(join(packageDir, 'entry.py'), gatewayEntrypoint(origin))
}

describe('GatewayClient spawned gateway import isolation', () => {
  it('loads tui_gateway from the installed source root instead of HERMES_CWD', async () => {
    const fixtureRoot = await mkdtemp(join(tmpdir(), 'hermes-tui-gateway-spawn-'))
    const installedRoot = join(fixtureRoot, 'installed')
    const workspaceRoot = join(fixtureRoot, 'workspace')
    const savedEnv = new Map(ENV_KEYS.map(key => [key, process.env[key]]))
    const client = new GatewayClient()
    let started = false

    try {
      await writeGatewayPackage(installedRoot, 'installed')
      await writeGatewayPackage(workspaceRoot, 'workspace')

      process.env.HERMES_PYTHON_SRC_ROOT = installedRoot
      process.env.HERMES_CWD = workspaceRoot
      process.env.HERMES_PYTHON ||= process.platform === 'win32' ? 'python' : 'python3'
      delete process.env.HERMES_TUI_GATEWAY_URL
      delete process.env.HERMES_TUI_SIDECAR_URL

      const origin = new Promise<string>((resolve, reject) => {
        const timeout = setTimeout(
          () => reject(new Error(`gateway did not become ready:\n${client.getLogTail(20)}`)),
          5000
        )

        client.on('event', event => {
          if (event.type === 'gateway.ready') {
            clearTimeout(timeout)
            resolve(String(event.payload?.origin ?? 'missing'))
          }
        })
      })

      client.start()
      started = true
      client.drain()

      await expect(origin).resolves.toBe('installed')
    } finally {
      try {
        if (started) {
          const exited = new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('gateway child did not exit after kill')), 5000)

            client.once('exit', () => {
              clearTimeout(timeout)
              resolve()
            })
          })

          client.kill('gatewayClientSpawn.test cleanup')
          await exited
        }
      } finally {
        for (const [key, value] of savedEnv) {
          if (value === undefined) {
            delete process.env[key]
          } else {
            process.env[key] = value
          }
        }

        await rm(fixtureRoot, { force: true, recursive: true })
      }
    }
  })
})
