import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, it, vi } from 'vitest'

import { bootstrapLocalGateway } from '../localGatewayBootstrap.js'

// The TUI now attaches to a shared owner. Only its short-lived bootstrap helper
// is owned by the TUI; timing out must reap that helper, never the shared owner.
it('reaps a bootstrap helper that never returns a shared-owner grant', async () => {
  const root = mkdtempSync(join(tmpdir(), 'hermes-bootstrap-timeout-'))
  const python = join(root, 'hang-python')
  const pidFile = join(root, 'pid')

  writeFileSync(python, `#!/bin/sh
exec python3 -c 'import os, pathlib, threading
pathlib.Path("${pidFile}").write_text(str(os.getpid()))
threading.Event().wait()'
`)
  chmodSync(python, 0o755)
  vi.stubEnv('HERMES_PYTHON', python)
  vi.stubEnv('HERMES_PYTHON_SRC_ROOT', root)

  try {
    await expect(bootstrapLocalGateway(true)).rejects.toMatchObject({ killed: true, signal: 'SIGTERM' })
    const pid = Number(readFileSync(pidFile, 'utf8'))

    expect(pid).toBeGreaterThan(0)
    expect(() => process.kill(pid, 0)).toThrow()
  } finally {
    vi.unstubAllEnvs()
  }
}, 55_000)
