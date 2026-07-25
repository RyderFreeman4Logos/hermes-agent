import { describe, expect, it, vi } from 'vitest'

import { sessionCommands } from '../app/slash/commands/session.js'

const compressCommand = sessionCommands.find(cmd => cmd.name === 'compress')!

const guarded =
  <T>(fn: (r: T) => void) =>
  (r: null | T) => {
    if (r) {
      fn(r)
    }
  }

const buildCtx = (result: unknown) => {
  const sys = vi.fn()
  const rpc = vi.fn(() => Promise.resolve(result))
  const ctx = {
    gateway: { rpc },
    guarded,
    guardedErr: vi.fn(),
    sid: 'sid-1',
    transcript: { setHistoryItems: vi.fn(), sys }
  }

  const run = async () => {
    compressCommand.run('', ctx as any, 'compress')
    await rpc.mock.results[0]?.value
    await Promise.resolve()
    await Promise.resolve()
  }

  return { run, sys }
}

describe('/compress slash command', () => {
  it('test_tui_lock_skip_message', async () => {
    const { run, sys } = buildCtx({ lock_held: true })

    await run()

    expect(sys).toHaveBeenCalledWith('⏳ Compression skipped: database busy, try again')
    expect(sys).not.toHaveBeenCalledWith('nothing to compress')
  })
})
