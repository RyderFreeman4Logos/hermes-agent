import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const inputHarness = vi.hoisted(() => ({
  handler: undefined as undefined | ((input: string, key: Record<string, boolean>) => void)
}))

vi.mock('@hermes/ink', async importOriginal => {
  const mod = await importOriginal()

  return {
    ...mod,
    useInput: (handler: (input: string, key: Record<string, boolean>) => void) => {
      inputHarness.handler = handler
    }
  }
})

import type { InputHandlerContext } from '../app/interfaces.js'
import { patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useInputHandlers } from '../app/useInputHandlers.js'

const noop = () => {}

function mountInputHandlers(request: ReturnType<typeof vi.fn>) {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })

  const context = {
    actions: {
      appendMessage: noop,
      sys: noop
    },
    composer: {
      actions: {},
      refs: {},
      state: {
        completions: [],
        historyIdx: null,
        input: '',
        inputBuf: []
      }
    },
    gateway: {
      gw: { request },
      rpc: vi.fn()
    },
    terminal: { stdout },
    voice: {},
    wheelStep: 3
  } as unknown as InputHandlerContext

  function Harness() {
    useInputHandlers(context)

    return null
  }

  inputHarness.handler = undefined

  const instance = renderSync(React.createElement(Harness), {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  return () => {
    instance.unmount()
    instance.cleanup()
  }
}

describe('useInputHandlers — Esc stop-loss through blocked overlays', () => {
  beforeEach(() => {
    turnController.fullReset()
    resetOverlayState()
    resetUiState()
  })

  it.each([
    ['UI busy', { busy: true, gatewayTurnRunning: false }],
    ['gateway-running zombie', { busy: false, gatewayTurnRunning: true }]
  ])('interrupts an active turn behind a pager overlay when %s', (_label, running) => {
    const request = vi.fn().mockResolvedValue(null)

    patchUiState({ ...running, sid: 'sid-blocked' })
    patchOverlayState({ pager: { lines: ['blocked'], offset: 0 } })
    const cleanup = mountInputHandlers(request)

    expect(inputHarness.handler).toBeTypeOf('function')
    inputHarness.handler?.('', { ctrl: false, escape: true })

    expect(request).toHaveBeenCalledWith('session.interrupt', { session_id: 'sid-blocked' })
    cleanup()
  })
})
