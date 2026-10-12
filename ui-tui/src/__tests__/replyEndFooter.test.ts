import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import type { Msg } from '../types.js'

const ref = <T>(current: T) => ({ current })

const buildCtx = (appended: Msg[]) =>
  ({
    composer: { dequeue: () => undefined, queueEditRef: ref<null | number>(null), sendQueued: vi.fn(), setInput: vi.fn() },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: { STARTUP_RESUME_ID: '', colsRef: ref(80), newSession: vi.fn(), resetSession: vi.fn(), resumeById: vi.fn(), setCatalog: vi.fn() },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: {
      appendMessage: (msg: Msg) => appended.push(msg),
      panel: vi.fn(),
      setHistoryItems: (next: ((prev: Msg[]) => Msg[]) | Msg[]) => {
        const value = typeof next === 'function' ? next([...appended]) : next
        appended.splice(0, appended.length, ...value)
      }
    },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

describe('reply-end footer', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  it('prints elapsed and raw hit/total, and elapsed alone when cache is absent', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    onEvent({
      payload: { status: 'complete', text: 'done', usage: { reply_cache_read: 800, reply_elapsed_s: 12.34, reply_prompt_tokens: 1000 } },
      type: 'message.complete'
    } as any)
    expect(appended.at(-1)).toMatchObject({ role: 'system', text: '12.3s · cache 800/1000 (80.0%)' })

    appended.length = 0
    onEvent({ payload: { status: 'complete', text: 'done', usage: { reply_elapsed_s: 2 } }, type: 'message.complete' } as any)
    expect(appended.at(-1)).toMatchObject({ role: 'system', text: '2.0s' })
    expect(appended.some(m => m.text.includes('cache'))).toBe(false)
  })

  it('prints elapsed for an interrupted reply that kept text', () => {
    vi.useFakeTimers()

    try {
      const history: Msg[] = []
      const ctx = buildCtx(history)
      ctx.gateway.gw.request = vi.fn(async () => ({ status: 'interrupted' }))
      const onEvent = createGatewayEventHandler(ctx)
      patchUiState({ sid: 'sess-1' })
      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({ payload: { text: 'kept partial' }, type: 'message.delta' } as any)
      turnController.interruptTurn({
        appendMessage: (msg: Msg) => history.push(msg),
        gw: ctx.gateway.gw,
        sid: 'sess-1',
        sys: ctx.system.sys
      })
      onEvent({
        payload: { status: 'interrupted', text: 'kept partial', usage: { reply_elapsed_s: 4.2 } },
        type: 'message.complete'
      } as any)
      expect(history.some(m => m.role === 'system' && m.text === '4.2s')).toBe(true)
      expect(history.some(m => m.text.includes('cache'))).toBe(false)
    } finally {
      vi.runAllTimers()
      vi.useRealTimers()
    }
  })
})
