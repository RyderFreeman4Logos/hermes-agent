import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { resetUiState } from '../app/uiStore.js'
import type { Msg } from '../types.js'

const ref = <T>(current: T) => ({ current })

const buildCtx = (appended: Msg[]) =>
  ({
    composer: {
      dequeue: () => undefined,
      queueEditRef: ref<null | number>(null),
      sendQueued: vi.fn(),
      setInput: vi.fn()
    },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: {
      appendMessage: (msg: Msg) => appended.push(msg),
      panel: vi.fn(),
      setHistoryItems: vi.fn()
    },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

const texts = (appended: Msg[]) => appended.filter(m => m.role === 'assistant' && m.text).map(m => m.text)

const turn = (interim: string, final: string, previewed = false) => {
  const appended: Msg[] = []
  const onEvent = createGatewayEventHandler(buildCtx(appended))

  onEvent({ payload: {}, type: 'message.start' } as any)
  onEvent({ payload: { already_streamed: true, text: interim }, type: 'message.interim' } as any)
  onEvent({ payload: { response_previewed: previewed, text: final }, type: 'message.complete' } as any)

  return texts(appended)
}

describe('commentary and final dedupe', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  it('collapses a byte-identical commentary and final in one turn', () => {
    expect(turn('same reply', 'same reply')).toEqual(['same reply'])
  })

  it('keeps a prefix-distinct final', () => {
    expect(turn('progress', 'progress complete')).toEqual(['progress', 'progress complete'])
  })

  it('keeps a final that differs only by trailing whitespace', () => {
    expect(turn('same reply  ', 'same reply')).toEqual(['same reply  ', 'same reply'])
    expect(turn('same reply', 'same reply  ')).toEqual(['same reply', 'same reply'])
  })

  it('keeps distinct commentary and final replies', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { already_streamed: true, text: 'interim answer' }, type: 'message.interim' } as any)
    onEvent({ payload: { text: 'final answer' }, type: 'message.delta' } as any)
    onEvent({ payload: { text: 'final answer' }, type: 'message.complete' } as any)

    expect(texts(appended)).toEqual(['interim answer', 'final answer'])
  })

  it('does not collapse the same text across turns', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { already_streamed: true, text: 'same reply' }, type: 'message.interim' } as any)
    onEvent({ payload: { text: 'same reply' }, type: 'message.complete' } as any)
    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { text: 'same reply' }, type: 'message.complete' } as any)

    expect(texts(appended)).toEqual(['same reply', 'same reply'])
  })

  it('still settles a previewed interim onto one message', () => {
    expect(turn('same reply', 'same reply', true)).toEqual(['same reply'])
  })
})
