import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { $turnState, getTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { STREAM_BATCH_MS } from '../config/timing.js'

const REASONING_COMPACT_AT_CHARS = 80_000

describe('TurnController reasoning delta batching', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    resetUiState()
    turnController.fullReset()
    patchUiState({ showReasoning: true })
  })

  afterEach(() => {
    turnController.fullReset()
    vi.useRealTimers()
  })

  it('publishes one cumulative reasoning segment per stream batch', () => {
    const chunks = Array.from({ length: 500 }, (_, index) => `${index}|`)
    let publications = 0

    const unlisten = $turnState.listen((state, previous) => {
      if (state.streamSegments !== previous.streamSegments) {
        publications++
      }
    })

    for (const chunk of chunks) {
      turnController.recordReasoningDelta(chunk)
    }

    expect(publications).toBe(0)
    vi.advanceTimersByTime(STREAM_BATCH_MS - 1)
    expect(publications).toBe(0)
    vi.advanceTimersByTime(1)
    expect(publications).toBe(1)
    expect(getTurnState().streamSegments[0]?.thinking).toBe(chunks.join(''))

    turnController.recordReasoningDelta('tail-1')
    turnController.recordReasoningDelta('tail-2')
    expect(publications).toBe(1)
    vi.advanceTimersByTime(STREAM_BATCH_MS)
    expect(publications).toBe(2)
    expect(getTurnState().streamSegments[0]?.thinking).toBe(`${chunks.join('')}tail-1tail-2`)

    unlisten()
  })

  it('flushes pending reasoning when visible answer streaming starts', () => {
    turnController.recordReasoningDelta('phase reasoning')
    turnController.recordMessageDelta({ text: 'answer' })

    expect(getTurnState().streamSegments.map(message => message.thinking)).toEqual(['phase reasoning'])
  })

  it('flushes pending reasoning at tool and terminal boundaries without merging phases', () => {
    turnController.recordReasoningDelta('plan ')
    turnController.recordReasoningDelta('one')
    turnController.recordToolStart('tool-1', 'read_file', 'src/a.ts')
    turnController.recordToolComplete('tool-1', 'read_file', undefined, 'done')
    turnController.recordReasoningDelta('plan two')

    const { finalMessages } = turnController.recordMessageComplete({ text: 'done' })
    const reasoning = finalMessages.filter(message => message.thinking).map(message => message.thinking)

    expect(reasoning).toEqual(['plan one', 'plan two'])
    expect(finalMessages.find(message => message.thinking === 'plan one')?.tools).toHaveLength(1)
    expect(finalMessages.at(-1)).toMatchObject({ role: 'assistant', text: 'done' })

    vi.runAllTimers()
    expect(getTurnState().streamSegments).toEqual([])
  })

  it('bounds pending live text while retaining the exact final stream and isolates resets', () => {
    const chunks = Array.from({ length: 101 }, (_, index) => `${String(index).padStart(3, '0')}:${'x'.repeat(997)}`)
    const expected = chunks.join('')

    for (const chunk of chunks) {
      turnController.recordReasoningDelta(chunk)
    }

    const pendingChars = (turnController as unknown as { reasoningDeltaChars: number }).reasoningDeltaChars

    expect(pendingChars).toBeLessThanOrEqual(REASONING_COMPACT_AT_CHARS)

    const { finalMessages } = turnController.recordMessageComplete({ text: '' })
    const retained = finalMessages.map(message => message.thinking ?? '').join('')

    expect(retained).toBe(expected)

    turnController.startMessage()
    turnController.recordReasoningDelta('session A')
    turnController.fullReset()
    vi.advanceTimersByTime(STREAM_BATCH_MS)
    expect(getTurnState().streamSegments).toEqual([])
    expect(getTurnState().reasoning).toBe('')

    turnController.startMessage()
    turnController.recordReasoningDelta('session B')
    vi.advanceTimersByTime(STREAM_BATCH_MS)
    expect(getTurnState().streamSegments.map(message => message.thinking)).toEqual(['session B'])
  })
})
