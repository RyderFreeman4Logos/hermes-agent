import type { Usage } from '@hermes/shared/gateway-events'

import type { Msg } from '../types.js'

const replyKeptText = (
  wasInterrupted: boolean,
  interruptedTo: string | undefined,
  finalText: string,
  finalMessages: Msg[]
): boolean =>
  Boolean(
    (wasInterrupted ? (interruptedTo ?? finalText) : finalText).trim() ||
      finalMessages.some(m => m.role === 'assistant' && m.text.trim())
  )

// Failed turn, no reply: the backend's assistant slot is "Error: <raw body>".
// Swap only that trailing slot for the structured title; interim segments stay.
export const completeMessages = (
  finalMessages: Msg[],
  finalText: string,
  failed: boolean,
  failureText: string,
  isBare: (text: string) => boolean
): Msg[] => {
  if (!failed) {
    return finalMessages.length ? finalMessages : [{ role: 'assistant', text: finalText }]
  }

  const kept = finalMessages.filter(
    (m, i) => !(i === finalMessages.length - 1 && m.role === 'assistant' && isBare(m.text))
  )

  return [...kept, { role: 'assistant', text: failureText }]
}

// Display only. Elapsed is the backend's frozen turn duration. Cache counts are
// this turn's measured hit/total; a missing count is omitted, never shown as 0.
// An interrupted reply that kept text still gets the line. No retained text → no footer.
export const appendReplyEnd = (
  usage: Usage | null | undefined,
  appendMessage: (msg: Msg) => void,
  kept: { finalMessages: Msg[]; finalText: string; interruptedTo?: string; wasInterrupted: boolean }
): void => {
  if (!replyKeptText(kept.wasInterrupted, kept.interruptedTo, kept.finalText, kept.finalMessages)) {
    return
  }

  const parts: string[] = []
  const elapsed = usage?.reply_elapsed_s

  if (typeof elapsed === 'number' && Number.isFinite(elapsed) && elapsed >= 0) {
    parts.push(`${elapsed.toFixed(1)}s`)
  }

  const read = usage?.reply_cache_read
  const total = usage?.reply_prompt_tokens

  if (typeof read === 'number' && typeof total === 'number' && Number.isInteger(read) && Number.isInteger(total) && total > 0) {
    parts.push(`cache ${read}/${total} (${((100 * read) / total).toFixed(1)}%)`)
  }

  if (parts.length) {
    appendMessage({ role: 'system', text: parts.join(' · ') })
  }
}
