import { LONG_MSG } from '../config/limits.js'
import { t } from '../i18n/runtime.js'
import { buildToolTrailLine } from '../lib/text.js'
import type { Msg, SessionInfo } from '../types.js'

export const introMsg = (info: SessionInfo): Msg => ({ info, kind: 'intro', role: 'system', text: '' })

export const userDisplay = (text: string) => {
  if (text.length <= LONG_MSG) {
    return text
  }

  const first = text.split('\n')[0]?.trim() ?? ''
  const words = first.split(/\s+/).filter(Boolean)
  const prefix = (words.length > 1 ? words.slice(0, 4).join(' ') : first).slice(0, 80)

  return t('libText.messages.longMessage', prefix || t('libText.messages.messageFallback'))
}

export const replyEndLine = (value: unknown): string | undefined => {
  if (!value || typeof value !== 'object') {
    return undefined
  }

  const fields = value as Record<string, unknown>
  const elapsed = fields.elapsed_s ?? fields.reply_elapsed_s
  const cacheRead = fields.cache_read ?? fields.reply_cache_read
  const promptTotal = fields.prompt_total ?? fields.reply_prompt_tokens
  const parts: string[] = []

  if (typeof elapsed === 'number' && Number.isFinite(elapsed) && elapsed >= 0) {
    parts.push(`${elapsed.toFixed(1)}s`)
  }

  if (
    typeof cacheRead === 'number' &&
    Number.isSafeInteger(cacheRead) &&
    cacheRead >= 0 &&
    typeof promptTotal === 'number' &&
    Number.isSafeInteger(promptTotal) &&
    promptTotal >= 0 &&
    cacheRead <= promptTotal
  ) {
    const rate = promptTotal > 0 ? ` (${((100 * cacheRead) / promptTotal).toFixed(1)}%)` : ''
    parts.push(`cache ${cacheRead}/${promptTotal}${rate}`)
  }

  return parts.length ? parts.join(' · ') : undefined
}

export const toTranscriptMessages = (rows: unknown): Msg[] => {
  if (!Array.isArray(rows)) {
    return []
  }

  const out: Msg[] = []
  let pending: string[] = []

  for (const row of rows) {
    if (!row || typeof row !== 'object') {
      continue
    }

    const { context, display_kind, name, role, text, timestamp } = row as TranscriptRow

    const createdAt =
      typeof timestamp === 'number' && Number.isFinite(timestamp) && timestamp > 0 ? timestamp : undefined

    if (role === 'tool') {
      pending.push(buildToolTrailLine(name ?? 'tool', context ?? ''))

      continue
    }

    if (typeof text !== 'string' || !text.trim()) {
      continue
    }

    // Display-only timeline events: render as dim ◈ markers instead of
    // opaque user messages. Hidden compaction handoffs are skipped entirely.
    if (display_kind === 'hidden') {
      continue
    }

    if (display_kind === 'model_switch') {
      out.push({ kind: 'event', role: 'system', text: t('libText.messages.modelChanged') })
      pending = []

      continue
    }

    if (display_kind === 'auto_continue') {
      out.push({ kind: 'event', role: 'system', text: t('libText.messages.resumedInterruptedTurn') })
      pending = []

      continue
    }

    if (display_kind === 'personality_switch') {
      out.push({ kind: 'event', role: 'system', text: t('libText.messages.personalityChanged') })
      pending = []

      continue
    }

    if (display_kind === 'async_delegation_complete' || display_kind === 'process_complete') {
      const meta = (row as TranscriptRow).display_metadata
      const count = meta && typeof meta.task_count === 'number' ? meta.task_count : undefined

      const label =
        display_kind === 'process_complete'
          ? t('libText.messages.backgroundProcessFinished')
          : count === undefined
            ? t('libText.messages.backgroundAgentWorkFinished')
            : t(
                count === 1
                  ? 'libText.messages.backgroundAgentsFinishedOne'
                  : 'libText.messages.backgroundAgentsFinishedOther',
                count
              )

      out.push({
        kind: 'event',
        role: 'system',
        text: typeof meta?.display_text === 'string' ? meta.display_text : label
      })
      pending = []

      continue
    }

    if (role === 'assistant') {
      const replyEnd = replyEndLine((row as TranscriptRow).display_metadata?.reply_end)
      out.push({
        role,
        text,
        ...(createdAt !== undefined && { createdAt }),
        ...(replyEnd && { replyEnd }),
        ...(pending.length && { tools: pending })
      })
      pending = []
    } else if (role === 'user' || role === 'system') {
      out.push({ role, text, ...(createdAt !== undefined && { createdAt }) })
      pending = []
    }
  }

  return out
}

export const fmtDuration = (ms: number) => {
  const t = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = t % 60

  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`
}

interface TranscriptRow {
  context?: string
  display_kind?: string
  display_metadata?: { task_count?: number; [key: string]: unknown }
  name?: string
  role?: string
  text?: string
  timestamp?: number
}
