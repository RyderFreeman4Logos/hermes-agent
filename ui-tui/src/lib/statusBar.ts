export const STATUS_BAR_SEGMENTS = [
  'battery',
  'indicator',
  'model',
  'context_tokens',
  'context_bar',
  'context_percent',
  'focus',
  'heartbeat',
  'session_duration',
  'idle',
  'compressions',
  'voice',
  'sessions',
  'bg_tasks',
  'subagents',
  'resume',
  'dev_credits',
  'spawn_hud',
  'cwd'
] as const

export type StatusBarSegment = (typeof STATUS_BAR_SEGMENTS)[number]

export const DEFAULT_STATUS_BAR_SEGMENTS: readonly StatusBarSegment[] = STATUS_BAR_SEGMENTS

const STATUS_BAR_SEGMENT_SET: ReadonlySet<string> = new Set(STATUS_BAR_SEGMENTS)
const LEGACY_CONTEXT_SEGMENTS: readonly StatusBarSegment[] = ['context_tokens', 'context_bar', 'context_percent']

export function normalizeStatusBarSegments(raw: unknown): StatusBarSegment[] {
  if (!Array.isArray(raw)) {
    return [...DEFAULT_STATUS_BAR_SEGMENTS]
  }

  if (raw.length === 0) {
    return []
  }

  const normalized: StatusBarSegment[] = []
  const add = (segment: StatusBarSegment) => {
    if (!normalized.includes(segment)) {
      normalized.push(segment)
    }
  }

  for (const value of raw) {
    if (typeof value !== 'string') {
      continue
    }

    const segment = value.trim().toLowerCase()

    if (segment === 'context') {
      LEGACY_CONTEXT_SEGMENTS.forEach(add)
    } else if (STATUS_BAR_SEGMENT_SET.has(segment)) {
      add(segment as StatusBarSegment)
    }
  }

  return normalized.length ? normalized : [...DEFAULT_STATUS_BAR_SEGMENTS]
}

export interface StatusRowItem {
  breakBefore?: boolean
  width: number
}

export function packStatusRows<T extends StatusRowItem>(
  items: readonly T[],
  width: number,
  separatorWidth: number,
  firstRowPrefixWidth = 0
): T[][] {
  const available = Math.max(1, Math.floor(width || 1))
  const separator = Math.max(0, Math.floor(separatorWidth || 0))
  const rows: T[][] = []
  let row: T[] = []
  let used = Math.max(0, Math.floor(firstRowPrefixWidth || 0))

  for (const item of items) {
    const itemWidth = Math.max(0, Math.floor(item.width || 0))
    const nextWidth = row.length ? used + separator + itemWidth : used + itemWidth

    if (row.length && (item.breakBefore || nextWidth > available)) {
      rows.push(row)
      row = []
      used = 0
    }

    row.push(item)
    used = row.length === 1 ? used + itemWidth : used + separator + itemWidth
  }

  if (row.length) {
    rows.push(row)
  }

  return rows
}
