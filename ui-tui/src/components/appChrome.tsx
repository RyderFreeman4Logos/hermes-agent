import { Box, type ScrollBoxHandle, stringWidth, Text } from '@hermes/ink'
import { compactNumber } from '@hermes/shared/format'
import type { Usage } from '@hermes/shared/gateway-events'
import { useStore } from '@nanostores/react'
import { Fragment, type ReactNode, type RefObject, useEffect, useMemo, useRef, useState } from 'react'
import unicodeSpinners from 'unicode-animations'

import { $delegationState } from '../app/delegationStore.js'
import type { BatteryInfo, IndicatorStyle, Notice } from '../app/interfaces.js'
import { $isStatusRuleOccluded } from '../app/overlayStore.js'
import { useTurnSelector } from '../app/turnStore.js'
import { DEV_CREDITS_MODE } from '../config/env.js'
import { faces } from '../content/faces.js'
import { VERBS } from '../content/verbs.js'
import { fmtDuration } from '../domain/messages.js'
import { stickyPromptFromViewport } from '../domain/viewport.js'
import { messages, t as tr } from '../i18n/runtime.js'
import { buildSubagentTree, treeTotals, widthByDepth } from '../lib/subagentTree.js'
import { useScrollbarSnapshot, useViewportSnapshot } from '../lib/viewportStore.js'
import type { Theme } from '../theme.js'
import type { Msg } from '../types.js'

import { scrollbarColors } from './overlayPrimitives.js'

const FACE_TICK_MS = 2500
const HEART_COLORS = ['#ff5fa2', '#ff4d6d']

// Keep verb segment width stable so status-bar content to the right doesn't
// jitter when the ticker rotates between short/long verbs.
export const VERB_PAD_LEN = VERBS.reduce((max, v) => Math.max(max, v.length), 0) + 1 // + ellipsis
export const padVerb = (verb: string) => `${verb}…`.padEnd(VERB_PAD_LEN, ' ')

// Compact alternates for the `emoji` and `ascii` indicator styles.
// Each entry is a fixed-width (display-width) glyph.
const EMOJI_FRAMES = ['☤ ', '🌀', '🤔', '✨', '🍵', '🔮']
const ASCII_FRAMES = ['|', '/', '-', '\\']

// Faster tick for spinner-style indicators — they read as motion only
// at frame rates closer to their authored interval.
const SPINNER_TICK_MS = 100

interface IndicatorRender {
  frame: string
  intervalMs: number
  // When false, FaceTicker hides the rotating verb and just shows the
  // glyph + duration.  Lets `unicode` stay minimal while the other
  // styles keep the verb-rotation flavour users associate with the
  // running… status.
  showVerb: boolean
}

const renderIndicator = (style: IndicatorStyle, tick: number): IndicatorRender => {
  if (style === 'kaomoji') {
    const frames = faces()

    return { frame: frames[tick % frames.length] ?? '', intervalMs: FACE_TICK_MS, showVerb: true }
  }

  if (style === 'emoji') {
    return {
      frame: EMOJI_FRAMES[tick % EMOJI_FRAMES.length] ?? '☤ ',
      intervalMs: SPINNER_TICK_MS * 6,
      showVerb: true
    }
  }

  if (style === 'ascii') {
    return {
      frame: ASCII_FRAMES[tick % ASCII_FRAMES.length] ?? '|',
      intervalMs: SPINNER_TICK_MS,
      showVerb: true
    }
  }

  // 'unicode' — braille spinner (fixed 1-col).  Authored interval is
  // ~80ms; honour it but bound below at a safe minimum so React
  // re-renders stay reasonable.  This style is for users who want
  // the cleanest possible status, so no verb rotation either.
  const spinner = unicodeSpinners.braille
  const frame = spinner.frames[tick % spinner.frames.length] ?? '⠋'

  return { frame, intervalMs: Math.max(SPINNER_TICK_MS, spinner.interval), showVerb: false }
}

// `EMOJI_FRAMES` is static, so measure its widest glyph once at module load
// instead of rescanning on every status render. `faces()` follows the active
// catalog, so its width is memoised per catalog table instead.
const EMOJI_FRAME_WIDTH = EMOJI_FRAMES.reduce((max, f) => Math.max(max, stringWidth(f)), 1)

let kaomojiWidthCache: { table: unknown; width: number } | null = null

const kaomojiFrameWidth = (): number => {
  const table = messages().content.faces

  if (kaomojiWidthCache?.table !== table) {
    kaomojiWidthCache = { table, width: Object.values(table).reduce((max, f) => Math.max(max, stringWidth(f)), 1) }
  }

  return kaomojiWidthCache.width
}

const indicatorFrameWidth = (style: IndicatorStyle): number => {
  if (style === 'kaomoji') {
    return kaomojiFrameWidth()
  }

  if (style === 'emoji') {
    return EMOJI_FRAME_WIDTH
  }

  // 'ascii' and 'unicode' are single-column glyphs.
  return 1
}

// Bounded width of the elapsed-time clock, derived from `fmtDuration` itself so
// the reservation/budget stays consistent with what actually renders (it emits
// a space between units, e.g. `59m 59s` / `99h 59m`). Durations beyond this
// (100h+) are left to clip rather than reserving unbounded width.
export const MAX_DURATION_WIDTH = Math.max(
  stringWidth(fmtDuration(59 * 60_000 + 59_000)), // "59m 59s"
  stringWidth(fmtDuration(99 * 3_600_000 + 59 * 60_000)) // "99h 59m"
)

// Display width to reserve for the busy indicator so its verb + elapsed-time
// tail can't shove the model off-screen on narrow terminals. Style-aware:
// `unicode` is a bare 1-col braille spinner with no verb, while kaomoji/emoji/
// ascii add a fixed-width verb; any style adds a bounded elapsed-time tail.
// Mirrors FaceTicker's `frame + verbSegment + durationSegment` layout.
export const busyIndicatorWidth = (style: IndicatorStyle, hasDuration: boolean): number => {
  const { showVerb } = renderIndicator(style, 0)
  const verb = showVerb ? 1 + VERB_PAD_LEN : 0
  // ` · ` plus the bounded clock (e.g. `59m 59s`).
  const duration = hasDuration ? stringWidth(' · ') + MAX_DURATION_WIDTH : 0

  return indicatorFrameWidth(style) + verb + duration
}

function FaceTicker({
  color,
  startedAt,
  style,
  verbOverride
}: {
  color: string
  startedAt?: null | number
  style: IndicatorStyle
  verbOverride?: string
}) {
  const [tick, setTick] = useState(() => Math.floor(Math.random() * 1000))
  const [verbTick, setVerbTick] = useState(() => Math.floor(Math.random() * VERBS.length))
  const [now, setNow] = useState(() => Date.now())
  const isOccluded = useStore($isStatusRuleOccluded)

  // Pre-compute cadence + verb-visibility for the active style so an
  // `/indicator` switch re-arms the interval (and skips the verb timer
  // for verb-less styles like `unicode`) without leaving the previous
  // timer dangling. A frozen override (idle compaction) always shows the
  // verb so "compacting…" is visible even in unicode style (#97239).
  const { intervalMs, showVerb } = renderIndicator(style, 0)
  const freezeVerb = Boolean(verbOverride)
  const displayVerb = freezeVerb || showVerb

  useEffect(() => {
    // An overlay is painted OVER the status rule (the modal widget slot, or a
    // floating panel growing up over the top rule), so every tick below is a
    // re-render nobody can see — in an Ink TUI that churn reads as the dialog
    // tearing.  Arm nothing while occluded.  The effect re-runs when the rule
    // is revealed again and re-seeds `now` from the wall clock, so the elapsed
    // read-out resumes live rather than frozen at the moment it was covered.
    // See `$isStatusRuleOccluded` for why this is NOT `$isBlocked`.
    if (isOccluded) {
      return
    }

    setNow(Date.now())

    const glyph = setInterval(() => setTick(n => n + 1), intervalMs)
    const clock = setInterval(() => setNow(Date.now()), 1000)
    // Verb timer is gated on `displayVerb` — `unicode` style hides the verb
    // entirely, so cycling `verbTick` would be an avoidable re-render. A
    // frozen override does not rotate.
    const verb = displayVerb && !freezeVerb ? setInterval(() => setVerbTick(n => n + 1), FACE_TICK_MS) : null

    return () => {
      clearInterval(glyph)
      clearInterval(clock)

      if (verb !== null) {
        clearInterval(verb)
      }
    }
  }, [displayVerb, freezeVerb, intervalMs, isOccluded])

  const { frame } = renderIndicator(style, tick)
  const verb = verbOverride ?? VERBS[verbTick % VERBS.length] ?? ''
  const verbSegment = displayVerb ? ` ${padVerb(verb)}` : ''
  // Leading space keeps a gap between the frame and the duration when the
  // verb segment is hidden (e.g. `unicode` spinner style).  When the verb
  // IS shown, its trailing padding already provides the gap, so the extra
  // space is harmless.
  const durationSegment = startedAt ? ` · ${fmtDuration(now - startedAt)}` : ''

  return (
    <Text color={color}>
      {frame}
      {verbSegment}
      {durationSegment}
    </Text>
  )
}

function ctxBarColor(pct: number | undefined, t: Theme) {
  if (pct == null) {
    return t.color.muted
  }

  if (pct >= 95) {
    return t.color.statusCritical
  }

  if (pct > 80) {
    return t.color.statusBad
  }

  if (pct >= 50) {
    return t.color.statusWarn
  }

  return t.color.statusGood
}

function statusSessionCountLabel(count: number) {
  return tr('status.sessionCount', count)
}

// State values the app layer compares against ('ready', 'running…') are shown
// through the catalog; anything else (a live verb, a slash reply) passes through.
export function displayStatus(status: string): string {
  const s = messages().status

  if (status === 'ready') {
    return s.ready
  }

  if (status === 'running…') {
    return s.running
  }

  if (status === 'summoning hermes…') {
    return s.summoning
  }

  return status
}

// Colour the battery read-out by its (Python-computed) category. Inverted vs
// the context bar — a full battery is "good", an empty one "critical".
function batteryColor(info: BatteryInfo, t: Theme): string {
  if (info.category === 'good') {
    return t.color.statusGood
  }

  if (info.category === 'warn') {
    return t.color.statusWarn
  }

  if (info.category === 'bad') {
    return t.color.statusBad
  }

  if (info.category === 'critical') {
    return t.color.statusCritical
  }

  return t.color.muted
}

// Compact battery label: a bolt while charging, else a battery glyph.
// Renders `--` for an unknown percent so a null can never surface as "null%".
function batteryLabel(info: BatteryInfo): string {
  return `${info.plugged ? '⚡' : '🔋'} ${info.percent ?? '--'}%`
}

// Colour a credits notice by its level. The notice TEXT already carries its
// own glyph (⚠ • ✕ ✓) from the Python policy — we only tint it here, never
// prepend another glyph. `success` maps to the theme's green status colour.
function noticeColor(level: Notice['level'], t: Theme): string {
  if (level === 'error') {
    return t.color.error
  }

  if (level === 'warn') {
    return t.color.warn
  }

  if (level === 'success') {
    return t.color.statusGood
  }

  // 'info' / undefined — keep it readable but understated.
  return t.color.accent
}

function ctxBar(pct: number | undefined, w = 10) {
  const p = Math.max(0, Math.min(100, pct ?? 0))
  const filled = Math.round((p / 100) * w)

  return '█'.repeat(filled) + '░'.repeat(w - filled)
}

// `minLeftContent` is the display width of the high-priority left segments
// (status indicator + model + context). Reserving it makes the cwd/branch
// segment on the right yield FIRST on narrow terminals, instead of squeezing
// the loading indicator and model down to nothing.
export function statusRuleWidths(cols: number, cwdLabel: string, minLeftContent = 0) {
  const width = Math.max(1, Math.floor(cols || 1))
  const desiredSeparatorWidth = width >= 24 ? 3 : 1
  const baseMinLeft = width >= 24 ? 8 : 1
  // Never reserve more than the terminal width; never less than the historical
  // floor. With the default `minLeftContent = 0` this is identical to the old
  // behaviour, so callers that don't pass content are unaffected.
  const minLeftWidth = Math.min(width, Math.max(baseMinLeft, Math.floor(minLeftContent)))
  const maxRightWidth = Math.max(0, width - desiredSeparatorWidth - minLeftWidth)

  if (!cwdLabel || maxRightWidth <= 0) {
    return { leftWidth: width, rightWidth: 0, separatorWidth: 0 }
  }

  const rightWidth = Math.max(0, Math.min(stringWidth(cwdLabel), maxRightWidth))
  const separatorWidth = rightWidth > 0 ? desiredSeparatorWidth : 0
  const leftWidth = Math.max(1, width - separatorWidth - rightWidth)

  return { leftWidth, rightWidth, separatorWidth }
}

// Progressive disclosure for the status rule's lower-priority tail segments.
// As the terminal narrows we shed the least important pieces first (cost →
// bg → voice → compressions → duration → context bar), and below the bar
// breakpoint the context read-out collapses to a bare token count. Status and
// model are never gated here — they're guaranteed room by `statusRuleWidths`.
export interface StatusBarSegments {
  bar: boolean
  bg: boolean
  cacheHit: boolean
  compactCtx: boolean
  compressions: boolean
  duration: boolean
  latency: boolean
  subagents: boolean
  tps: boolean
  voice: boolean
}

export function statusBarSegments(cols: number): StatusBarSegments {
  const w = Math.max(1, Math.floor(cols || 1))

  return {
    compactCtx: w < 72,
    bar: w >= 72,
    duration: w >= 76,
    compressions: w >= 80,
    voice: w >= 84,
    bg: w >= 88,
    subagents: w >= 92,
    cacheHit: w >= 96,
    latency: w >= 104,
    tps: w >= 110
  }
}

function SpawnHud({
  cols,
  leadingRule = false,
  standalone = false,
  t
}: {
  cols?: number
  leadingRule?: boolean
  standalone?: boolean
  t: Theme
}) {
  // Tight HUD that only appears when the session is actually fanning out.
  // Colour escalates to warn/error as depth or concurrency approaches the cap.
  const delegation = useStore($delegationState)
  const subagents = useTurnSelector(state => state.subagents)

  const tree = useMemo(() => buildSubagentTree(subagents), [subagents])
  const totals = useMemo(() => treeTotals(tree), [tree])

  if (!totals.descendantCount && !delegation.paused) {
    return null
  }

  const maxDepth = delegation.maxSpawnDepth
  const maxConc = delegation.maxConcurrentChildren
  const depth = Math.max(0, totals.maxDepthFromHere)
  const active = totals.activeCount

  // `max_concurrent_children` is a per-parent cap, not a global one.
  // `activeCount` sums every running agent across the tree and would
  // over-warn for multi-orchestrator runs.  The widest level of the tree
  // is a closer proxy to "most concurrent spawns that could be hitting a
  // single parent's slot budget".
  const widestLevel = widthByDepth(tree).reduce((a, b) => Math.max(a, b), 0)
  const depthRatio = maxDepth ? depth / maxDepth : 0
  const concRatio = maxConc ? widestLevel / maxConc : 0
  const ratio = Math.max(depthRatio, concRatio)

  const color = delegation.paused || ratio >= 1 ? t.color.error : ratio >= 0.66 ? t.color.warn : t.color.muted

  const pieces: string[] = []

  if (delegation.paused) {
    pieces.push(tr('status.paused'))
  }

  if (totals.descendantCount > 0) {
    const depthLabel = maxDepth ? `${depth}/${maxDepth}` : `${depth}`
    pieces.push(`d${depthLabel}`)

    if (active > 0) {
      // Label pairs the widest-level count (drives concRatio above) with
      // the total active count for context.  `W/cap` triggers the warn,
      // `+N` is everything else currently running across the tree.
      const extra = Math.max(0, active - widestLevel)
      const widthLabel = maxConc ? `${widestLevel}/${maxConc}` : `${widestLevel}`
      const suffix = extra > 0 ? `+${extra}` : ''
      pieces.push(`⚡${widthLabel}${suffix}`)
    }
  }

  const atCap = depthRatio >= 1 || concRatio >= 1

  const content = (
    <Text color={color}>
      {standalone ? `${leadingRule ? '─ ' : ''}${atCap ? '⚠ ' : ''}` : atCap ? ' │ ⚠ ' : ' │ '}
      {pieces.join(' ')}
    </Text>
  )

  return standalone ? (
    <Box flexShrink={0} height={1} overflow="hidden" width={Math.max(1, Math.floor(cols || 1))}>
      {content}
    </Box>
  ) : (
    content
  )
}

function SessionDuration({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => Date.now())
  const isOccluded = useStore($isStatusRuleOccluded)

  useEffect(() => {
    // Paused only while an overlay actually covers the status rule — see
    // FaceTicker.  The `setNow` below already re-seeds from the wall clock
    // on every re-arm, so it doubles as the reveal catch-up.
    if (isOccluded) {
      return
    }

    setNow(Date.now())
    const id = setInterval(() => setNow(Date.now()), 1000)

    return () => clearInterval(id)
  }, [isOccluded, startedAt])

  return fmtDuration(now - startedAt)
}

function IdleSince({ endedAt }: { endedAt: number }) {
  // Time since the last final agent response. Re-ticks every second like
  // SessionDuration so the read-out stays live while the session idles.
  const [now, setNow] = useState(() => Date.now())
  const isOccluded = useStore($isStatusRuleOccluded)

  useEffect(() => {
    // Paused only while an overlay actually covers the status rule — see
    // FaceTicker.  The `setNow` below re-seeds from the wall clock on reveal
    // so the idle read-out is not frozen when the overlay closes.
    if (isOccluded) {
      return
    }

    setNow(Date.now())
    const id = setInterval(() => setNow(Date.now()), 1000)

    return () => clearInterval(id)
  }, [endedAt, isOccluded])

  return `✓ ${fmtDuration(now - endedAt)}`
}

// `wire` is the level the route actually sends (session.info.reasoning_effort_wire):
// a clamped Hermes step such as `ultra` reads `ultra→max`, like the CLI's
// "ultra (sends max on this route)", never as a distinct wire level (#61634).
const effortLabel = (effort?: string, wire?: string) => {
  const value = String(effort ?? '')
    .trim()
    .toLowerCase()

  const sent = String(wire ?? '')
    .trim()
    .toLowerCase()

  if (!value || value === 'medium' || value === 'normal' || value === 'default') {
    return ''
  }

  return sent && sent !== value ? `${value}→${sent}` : value
}

const shortModelLabel = (model: string) =>
  model
    .split('/')
    .pop()!
    .replace(/^claude[-_]/, '')
    .replace(/^anthropic[-_]/, '')
    .replace(/[-_]/g, ' ')
    .replace(/\b(\d+)\s+(\d+)\b/g, '$1.$2')
    .trim()

const modelLabel = (model: string, effort?: string, fast?: boolean, effortWire?: string) =>
  [shortModelLabel(model), effortLabel(effort, effortWire), fast ? tr('status.fast') : ''].filter(Boolean).join(' ')

interface StatusRenderItem {
  id: string
  node: ReactNode
  ownRow?: (leadingRule: boolean) => ReactNode
  separator?: boolean
  alignRight?: boolean
  width: number
}

function packStatusRows(items: readonly StatusRenderItem[], cols: number): StatusRenderItem[][] {
  const width = Math.max(1, Math.floor(cols || 1))
  const separatorWidth = stringWidth(' │ ')
  const rows: StatusRenderItem[][] = []
  let row: StatusRenderItem[] = []
  let used = stringWidth('─ ')

  for (const item of items) {
    if (item.ownRow) {
      if (row.length) {
        rows.push(row)
        row = []
      }

      rows.push([item])
      used = 0

      continue
    }

    const itemWidth = Math.min(item.width, width)
    const next = used + (row.length && item.separator !== false ? separatorWidth : 0) + itemWidth

    if (row.length && next > width) {
      rows.push(row)
      row = []
      used = 0
    }

    row.push(item)
    used += (row.length > 1 && item.separator !== false ? separatorWidth : 0) + itemWidth
  }

  if (row.length) {
    rows.push(row)
  }

  return rows
}

function StatusRows({ cols, items, t }: { cols: number; items: readonly StatusRenderItem[]; t: Theme }) {
  const width = Math.max(1, Math.floor(cols || 1))
  const rows = packStatusRows(items, width)

  if (!rows.length) {
    return null
  }

  return (
    <Box flexDirection="column" flexShrink={0} width={width}>
      {rows.map((row, rowIndex) =>
        row.length === 1 && row[0]?.ownRow ? (
          <Fragment key={row[0].id}>{row[0].ownRow(rowIndex === 0)}</Fragment>
        ) : (
          <Box
            flexDirection="row"
            flexShrink={0}
            height={1}
            key={row.map(item => item.id).join(':')}
            overflow="hidden"
            width={width}
          >
            {rowIndex === 0 ? <Text color={t.color.border}>─ </Text> : null}
            {row.map((item, itemIndex) => (
              <Box
                flexDirection="row"
                flexGrow={item.alignRight ? 1 : 0}
                flexShrink={item.width > width ? 1 : 0}
                justifyContent={item.alignRight ? 'flex-end' : undefined}
                key={item.id}
                maxWidth={width}
                overflow="hidden"
              >
                {item.separator !== false && itemIndex ? <Text color={t.color.muted}> │ </Text> : null}
                {item.width > width ? (
                  <Box overflow="hidden" width={width}>
                    {item.node}
                  </Box>
                ) : (
                  item.node
                )}
              </Box>
            ))}
          </Box>
        )
      )}
    </Box>
  )
}

export function GoodVibesHeart({ tick, t }: { tick: number; t: Theme }) {
  const [active, setActive] = useState(false)
  const [color, setColor] = useState(t.color.accent)

  useEffect(() => {
    if (tick <= 0) {
      return
    }

    const palette = [t.color.error, t.color.warn, t.color.accent]
    setColor(palette[Math.floor(Math.random() * palette.length)]!)
    setActive(true)

    const id = setTimeout(() => setActive(false), 650)

    return () => clearTimeout(id)
  }, [t.color.accent, t.color.error, t.color.warn, tick])

  if (!active) {
    return null
  }

  return <Text color={color}>♥</Text>
}

export function StatusRule({
  battery,
  cacheStatus,
  focusView,
  cwdLabel,
  cols,
  busy,
  compacting = false,
  status,
  statusBarFields = null,
  statusBarSegments: configuredStatusBarSegments = null,
  statusColor,
  model,
  modelFast,
  modelReasoningEffort,
  modelReasoningEffortWire,
  indicatorStyle = 'kaomoji',
  notice,
  usage,
  bgCount,
  lastTurnEndedAt,
  liveSessionCount,
  sessionTitle,
  sessionStartedAt,
  turnStartedAt,
  voiceLabel,
  onSessionCountClick,
  t
}: StatusRuleProps) {
  // Not a hook: tests call StatusRule as a plain function, and the rule re-renders on every tick anyway.
  const T = messages()
  const pct = usage.context_percent ?? undefined
  const contextMark = usage.context_estimated ? '~' : ''
  const barColor = ctxBarColor(pct, t)
  const segs = statusBarSegments(cols)

  // display.status_bar.fields visibility gate (same key + names as the
  // classic CLI bar). null = user hasn't customized → everything shows.
  const ok = (name: string) => statusBarFields === null || statusBarFields.has(name)

  // On narrow terminals the context read-out collapses to a bare token count
  // (`12k tok`) and the visual fill bar is dropped entirely.
  const ctxLabel =
    ok('context_detail') || ok('context_pct')
      ? usage.context_max
        ? segs.compactCtx
          ? `${contextMark}${compactNumber(usage.context_used ?? 0)} tok`
          : `${contextMark}${compactNumber(usage.context_used ?? 0)}/${compactNumber(usage.context_max ?? 0)}`
        : (usage.total ?? 0) > 0
          ? `${compactNumber(usage.total)} tok`
          : ''
      : ''

  const bar = !segs.compactCtx && usage.context_max && ok('context_pct') ? ctxBar(pct) : ''
  const modelText = modelLabel(model, modelReasoningEffort, modelFast, modelReasoningEffortWire)

  // Battery read-out — the first (pinned) status-bar element when enabled.
  const showBattery = !!battery && battery.available && battery.percent != null && ok('battery')
  const batteryText = showBattery ? batteryLabel(battery!) : ''
  const batteryColorVal = showBattery ? batteryColor(battery!, t) : ''
  // A credits notice replaces the status/verb slot, but only when idle —
  // while busy the FaceTicker always wins (R1 render priority).
  const showNotice = !busy && !!notice?.text
  const NOTICE_RESERVE_MAX = 24
  const noticeReserve = showNotice ? Math.min(stringWidth(notice!.text), NOTICE_RESERVE_MAX) : 0

  const slotWidth = busy
    ? busyIndicatorWidth(indicatorStyle, turnStartedAt != null)
    : showNotice
      ? noticeReserve
      : stringWidth(status)

  const rightLabel = sessionTitle && ok('title') ? ` ${sessionTitle} ` : cwdLabel

  const sessionCountText = liveSessionCount > 0 ? statusSessionCountLabel(liveSessionCount) : ''
  const compressions = typeof usage.compressions === 'number' ? usage.compressions : 0

  const devCreditsText =
    typeof usage.dev_credits_spent_micros === 'number'
      ? `Δ ${(usage.dev_credits_spent_micros / 10000).toFixed(1)}¢`
      : ''

  const showBar = !!bar
  const showDuration = ok('duration') && !!sessionStartedAt

  const showIdle = ok('idle_since') && !busy && lastTurnEndedAt != null
  const showCompressions = ok('compressions') && compressions > 0

  // Cache-hit % + rolling latency / tokens-per-sec — mirrored from the classic
  // CLI bar (PR #98250). The server omits the keys when no data exists (zero
  // cache reads, Codex app-server with no latency), so these self-hide.
  // The gateway omits this field for zero reads; numeric zero is a positive hit rounded below 1%.
  const cacheHitText =
    cacheStatus ||
    (typeof usage.cache_hit_pct === 'number' ? `◎ ${usage.cache_hit_pct === 0 ? '<1' : usage.cache_hit_pct}%` : '')

  const showCacheHit = ok('cache_hit') && !!cacheHitText
  const latencyText = typeof usage.avg_latency_s === 'number' ? `◷ ${usage.avg_latency_s.toFixed(1)}s` : ''
  const showLatency = ok('latency') && !!latencyText
  const tpsText = typeof usage.avg_tps === 'number' ? `↑ ${Math.round(usage.avg_tps)} t/s` : ''
  const showTps = ok('tps') && !!tpsText
  const showVoice = ok('voice') && !!voiceLabel
  const showSessionCount = ok('sessions') && !!sessionCountText
  const showBg = ok('bg_tasks') && bgCount > 0
  const subagentCount = typeof usage.active_subagents === 'number' ? usage.active_subagents : 0

  const showSubagents = ok('bg_subagents') && subagentCount > 0

  const resumeHintText =
    subagentCount === 1 ? T.status.resumesWhenSubagentFinishes : T.status.resumesWhenSubagentsFinish(subagentCount)

  const showResumeHint = ok('resume') && !busy && subagentCount > 0
  const showDevCredits = ok('dev_credits') && !!devCreditsText
  const showFocus = !!focusView

  const handleSessionCountClick = (event: { stopImmediatePropagation?: () => void }) => {
    event.stopImmediatePropagation?.()
    onSessionCountClick?.()
  }

  const sessionCountNode = onSessionCountClick ? (
    <Box flexShrink={0} onClick={handleSessionCountClick}>
      <Text color={t.color.accent}>{sessionCountText}</Text>
    </Box>
  ) : (
    <Text color={t.color.muted}>{sessionCountText}</Text>
  )

  // The historical `display.tui_statusbar_segments` key controls both
  // visibility and order. Narrow terminals use the same item list, but wrap
  // complete segments onto additional rows instead of silently dropping them.
  if (cols < 72 || configuredStatusBarSegments !== null) {
    if (configuredStatusBarSegments?.length === 0) {
      return null
    }

    const configured = configuredStatusBarSegments === null ? null : new Set(configuredStatusBarSegments)
    const visible = (name: string, modernName = name) => (configured === null ? ok(modernName) : configured.has(name))

    const contextTokens = usage.context_max
      ? `${contextMark}${compactNumber(usage.context_used ?? 0)} tok`
      : (usage.total ?? 0) > 0
        ? `${compactNumber(usage.total ?? 0)} tok`
        : ''

    const contextBar = usage.context_max ? `[${ctxBar(pct)}]` : ''
    const contextPercent = pct == null ? '' : `${contextMark}${pct}%`

    const narrowRightLabel =
      sessionTitle && (configured !== null || ok('title')) ? ` ${sessionTitle} ` : visible('cwd') ? cwdLabel : ''

    const customSessionCountNode = onSessionCountClick ? (
      <Box flexShrink={0} onClick={handleSessionCountClick}>
        <Text color={t.color.accent}>{sessionCountText}</Text>
      </Box>
    ) : (
      <Text color={t.color.muted}>{sessionCountText}</Text>
    )

    const item = (id: string, node: ReactNode, text: number | string): StatusRenderItem => ({
      id,
      node,
      width: typeof text === 'number' ? text : stringWidth(text)
    })

    const available: Record<string, StatusRenderItem | null> = {
      battery:
        visible('battery') && showBattery
          ? item('battery', <Text color={batteryColorVal}>{batteryText}</Text>, batteryText)
          : null,
      indicator:
        visible('indicator') && (busy || showNotice || status)
          ? item(
              'indicator',
              busy ? (
                <FaceTicker
                  color={statusColor}
                  startedAt={turnStartedAt}
                  style={indicatorStyle}
                  verbOverride={compacting ? T.status.compacting : undefined}
                />
              ) : showNotice ? (
                <Text color={noticeColor(notice!.level, t)} wrap="truncate-end">
                  {notice!.text}
                </Text>
              ) : (
                <Text color={statusColor}>{displayStatus(status)}</Text>
              ),
              busy ? slotWidth : showNotice ? stringWidth(notice!.text) : stringWidth(status)
            )
          : null,
      model:
        visible('model') && modelText ? item('model', <Text color={t.color.muted}>{modelText}</Text>, modelText) : null,
      context_tokens:
        (visible('context_tokens', 'context_detail') || (configured === null && ok('context_pct'))) && contextTokens
          ? item('context_tokens', <Text color={t.color.muted}>{contextTokens}</Text>, contextTokens)
          : null,
      context_bar:
        visible('context_bar', 'context_pct') && contextBar
          ? item('context_bar', <Text color={barColor}>{contextBar}</Text>, contextBar)
          : null,
      context_percent:
        visible('context_percent', 'context_pct') && contextPercent
          ? item('context_percent', <Text color={barColor}>{contextPercent}</Text>, contextPercent)
          : null,
      focus: visible('focus') && showFocus ? item('focus', <Text color={t.color.warn}>◉ focus</Text>, '◉ focus') : null,
      session_duration:
        visible('session_duration', 'duration') && sessionStartedAt
          ? {
              id: 'session_duration',
              node: (
                <Text color={t.color.muted}>
                  <SessionDuration startedAt={sessionStartedAt} />
                </Text>
              ),
              width: MAX_DURATION_WIDTH
            }
          : null,
      idle:
        visible('idle', 'idle_since') && !busy && lastTurnEndedAt != null
          ? {
              id: 'idle',
              node: (
                <Text color={t.color.muted}>
                  <IdleSince endedAt={lastTurnEndedAt} />
                </Text>
              ),
              width: stringWidth('✓ ') + MAX_DURATION_WIDTH
            }
          : null,
      compressions:
        visible('compressions') && compressions > 0
          ? item(
              'compressions',
              <Text color={compressions >= 10 ? t.color.error : compressions >= 5 ? t.color.warn : t.color.muted}>
                {T.status.compressions(compressions)}
              </Text>,
              T.status.compressions(compressions)
            )
          : null,
      cache_hit:
        configured === null && ok('cache_hit') && cacheHitText
          ? item(
              'cache_hit',
              <Text
                color={
                  usage.cache_hit_pct! >= 70
                    ? t.color.statusGood
                    : usage.cache_hit_pct! >= 40
                      ? t.color.statusWarn
                      : t.color.muted
                }
              >
                {cacheHitText}
              </Text>,
              cacheHitText
            )
          : null,
      latency:
        configured === null && ok('latency') && latencyText
          ? item('latency', <Text color={t.color.muted}>{latencyText}</Text>, latencyText)
          : null,
      tps:
        configured === null && ok('tps') && tpsText
          ? item('tps', <Text color={t.color.muted}>{tpsText}</Text>, tpsText)
          : null,
      voice:
        visible('voice') && voiceLabel
          ? item(
              'voice',
              <Text
                color={
                  voiceLabel!.startsWith('●')
                    ? t.color.error
                    : voiceLabel!.startsWith('◉')
                      ? t.color.warn
                      : t.color.muted
                }
              >
                {voiceLabel}
              </Text>,
              voiceLabel
            )
          : null,
      sessions:
        visible('sessions') && sessionCountText ? item('sessions', customSessionCountNode, sessionCountText) : null,
      bg_tasks:
        visible('bg_tasks') && bgCount > 0
          ? item(
              'bg_tasks',
              <Text color={t.color.muted}>{T.status.bgTasks(bgCount)}</Text>,
              `${T.status.bgTasks(bgCount)}`
            )
          : null,
      subagents:
        visible('subagents', 'bg_subagents') && subagentCount > 0
          ? item('subagents', <Text color={t.color.muted}>⛓ {subagentCount}</Text>, `⛓ ${subagentCount}`)
          : null,
      resume:
        visible('resume') && !busy && subagentCount > 0
          ? item('resume', <Text color={t.color.muted}>{resumeHintText}</Text>, resumeHintText)
          : null,
      dev_credits:
        visible('dev_credits') && devCreditsText
          ? item('dev_credits', <Text color={t.color.accent}>{devCreditsText}</Text>, devCreditsText)
          : null,
      cwd: narrowRightLabel
        ? item(
            'cwd',
            <Text bold={!!sessionTitle} color={sessionTitle ? t.color.accent : t.color.label} wrap="truncate-end">
              {narrowRightLabel}
            </Text>,
            narrowRightLabel
          )
        : null,
      spawn_hud: visible('spawn_hud')
        ? {
            id: 'spawn_hud',
            node: null,
            ownRow: leadingRule => <SpawnHud cols={Math.max(1, cols - 2)} leadingRule={leadingRule} standalone t={t} />,
            width: cols
          }
        : null
    }

    const defaultOrder = [
      'battery',
      'indicator',
      'model',
      'context_tokens',
      'context_bar',
      'context_percent',
      'focus',
      'session_duration',
      'idle',
      'compressions',
      'cache_hit',
      'latency',
      'tps',
      'voice',
      'sessions',
      'bg_tasks',
      'subagents',
      'resume',
      'dev_credits',
      'cwd',
      'spawn_hud'
    ]

    const ordered = (configuredStatusBarSegments ?? defaultOrder)
      .map(id => available[id])
      .filter((value): value is StatusRenderItem => value !== null && value !== undefined)

    return (
      <Box flexDirection="column" flexShrink={0}>
        {StatusRows({ cols, items: ordered, t })}
      </Box>
    )
  }

  const items: StatusRenderItem[] = []

  const add = (id: string, node: ReactNode, width: number, separator = true, alignRight = false) => {
    if (node != null) {
      items.push({ id, node, width, separator, alignRight })
    }
  }

  add('battery', showBattery ? <Text color={batteryColorVal}>{batteryText}</Text> : null, stringWidth(batteryText))
  add(
    'status',
    busy ? (
      <FaceTicker
        color={statusColor}
        startedAt={turnStartedAt}
        style={indicatorStyle}
        verbOverride={compacting ? T.status.compacting : undefined}
      />
    ) : showNotice ? (
      <Text color={noticeColor(notice!.level, t)} wrap="truncate-end">
        {notice!.text}
      </Text>
    ) : (
      <Text color={statusColor} wrap="truncate-end">
        {displayStatus(status)}
      </Text>
    ),
    busy
      ? busyIndicatorWidth(indicatorStyle, turnStartedAt != null)
      : showNotice
        ? stringWidth(notice!.text)
        : stringWidth(displayStatus(status))
  )
  add(
    'dev-badge',
    DEV_CREDITS_MODE ? <Text color={t.color.warn}>{T.status.devCredits}</Text> : null,
    stringWidth(T.status.devCredits)
  )
  add('model', <Text color={t.color.muted}>{modelText}</Text>, stringWidth(modelText))
  add('context', ctxLabel ? <Text color={t.color.muted}>{ctxLabel}</Text> : null, stringWidth(ctxLabel))
  add('focus', showFocus ? <Text color={t.color.warn}>◉ focus</Text> : null, stringWidth('◉ focus'))
  add(
    'context-bar',
    showBar ? (
      <Text color={t.color.muted}>
        <Text color={barColor}>[{bar}]</Text> <Text color={barColor}>{pct != null ? `${contextMark}${pct}%` : ''}</Text>
      </Text>
    ) : null,
    stringWidth(`[${bar}] ${pct != null ? `${contextMark}${pct}%` : ''}`)
  )
  add(
    'duration',
    showDuration ? (
      <Text color={t.color.muted}>
        <SessionDuration startedAt={sessionStartedAt!} />
      </Text>
    ) : null,
    MAX_DURATION_WIDTH
  )
  add(
    'idle',
    showIdle ? (
      <Text color={t.color.muted}>
        <IdleSince endedAt={lastTurnEndedAt!} />
      </Text>
    ) : null,
    stringWidth('✓ ') + MAX_DURATION_WIDTH
  )
  add(
    'compressions',
    showCompressions ? (
      <Text color={compressions >= 10 ? t.color.error : compressions >= 5 ? t.color.warn : t.color.muted}>
        {T.status.compressions(compressions)}
      </Text>
    ) : null,
    stringWidth(T.status.compressions(compressions))
  )
  add(
    'cache-hit',
    showCacheHit ? (
      <Text
        color={
          usage.cache_hit_pct! >= 70
            ? t.color.statusGood
            : usage.cache_hit_pct! >= 40
              ? t.color.statusWarn
              : t.color.muted
        }
      >
        {cacheHitText}
      </Text>
    ) : null,
    stringWidth(cacheHitText)
  )
  add('latency', showLatency ? <Text color={t.color.muted}>{latencyText}</Text> : null, stringWidth(latencyText))
  add('tps', showTps ? <Text color={t.color.muted}>{tpsText}</Text> : null, stringWidth(tpsText))
  add(
    'voice',
    showVoice ? (
      <Text
        color={voiceLabel!.startsWith('●') ? t.color.error : voiceLabel!.startsWith('◉') ? t.color.warn : t.color.muted}
      >
        {voiceLabel}
      </Text>
    ) : null,
    stringWidth(voiceLabel ?? '')
  )
  add('session-count', showSessionCount ? sessionCountNode : null, stringWidth(sessionCountText))
  add(
    'bg',
    showBg ? <Text color={t.color.muted}>{T.status.bgTasks(bgCount)}</Text> : null,
    stringWidth(T.status.bgTasks(bgCount))
  )
  add(
    'subagents',
    showSubagents ? <Text color={t.color.muted}>⛓ {subagentCount}</Text> : null,
    stringWidth(`⛓ ${subagentCount}`)
  )
  add(
    'resume-hint',
    showResumeHint ? (
      <Text color={t.color.muted} dim>
        {resumeHintText}
      </Text>
    ) : null,
    stringWidth(resumeHintText)
  )
  add(
    'dev-credits',
    showDevCredits ? <Text color={t.color.accent}>{devCreditsText}</Text> : null,
    stringWidth(devCreditsText)
  )
  // SpawnHud supplies its own divider and clips itself after the budgeted fields.
  add('spawn-hud', <SpawnHud t={t} />, 0, false)
  add(
    'title',
    rightLabel ? (
      <Text bold={!!sessionTitle} color={sessionTitle ? t.color.accent : t.color.label} wrap="truncate-end">
        {' ─ '}
        {rightLabel}
      </Text>
    ) : null,
    stringWidth(` ─ ${rightLabel}`),
    false,
    true
  )

  // StatusRulePane sits inside NoSelect's 1-cell horizontal padding. Pack to
  // the actual content width so the right-aligned title doesn't clip on the edge.
  return StatusRows({ cols: Math.max(1, cols - 2), items, t })
}

export function FloatBox({ children, color }: { children: ReactNode; color: string }) {
  return (
    <Box
      alignSelf="flex-start"
      borderColor={color}
      borderStyle="double"
      flexDirection="column"
      marginTop={1}
      opaque
      paddingX={1}
    >
      {children}
    </Box>
  )
}

export function StickyPromptTracker({ messages, offsets, scrollRef, onChange }: StickyPromptTrackerProps) {
  const { atBottom, bottom, top } = useViewportSnapshot(scrollRef)
  const text = stickyPromptFromViewport(messages, offsets, top, bottom, atBottom)

  useEffect(() => onChange(text), [onChange, text])

  return null
}

export function TranscriptScrollbar({ scrollRef, t }: TranscriptScrollbarProps) {
  const [hover, setHover] = useState(false)
  const [grab, setGrab] = useState<number | null>(null)
  const grabRef = useRef<number | null>(null)
  const { scrollHeight: total, top: pos, viewportHeight: vp } = useScrollbarSnapshot(scrollRef)

  if (!vp) {
    return <Box width={1} />
  }

  const s = scrollRef.current
  const scrollable = total > vp
  const thumb = scrollable ? Math.max(1, Math.round((vp * vp) / total)) : vp
  const travel = Math.max(1, vp - thumb)
  const thumbTop = scrollable ? Math.round((pos / Math.max(1, total - vp)) * travel) : 0
  const { thumb: thumbColor, track: trackColor } = scrollbarColors(t, hover, grab !== null)

  const jump = (row: number, offset: number) => {
    if (!s || !scrollable) {
      return
    }

    s.scrollTo(Math.round((Math.max(0, Math.min(travel, row - offset)) / travel) * Math.max(0, total - vp)))
  }

  return (
    <Box
      flexDirection="column"
      onMouseDown={(e: { localRow?: number }) => {
        const row = Math.max(0, Math.min(vp - 1, e.localRow ?? 0))
        const off = row >= thumbTop && row < thumbTop + thumb ? row - thumbTop : Math.floor(thumb / 2)

        grabRef.current = off
        setGrab(off)
        jump(row, off)
      }}
      onMouseDrag={(e: { localRow?: number }) =>
        jump(Math.max(0, Math.min(vp - 1, e.localRow ?? 0)), grabRef.current ?? Math.floor(thumb / 2))
      }
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onMouseUp={() => {
        grabRef.current = null
        setGrab(null)
      }}
      width={1}
    >
      {/* Nothing to scroll → draw nothing (the width={1} Box still reserves
          the column). Drawn-blank cells composite to a black bar on
          transparent terminals — same class as the removed opaque fills. */}
      {!scrollable ? null : (
        <>
          {thumbTop > 0 ? (
            <Text color={trackColor}>{`${'│\n'.repeat(Math.max(0, thumbTop - 1))}${thumbTop > 0 ? '│' : ''}`}</Text>
          ) : null}
          {thumb > 0 ? (
            <Text color={thumbColor}>{`${'┃\n'.repeat(Math.max(0, thumb - 1))}${thumb > 0 ? '┃' : ''}`}</Text>
          ) : null}
          {vp - thumbTop - thumb > 0 ? (
            <Text
              color={trackColor}
            >{`${'│\n'.repeat(Math.max(0, vp - thumbTop - thumb - 1))}${vp - thumbTop - thumb > 0 ? '│' : ''}`}</Text>
          ) : null}
        </>
      )}
    </Box>
  )
}

interface StatusRuleProps {
  battery?: BatteryInfo | null
  cacheStatus?: null | string
  // Focus view (/focus) badge — display-only reduced-output indicator.
  focusView?: boolean
  bgCount: number
  lastTurnEndedAt?: null | number
  liveSessionCount: number
  busy: boolean
  // Context compaction in progress — FaceTicker freezes on "compacting".
  compacting?: boolean
  cols: number
  cwdLabel: string
  model: string
  modelFast?: boolean
  modelReasoningEffort?: string
  modelReasoningEffortWire?: string
  indicatorStyle?: IndicatorStyle
  notice?: Notice | null
  sessionStartedAt?: null | number
  sessionTitle?: string
  status: string
  // display.status_bar.fields — segment visibility filter shared with the
  // classic CLI bar. null = defaults (everything shows).
  statusBarFields?: null | ReadonlySet<string>
  statusBarSegments?: null | readonly string[]
  statusColor: string
  t: Theme
  turnStartedAt?: null | number
  usage: Usage
  voiceLabel?: string
  onSessionCountClick?: () => void
}

interface StickyPromptTrackerProps {
  messages: readonly Msg[]
  offsets: ArrayLike<number>
  onChange: (text: string) => void
  scrollRef: RefObject<ScrollBoxHandle | null>
}

interface TranscriptScrollbarProps {
  scrollRef: RefObject<ScrollBoxHandle | null>
  t: Theme
}
