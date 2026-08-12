import { PassThrough } from 'stream'

import { AlternateScreen, Box, forceRedraw, renderSync, stringWidth, Text } from '@hermes/ink'
import React, { Fragment } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { patchDelegationState, resetDelegationState } from '../app/delegationStore.js'
import { StatusRule } from '../components/appChrome.js'
import { DEFAULT_STATUS_BAR_SEGMENTS } from '../lib/statusBar.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

type StatusRuleProps = React.ComponentProps<typeof StatusRule>

const baseProps: StatusRuleProps = {
  bgCount: 0,
  busy: false,
  cols: 44,
  cwdLabel: '~/repo',
  liveSessionCount: 0,
  model: 'qwen-27b',
  sessionStartedAt: null,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: {
    calls: 0,
    context_max: 128_000,
    context_percent: 25,
    context_used: 32_000,
    input: 0,
    output: 0,
    total: 32_000
  },
  voiceLabel: ''
}

const richProps: Partial<StatusRuleProps> = {
  battery: { available: true, category: 'good', percent: 82, plugged: true },
  bgCount: 2,
  cwdLabel: '目录/分支🌟',
  focusView: true,
  liveSessionCount: 3,
  model: '模型/qwen-长',
  usage: {
    active_subagents: 2,
    calls: 0,
    compressions: 3,
    context_max: 128_000,
    context_percent: 25,
    context_used: 32_000,
    dev_credits_spent_micros: 12_345,
    input: 0,
    output: 0,
    total: 32_000
  },
  voiceLabel: 'voice off'
}

const frameLines = (output: string) =>
  stripAnsi(output)
    .split('\n')
    .map(line => line.trimEnd())
    .filter(Boolean)

const flush = () => new Promise(resolve => setTimeout(resolve, 20))

const mount = (node: React.ReactElement, columns: number) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(node, {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  return {
    cleanup: () => {
      instance.unmount()
      instance.cleanup()
    },
    lines: () => frameLines(output),
    rerender: async (next: React.ReactElement, nextColumns: number) => {
      output = ''
      Object.assign(stdout, { columns: nextColumns })
      instance.rerender(next)
      await flush()
      forceRedraw(stdout as unknown as NodeJS.WriteStream)
      await flush()

      return frameLines(output)
    },
    stdout: stdout as unknown as NodeJS.WriteStream
  }
}

const status = (cols: number, props: Partial<StatusRuleProps> = {}) => (
  <StatusRule {...baseProps} cols={cols} {...props} />
)

const segmentFixture: Partial<StatusRuleProps> = {
  ...richProps,
  lastTurnEndedAt: Date.now() - 5_000,
  sessionStartedAt: Date.now() - 60_000,
  usage: {
    ...richProps.usage!,
    runtime_heartbeat: {
      active_count: 1,
      targets: [{ interval_s: 1700, kind: 'process', last_success_at: null, started_at: Date.now() / 1000 - 42 }]
    }
  }
}

describe('StatusRule responsive Ink layout', () => {
  it.each([
    { cols: 12, rows: 'multiple' },
    { cols: 44, rows: 'multiple' },
    { cols: 71, rows: 'multiple' },
    { cols: 72, rows: 'single' },
    { cols: 160, rows: 'single' }
  ] as const)('uses $rows natural-height rows at $cols columns', async ({ cols, rows }) => {
    const mounted = mount(status(cols), cols)

    try {
      await flush()
      expect(mounted.lines().length === 1 ? 'single' : 'multiple').toBe(rows)
    } finally {
      mounted.cleanup()
    }
  })

  it('keeps every present narrow segment instead of dropping pinned and tail fields', async () => {
    const mounted = mount(status(44, richProps), 44)

    try {
      await flush()
      const output = mounted.lines().join('\n')

      expect(output).toContain('⚡ 82%')
      expect(output).toContain('ready')
      expect(output).toContain('qwen 长')
      expect(output).toContain('32k tok')
      expect(output).toContain('◉ focus')
      expect(output).toContain('cmp 3')
      expect(output).toContain('voice off')
      expect(output).toContain('3 sessions')
      expect(output).toContain('2 bg')
      expect(output).toContain('⛓ 2')
      expect(output).toContain('resumes when 2 subagents finish')
      expect(output).toContain('Δ 1.2¢')
      expect(output).toContain('目录/分支🌟')
      expect(mounted.lines().length).toBeGreaterThan(1)
    } finally {
      mounted.cleanup()
    }
  })

  it('packs Unicode labels by terminal display width at tiny widths', async () => {
    const mounted = mount(
      status(12, {
        cwdLabel: '',
        model: '界界界界',
        status: '界界界界',
        usage: { calls: 0, input: 0, output: 0, total: 0 }
      }),
      12
    )

    try {
      await flush()
      const lines = mounted.lines()

      expect(lines).toHaveLength(2)

      for (const line of lines) {
        expect(stringWidth(line)).toBeLessThanOrEqual(12)
      }
    } finally {
      mounted.cleanup()
    }
  })

  it('ellipsizes over-wide fallback segments at the Ink boundary', async () => {
    const contextMeter = mount(
      status(12, {
        cwdLabel: '',
        model: '',
        status: '',
        statusBarSegments: ['context_bar', 'context_percent'],
        usage: {
          context_max: 128_000,
          context_percent: 25,
          context_used: 32_000,
          calls: 0,
          input: 0,
          output: 0,
          total: 0
        }
      }),
      12
    )

    const model = mount(
      status(12, {
        cwdLabel: '',
        model: 'provider/over-wide-model-name',
        status: '',
        statusBarSegments: ['model'],
        usage: { calls: 0, input: 0, output: 0, total: 0 }
      }),
      12
    )

    try {
      await flush()
      const contextOutput = contextMeter.lines().join('\n')
      const modelOutput = model.lines().join('\n')

      expect(contextOutput).toContain('[███')
      expect(contextOutput).toContain('…')
      expect(modelOutput).toContain('over wide…')
      expect(contextMeter.lines().every(line => stringWidth(line) <= 12)).toBe(true)
      expect(model.lines().every(line => stringWidth(line) <= 12)).toBe(true)
    } finally {
      contextMeter.cleanup()
      model.cleanup()
    }
  })

  it('ellipsizes an over-wide clickable session count without losing its click path', async () => {
    const openSwitcher = vi.fn()
    const props: Partial<StatusRuleProps> = {
      cwdLabel: '',
      liveSessionCount: 10,
      model: '',
      onSessionCountClick: openSwitcher,
      status: '',
      statusBarSegments: ['sessions'],
      usage: { calls: 0, input: 0, output: 0, total: 0 }
    }
    const mounted = mount(
      <AlternateScreen>
        {status(12, props)}
      </AlternateScreen>,
      12
    )

    try {
      await flush()
      const output = mounted.lines().join('\n')

      expect(output).toContain('10 sessio')
      expect(output).toContain('…')
      expect(mounted.lines().every(line => stringWidth(line) <= 12)).toBe(true)

      // StatusRule returns <StatusRows items=…/>; click lives on sessions.narrowNode, not element children.
      const root = StatusRule({ ...baseProps, cols: 12, ...props }) as React.ReactElement<{
        items: Array<{ id: string; narrowNode?: (width: number) => React.ReactNode; node: React.ReactNode; width: number }>
      }>
      const sessions = root.props.items.find(item => item.id === 'sessions')
      expect(sessions?.narrowNode).toEqual(expect.any(Function))
      const narrow = sessions!.narrowNode!(10)
      const onClick = (narrow as React.ReactElement<{ onClick?: (event: { stopImmediatePropagation?: () => void }) => void }>).props
        .onClick
      expect(onClick).toEqual(expect.any(Function))
      onClick!({ stopImmediatePropagation: vi.fn() })
      expect(openSwitcher).toHaveBeenCalledOnce()
    } finally {
      mounted.cleanup()
    }
  })

  it('bounds over-wide resume and Unicode cwd segments instead of clipping them', async () => {
    const resume = mount(
      status(12, {
        cwdLabel: '',
        model: '',
        status: '',
        statusBarSegments: ['resume'],
        usage: { active_subagents: 2, calls: 0, input: 0, output: 0, total: 0 }
      }),
      12
    )

    const cwd = mount(
      status(12, {
        cwdLabel: '目录/分支🌟長い作業ディレクトリ',
        model: '',
        status: '',
        statusBarSegments: ['cwd'],
        usage: { calls: 0, input: 0, output: 0, total: 0 }
      }),
      12
    )

    try {
      await flush()
      const resumeOutput = resume.lines().join('\n')
      const cwdOutput = cwd.lines().join('\n')

      expect(resumeOutput).toContain('↩ resumes…')
      expect(cwdOutput).toContain('目录/分支…')
      expect(resume.lines().every(line => stringWidth(line) <= 12)).toBe(true)
      expect(cwd.lines().every(line => stringWidth(line) <= 12)).toBe(true)
    } finally {
      resume.cleanup()
      cwd.cleanup()
    }
  })

  it('keeps bounded rows natural after resize', async () => {
    const props: Partial<StatusRuleProps> = {
      cwdLabel: '目录/分支🌟長い作業ディレクトリ',
      model: '',
      status: '',
      statusBarSegments: ['resume', 'cwd'],
      usage: { active_subagents: 2, calls: 0, input: 0, output: 0, total: 0 }
    }

    const mounted = mount(status(12, props), 12)

    try {
      await flush()
      expect(mounted.lines().every(line => stringWidth(line) <= 12)).toBe(true)

      const wider = await mounted.rerender(status(44, props), 44)

      expect(wider).toHaveLength(2)
      expect(wider.every(line => stringWidth(line) <= 44)).toBe(true)
    } finally {
      mounted.cleanup()
    }
  })

  it('reflows across 71 → 72 → tiny resize transitions', async () => {
    const mounted = mount(status(71), 71)

    try {
      await flush()
      expect(mounted.lines().length).toBeGreaterThan(1)
      expect(await mounted.rerender(status(72), 72)).toHaveLength(1)

      const tiny = await mounted.rerender(status(12), 12)

      expect(tiny.length).toBeGreaterThan(1)
      expect(tiny.every(line => stringWidth(line) <= 12)).toBe(true)
    } finally {
      mounted.cleanup()
    }
  })
})

describe('StatusRule configured segment visibility', () => {
  it('keeps default wide progressive disclosure for a hydrated default list', async () => {
    const mounted = mount(
      status(72, {
        cwdLabel: '',
        model: '',
        status: '',
        statusBarSegments: [...DEFAULT_STATUS_BAR_SEGMENTS],
        usage: { calls: 0, input: 0, output: 0, total: 0 },
        voiceLabel: 'voice off'
      }),
      72
    )

    try {
      await flush()
      expect(mounted.lines()).toEqual([])
    } finally {
      mounted.cleanup()
    }
  })

  it('renders no blank status row when every segment is hidden', async () => {
    const mounted = mount(status(44, { ...segmentFixture, statusBarSegments: [] }), 44)

    try {
      await flush()
      expect(mounted.lines()).toEqual([])
    } finally {
      mounted.cleanup()
    }
  })

  it.each([
    ['battery', '⚡ 82%'],
    ['indicator', 'ready'],
    ['model', 'qwen 长'],
    ['context_tokens', '32k/128k'],
    ['context_bar', '[███░░░░░░░]'],
    ['context_percent', '25%'],
    ['cwd', '目录/分支🌟'],
    ['focus', '◉ focus'],
    ['heartbeat', '→checkin'],
    ['idle', '✓ '],
    ['session_duration', '1m '],
    ['compressions', 'cmp 3'],
    ['voice', 'voice off'],
    ['sessions', '3 sessions'],
    ['bg_tasks', '2 bg'],
    ['subagents', '⛓ 2'],
    ['resume', 'resumes when 2 subagents finish'],
    ['dev_credits', 'Δ 1.2¢']
  ] as const)('can hide %s while keeping another configured segment', async (hidden, needle) => {
    const keeper = hidden === 'indicator' ? 'model' : 'indicator'
    const mounted = mount(status(160, { ...segmentFixture, statusBarSegments: [keeper, hidden] }), 160)

    try {
      await flush()
      const withField = mounted.lines().join('\n')

      expect(withField).toContain(needle)

      const hidden = mount(status(160, { ...segmentFixture, statusBarSegments: [keeper] }), 160)

      try {
        await flush()
        expect(hidden.lines().join('\n')).not.toContain(needle)
      } finally {
        hidden.cleanup()
      }
    } finally {
      mounted.cleanup()
    }
  })

  it('can hide the live spawn HUD', async () => {
    patchDelegationState({ paused: true })
    const mounted = mount(status(160, { statusBarSegments: ['indicator', 'spawn_hud'] }), 160)

    try {
      await flush()
      expect(mounted.lines().join('\n')).toContain('⏸ paused')

      const hidden = mount(status(160, { statusBarSegments: ['indicator'] }), 160)

      try {
        await flush()
        expect(hidden.lines().join('\n')).not.toContain('⏸ paused')
      } finally {
        hidden.cleanup()
      }
    } finally {
      mounted.cleanup()
      resetDelegationState()
    }
  })

  it('accounts for filtering before the 72-column tail breakpoints', async () => {
    const mounted = mount(status(72, { ...segmentFixture, statusBarSegments: ['voice'] }), 72)

    try {
      await flush()
      expect(mounted.lines()).toEqual(['─ voice off'])
    } finally {
      mounted.cleanup()
    }
  })

  it('keeps wide custom fields on one row and drops overflow by whole segment', async () => {
    const mounted = mount(
      status(72, {
        ...segmentFixture,
        statusBarSegments: [
          'indicator',
          'model',
          'context_tokens',
          'voice',
          'sessions',
          'bg_tasks',
          'subagents',
          'resume'
        ]
      }),
      72
    )

    try {
      await flush()
      const lines = mounted.lines()

      expect(lines).toHaveLength(1)
      expect(stringWidth(lines[0]!)).toBeLessThanOrEqual(72)
      expect(lines[0]).not.toContain('↩')
    } finally {
      mounted.cleanup()
    }
  })

  it('shows an explicitly selected context meter on a narrow row', async () => {
    const mounted = mount(status(44, { ...segmentFixture, statusBarSegments: ['context_bar', 'context_percent'] }), 44)

    try {
      await flush()
      expect(mounted.lines()).toEqual(['─ [███░░░░░░░] 25%'])
    } finally {
      mounted.cleanup()
    }
  })

  it('filters absent leading segments before separators and width accounting', async () => {
    const mounted = mount(status(120, { statusBarSegments: ['context_percent'] }), 120)

    try {
      await flush()
      const output = mounted.lines().join('\n')

      expect(output).toContain('25%')
      expect(output).not.toContain('│')
    } finally {
      mounted.cleanup()
    }
  })

  it('renders no phantom row when the only configured dynamic segment is absent', async () => {
    const mounted = mount(status(120, { statusBarSegments: ['spawn_hud'] }), 120)

    try {
      await flush()
      expect(mounted.lines()).toEqual([])
    } finally {
      mounted.cleanup()
    }
  })

  it('filters null nodes before separators so optional HUDs leave no phantom divider', async () => {
    const mounted = mount(status(120, { statusBarSegments: ['spawn_hud', 'cwd'] }), 120)

    try {
      await flush()
      const output = mounted.lines().join('\n')

      expect(output).toContain('~/repo')
      expect(output).not.toContain('│ │')
      expect(output.trimStart()).not.toMatch(/^│/)
    } finally {
      mounted.cleanup()
    }
  })
})

describe.each([
  { inline: false, name: 'alternate-screen' },
  { inline: true, name: 'inline' },
  { inline: true, name: 'Termux inline default' }
])('StatusRule reservation in $name mode', ({ inline }) => {
  it.each(['top', 'bottom'] as const)('reserves all natural-height rows at the %s of the composer', async at => {
    const Shell = inline ? Fragment : AlternateScreen
    const rule = status(44)

    const tree = (
      <Shell>
        <Box flexDirection="column" height={6} width={44}>
          <Box flexGrow={1} flexShrink={1}>
            <Text>TRANSCRIPT</Text>
          </Box>
          <Box flexDirection="column" flexShrink={0}>
            {at === 'top' ? rule : null}
            <Text>COMPOSER</Text>
            {at === 'bottom' ? rule : null}
          </Box>
        </Box>
      </Shell>
    )

    const mounted = mount(tree, 44)

    try {
      await flush()
      const lines = mounted.lines()
      const transcript = lines.findIndex(line => line.includes('TRANSCRIPT'))
      const indicator = lines.findIndex(line => line.includes('ready'))
      const model = lines.findIndex(line => line.includes('qwen 27b'))
      const composer = lines.findIndex(line => line.includes('COMPOSER'))

      expect(transcript).toBeGreaterThanOrEqual(0)
      expect(indicator).toBeGreaterThan(transcript)
      expect(model).toBeGreaterThan(indicator)

      if (at === 'top') {
        expect(composer).toBeGreaterThan(model)
      } else {
        expect(composer).toBeLessThan(indicator)
      }
    } finally {
      mounted.cleanup()
    }
  })
})
