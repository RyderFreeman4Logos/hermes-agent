import { PassThrough } from 'stream'

import { renderSync, stringWidth } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

type StatusRuleProps = React.ComponentProps<typeof StatusRule>

const baseProps: StatusRuleProps = {
  bgCount: 2,
  busy: false,
  cols: 44,
  cwdLabel: '目录/分支🌟',
  liveSessionCount: 3,
  model: '模型/qwen-长',
  sessionStartedAt: Date.now() - 60_000,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: {
    active_subagents: 2,
    calls: 0,
    compressions: 3,
    context_max: 128_000,
    context_percent: 25,
    context_used: 32_000,
    input: 0,
    output: 0,
    total: 32_000
  },
  voiceLabel: 'voice off'
}

const flush = () => new Promise(resolve => setTimeout(resolve, 20))

const mount = (props: StatusRuleProps, columns: number) => {
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
  const instance = renderSync(<StatusRule {...props} />, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  return {
    cleanup: () => {
      instance.unmount()
      instance.cleanup()
    },
    lines: () =>
      stripAnsi(output)
        .split('\n')
        .map(line => line.trimEnd())
        .filter(Boolean)
  }
}

describe('retained status-rule reflow', () => {
  it('keeps every present narrow field in natural-height bounded rows', async () => {
    const mounted = mount(baseProps, 44)

    try {
      await flush()
      const lines = mounted.lines()
      const output = lines.join('\n')

      expect(lines.length).toBeGreaterThan(1)
      expect(lines.every(line => stringWidth(line) <= 44)).toBe(true)
      for (const needle of ['ready', 'qwen 长', '32k tok', 'cmp 3', 'voice off', '3 sessions', '2 bg', '⛓ 2']) {
        expect(output).toContain(needle)
      }
    } finally {
      mounted.cleanup()
    }
  })

  it('bounds an over-wide clickable session count without losing its click path', async () => {
    const onSessionCountClick = vi.fn()
    const props: StatusRuleProps = {
      ...baseProps,
      bgCount: 0,
      cols: 12,
      cwdLabel: '',
      liveSessionCount: 10,
      model: '',
      onSessionCountClick,
      status: '',
      statusBarSegments: ['sessions'],
      usage: { calls: 0, input: 0, output: 0, total: 0 },
      voiceLabel: ''
    }
    const mounted = mount(props, 12)

    try {
      await flush()
      expect(mounted.lines().every(line => stringWidth(line) <= 12)).toBe(true)

      const root = StatusRule(props) as React.ReactElement<{
        items: Array<{ id: string; narrowNode?: (width: number) => React.ReactNode }>
      }>
      const sessions = root.props.items.find(item => item.id === 'sessions')
      const narrow = sessions!.narrowNode!(10) as React.ReactElement<{
        onClick?: (event: { stopImmediatePropagation?: () => void }) => void
      }>

      narrow.props.onClick!({ stopImmediatePropagation: vi.fn() })
      expect(onSessionCountClick).toHaveBeenCalledOnce()
    } finally {
      mounted.cleanup()
    }
  })
})
