import { PassThrough } from 'stream'

import { renderSync, stringWidth } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import { DEFAULT_THEME } from '../theme.js'

const renderRule = (cols: number, props: Partial<React.ComponentProps<typeof StatusRule>> = {}) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: cols, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <StatusRule
      bgCount={2}
      busy={false}
      cols={cols}
      cwdLabel="目录/分支🌟"
      liveSessionCount={3}
      model="模型/qwen-长"
      sessionStartedAt={Date.now() - 60_000}
      status="ready"
      statusColor={DEFAULT_THEME.color.ok}
      t={DEFAULT_THEME}
      turnStartedAt={null}
      usage={{
        active_subagents: 2,
        compressions: 3,
        context_max: 128_000,
        context_percent: 25,
        context_used: 32_000,
        total: 32_000
      }}
      voiceLabel="voice off"
      {...props}
    />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
    .split('\n')
    .map(line => line.trimEnd())
    .filter(Boolean)
}

const NEEDLES = ['ready', 'qwen 长', '32k tok', 'cmp 3', 'voice off', '3 sessions', '2 bg', '⛓ 2']

describe('status-rule reflow', () => {
  it('wraps every present field onto bounded rows when the terminal is narrow', () => {
    const lines = renderRule(44)
    const rendered = lines.join('\n')

    expect(lines.length).toBeGreaterThan(1)
    expect(lines.every(line => stringWidth(line) <= 44)).toBe(true)

    for (const needle of NEEDLES) {
      expect(rendered, needle).toContain(needle)
    }

    expect(rendered.match(/qwen 长/g)?.length).toBe(1)
  })

  it('keeps the same fields on one row when the terminal is wide', () => {
    const lines = renderRule(200)
    const rendered = lines.join('\n')

    expect(lines).toHaveLength(1)

    for (const needle of ['ready', 'qwen 长', '32k/128k', 'cmp 3', 'voice off', '3 sessions', '2 bg', '⛓ 2']) {
      expect(rendered, needle).toContain(needle)
    }
  })

  it('still hides a field the user turned off', () => {
    const rendered = renderRule(160, { statusBarFields: new Set(['model']) }).join('\n')

    expect(rendered).toContain('qwen 长')
    expect(rendered).not.toContain('voice off')
    expect(rendered).not.toContain('2 bg')
  })
})
