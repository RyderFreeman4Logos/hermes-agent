import { PassThrough } from 'stream'

import { renderSync, stringWidth } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const flush = () => new Promise(resolve => setTimeout(resolve, 20))

describe('status-rule narrow reflow', () => {
  it('keeps every present field in natural-height bounded rows', async () => {
    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdout, { columns: 44, isTTY: false, rows: 24 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const instance = renderSync(
      <StatusRule
        bgCount={2}
        busy={false}
        cols={44}
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
      />,
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    try {
      await flush()

      const lines = stripAnsi(output)
        .split('\n')
        .map(line => line.trimEnd())
        .filter(Boolean)

      const rendered = lines.join('\n')

      expect(lines.length).toBeGreaterThan(1)
      expect(lines.every(line => stringWidth(line) <= 44)).toBe(true)

      for (const needle of ['ready', 'qwen 长', '32k tok', 'cmp 3', 'voice off', '3 sessions', '2 bg', '⛓ 2']) {
        expect(rendered).toContain(needle)
      }
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })
})
