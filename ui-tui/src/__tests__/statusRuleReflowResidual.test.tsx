import { PassThrough } from 'stream'

import { renderSync, stringWidth } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { patchDelegationState, resetDelegationState } from '../app/delegationStore.js'
import { normalizeLegacyStatusBarSegments } from '../app/useConfigSync.js'
import { StatusRule } from '../components/appChrome.js'
import { stripAnsi } from '@hermes/shared/ansi'
import { DEFAULT_THEME } from '../theme.js'

const flush = () => new Promise(resolve => setTimeout(resolve, 20))

const renderSegments = async (segments: readonly string[]) => {
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
      bgCount={0}
      busy={false}
      cols={44}
      cwdLabel="~/repo"
      liveSessionCount={0}
      model="qwen"
      sessionStartedAt={null}
      status="ready"
      statusBarSegments={[...segments]}
      statusColor={DEFAULT_THEME.color.ok}
      t={DEFAULT_THEME}
      turnStartedAt={null}
      usage={{ context_max: 0, context_percent: 0, context_used: 0, total: 0 }}
      voiceLabel=""
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
    return stripAnsi(output)
  } finally {
    instance.unmount()
    instance.cleanup()
  }
}

describe('status-rule narrow reflow', () => {
  it('does not reserve a physical row for an inactive spawn HUD', async () => {
    resetDelegationState()
    const withoutHud = await renderSegments(['model'])
    const withInactiveHud = await renderSegments(['model', 'spawn_hud'])

    expect(withInactiveHud).toBe(withoutHud)
  })

  it('falls back to renderable defaults for a heartbeat-only legacy config', async () => {
    const configured = normalizeLegacyStatusBarSegments(['heartbeat'])
    const rendered = await renderSegments(configured ?? [])

    expect(rendered).toContain('qwen')
  })

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

  it.each([
    [44, ['spawn_hud', 'model'], 'paused', 'qwen'],
    [44, ['model', 'spawn_hud'], 'qwen', 'paused'],
    [120, ['spawn_hud', 'model'], 'paused', 'qwen'],
    [120, ['model', 'spawn_hud'], 'qwen', 'paused']
  ] as const)('honors spawn HUD legacy order at %i columns', async (cols, segments, first, second) => {
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
    patchDelegationState({ paused: true })

    const instance = renderSync(
      <StatusRule
        bgCount={0}
        busy={false}
        cols={cols}
        cwdLabel="~/repo"
        liveSessionCount={0}
        model="qwen"
        sessionStartedAt={null}
        status="ready"
        statusBarSegments={[...segments]}
        statusColor={DEFAULT_THEME.color.ok}
        t={DEFAULT_THEME}
        turnStartedAt={null}
        usage={{ context_max: 0, context_percent: 0, context_used: 0, total: 0 }}
        voiceLabel=""
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
      const rendered = stripAnsi(output)
      expect(rendered.match(/paused/g)).toHaveLength(1)
      expect(rendered.indexOf(first)).toBeLessThan(rendered.indexOf(second))
      expect(
        rendered
          .split('\n')
          .filter(Boolean)
          .every(line => stringWidth(line) <= cols)
      ).toBe(true)
    } finally {
      instance.unmount()
      instance.cleanup()
      resetDelegationState()
    }
  })
})
