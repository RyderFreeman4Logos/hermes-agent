import { describe, expect, it } from 'vitest'

import { normalizeStatusBarSegments, packStatusRows } from '../lib/statusBar.js'

describe('status bar segment configuration', () => {
  it('normalizes known IDs and expands legacy context', () => {
    expect(normalizeStatusBarSegments([' MODEL ', 'context', 'context_percent', 'cwd', 'model'])).toEqual([
      'model',
      'context_tokens',
      'context_bar',
      'context_percent',
      'cwd'
    ])
    expect(normalizeStatusBarSegments([])).toEqual([])
    expect(normalizeStatusBarSegments(['unknown'])).toEqual(expect.arrayContaining(['indicator', 'model', 'cwd']))
  })

  it('packs visible segments into bounded rows', () => {
    expect(packStatusRows([{ width: 4 }, { width: 5 }, { breakBefore: true, width: 2 }], 12, 3, 2)).toEqual([
      [{ width: 4 }],
      [{ width: 5 }],
      [{ breakBefore: true, width: 2 }]
    ])
  })
})
