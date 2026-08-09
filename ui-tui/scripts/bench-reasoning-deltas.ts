// Controller-only reasoning stream benchmark. React/Ink rendering is deliberately disabled.
// Run the unchanged file in baseline and candidate checkouts, for example:
//   npx tsx scripts/bench-reasoning-deltas.ts --label=baseline@<sha>
//   npx tsx scripts/bench-reasoning-deltas.ts --label=candidate@<sha>

import os from 'node:os'

import * as controllerModule from '../src/app/turnController.js'
import { $turnState, getTurnState } from '../src/app/turnStore.js'
import { patchUiState, resetUiState } from '../src/app/uiStore.js'
import { STREAM_BATCH_MS } from '../src/config/timing.js'

const controllerExports = controllerModule as unknown as Record<string, unknown>
const { turnController } = controllerModule

const args = new Map(
  process.argv.slice(2).map(arg => {
    const [key, value = ''] = arg.replace(/^--/, '').split('=', 2)

    return [key, value]
  })
)

const numberArg = (key: string, fallback: number) => {
  const value = Number(args.get(key) ?? fallback)

  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`--${key} must be a positive integer`)
  }

  return value
}

const deltaCount = numberArg('deltas', 20_000)
const chunkBytes = numberArg('chunk-bytes', 10)
const samples = numberArg('samples', 3)
const warmups = numberArg('warmups', 1)
const label = args.get('label') || 'working-tree'
const chunk = 'r'.repeat(chunkBytes)
const expected = chunk.repeat(deltaCount)

interface Sample {
  enqueueMs: number
  finalExact: boolean
  publications: number
  settledMs: number
}

const runSample = async (): Promise<Sample> => {
  resetUiState()
  turnController.fullReset()
  patchUiState({ showReasoning: true })
  turnController.startMessage()

  let publications = 0

  const unlisten = $turnState.listen((state, previous) => {
    if (state.streamSegments !== previous.streamSegments) {
      publications++
    }
  })

  const started = performance.now()

  for (let index = 0; index < deltaCount; index++) {
    turnController.recordReasoningDelta(chunk)
  }

  const enqueueMs = performance.now() - started

  await new Promise(resolve => setTimeout(resolve, STREAM_BATCH_MS + 4))
  const settledMs = performance.now() - started

  const live = getTurnState()
    .streamSegments.map(message => message.thinking ?? '')
    .join('')

  unlisten()

  const { finalMessages } = turnController.recordMessageComplete({ text: '' })
  const retained = finalMessages.map(message => message.thinking ?? '').join('')

  return { enqueueMs, finalExact: live === expected && retained === expected, publications, settledMs }
}

for (let index = 0; index < warmups; index++) {
  await runSample()
}

const results: Sample[] = []

for (let index = 0; index < samples; index++) {
  results.push(await runSample())
}

if (results.some(sample => !sample.finalExact)) {
  throw new Error('reasoning stream was not retained exactly')
}

const distribution = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b)

  return {
    max: sorted.at(-1),
    median: sorted[Math.floor(sorted.length / 2)],
    min: sorted[0]
  }
}

const cpus = os.cpus()

console.log(
  JSON.stringify(
    {
      bufferLimits: {
        pendingDeltaChars:
          typeof controllerExports.REASONING_PENDING_BUFFER_MAX_CHARS === 'number'
            ? controllerExports.REASONING_PENDING_BUFFER_MAX_CHARS
            : 'unbounded',
        reasoningTailCompactsAboveChars:
          typeof controllerExports.REASONING_LIVE_COMPACT_AT_CHARS === 'number'
            ? controllerExports.REASONING_LIVE_COMPACT_AT_CHARS
            : 80_000,
        reasoningTailRetainsChars:
          typeof controllerExports.REASONING_LIVE_RETAIN_CHARS === 'number'
            ? controllerExports.REASONING_LIVE_RETAIN_CHARS
            : 60_000
      },
      hardware: {
        arch: process.arch,
        cpuCount: cpus.length,
        cpuModel: cpus[0]?.model ?? 'unknown',
        platform: `${process.platform} ${os.release()}`,
        totalMemoryBytes: os.totalmem()
      },
      label,
      node: process.version,
      reactInkRendering: false,
      stream: { chunkBytes, deltaCount, totalBytes: Buffer.byteLength(expected) },
      timingMs: {
        enqueue: distribution(results.map(sample => sample.enqueueMs)),
        settled: distribution(results.map(sample => sample.settledMs))
      },
      streamSegmentPublications: distribution(results.map(sample => sample.publications)),
      samples: results,
      streamBatchMs: STREAM_BATCH_MS
    },
    null,
    2
  )
)
