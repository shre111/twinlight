'use client'

import type { BackendPreference, Stage, TierOption } from '@twinlight/stage'
import { StageHost } from '../../../components/stage-host'

declare global {
  interface Window {
    /** The stage once its city, tape and time are applied; shots and the perf report wait for it. */
    __twinlightStage?: Stage
    /** Set instead when the run fails to load, so shots fail fast. */
    __twinlightStageError?: string
  }
}

export type RunSource = { pack: string } | { residents: number }

// A generated pack has no tape until P-014, so it shows the city before anyone has heard.
async function runFor(source: RunSource) {
  if ('pack' in source) {
    const { fetchPack, idleTape } = await import('@twinlight/stage')
    const city = await fetchPack(`/dev/packs/${source.pack}`)
    return { city, tape: idleTape(city) }
  }
  const { synthesizeToyRun } = await import('@twinlight/stage/synthetic')
  return synthesizeToyRun(source.residents)
}

function loadRun(stage: Stage, source: RunSource, minutes: number) {
  let cancelled = false
  runFor(source)
    .then(async ({ city, tape }) => {
      if (cancelled) return
      await stage.loadCity(city)
      if (cancelled) return
      await stage.setTape(tape)
      if (cancelled) return
      stage.setTime(minutes)
      window.__twinlightStage = stage
    })
    .catch((error: unknown) => {
      console.error(error)
      if (!cancelled) window.__twinlightStageError = String(error)
    })
  return () => {
    cancelled = true
    window.__twinlightStageError = undefined
    if (window.__twinlightStage === stage) window.__twinlightStage = undefined
  }
}

export function DevStage(props: {
  tier?: TierOption
  backend?: BackendPreference
  hud: boolean
  still: boolean
  source: RunSource
  minutes: number
}) {
  return (
    <main>
      <StageHost
        tier={props.tier}
        backend={props.backend}
        hud={props.hud}
        still={props.still}
        onReady={(stage) => loadRun(stage, props.source, props.minutes)}
      />
    </main>
  )
}
