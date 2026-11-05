'use client'

import type { BackendPreference, Stage, StageTape, TierOption } from '@twinlight/stage'
import { useState } from 'react'
import { Readouts } from '../../../components/city/readouts'
import { Timeline } from '../../../components/city/timeline'
import { StageHost } from '../../../components/stage-host'
import type { TimelineEvent } from '../../../lib/timeline-track'

declare global {
  interface Window {
    /** The stage once its city, tape and time are applied; shots and the perf report wait for it. */
    __twinlightStage?: Stage
    /** Set instead when the run fails to load, so shots fail fast. */
    __twinlightStageError?: string
  }
}

export type RunSource = { fixture: string } | { pack: string } | { residents: number }

interface LoadedRun {
  stage: Stage
  tape: StageTape
  events: TimelineEvent[]
}

const LOAD_FAILURE = 'This run could not load. If its pack is missing, run just fixture-pack.'

// A generated pack has no tape until P-014, so it shows the city before anyone has heard.
// The fixture stands in for that tape with the toy sim's launch; P-014 replaces it with a file.
async function runFor(source: RunSource) {
  if ('fixture' in source) {
    const { fetchPack } = await import('@twinlight/stage')
    const { synthesizeTapeFor, TOY_EVENTS } = await import('@twinlight/stage/synthetic')
    const city = await fetchPack(`/dev/packs/${source.fixture}`)
    return { city, tape: synthesizeTapeFor(city), events: TOY_EVENTS }
  }
  if ('pack' in source) {
    const { fetchPack, idleTape } = await import('@twinlight/stage')
    const city = await fetchPack(`/dev/packs/${source.pack}`)
    return { city, tape: idleTape(city), events: [] }
  }
  const { synthesizeToyRun, TOY_EVENTS } = await import('@twinlight/stage/synthetic')
  return { ...synthesizeToyRun(source.residents), events: TOY_EVENTS }
}

function loadRun(
  stage: Stage,
  source: RunSource,
  minutes: number,
  onLoaded: (run: LoadedRun | null) => void,
  onFailed: (message: string | null) => void,
) {
  let cancelled = false
  runFor(source)
    .then(async ({ city, tape, events }) => {
      if (cancelled) return
      await stage.loadCity(city)
      if (cancelled) return
      await stage.setTape(tape)
      if (cancelled) return
      stage.setTime(minutes)
      onLoaded({ stage, tape, events })
      window.__twinlightStage = stage
    })
    .catch((error: unknown) => {
      console.error(error)
      if (cancelled) return
      window.__twinlightStageError = String(error)
      onFailed(LOAD_FAILURE)
    })
  return () => {
    cancelled = true
    onLoaded(null)
    onFailed(null)
    window.__twinlightStageError = undefined
    if (window.__twinlightStage === stage) window.__twinlightStage = undefined
  }
}

export function DevStage(props: {
  tier?: TierOption
  backend?: BackendPreference
  hud: boolean
  still: boolean
  ui: boolean
  source: RunSource
  minutes: number
}) {
  const [run, setRun] = useState<LoadedRun | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  return (
    <main>
      <StageHost
        tier={props.tier}
        backend={props.backend}
        hud={props.hud}
        still={props.still}
        failure={failure}
        onReady={(stage) => loadRun(stage, props.source, props.minutes, setRun, setFailure)}
      />
      {props.ui && run && (
        <>
          <Readouts stage={run.stage} tape={run.tape} />
          <Timeline stage={run.stage} tape={run.tape} events={run.events} />
        </>
      )}
    </main>
  )
}
