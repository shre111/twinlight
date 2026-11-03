'use client'

import type { BackendPreference, Stage, TierOption } from '@twinlight/stage'
import { StageHost } from '../../../components/stage-host'

declare global {
  interface Window {
    /** The stage once its city, tape and time are applied; shots and the perf report wait for it. */
    __twinlightStage?: Stage
    /** Set instead when the synthetic run fails to load, so shots fail fast. */
    __twinlightStageError?: string
  }
}

function loadSyntheticRun(stage: Stage, residents: number, minutes: number) {
  let cancelled = false
  import('@twinlight/stage/synthetic')
    .then(async ({ synthesizeToyRun }) => {
      if (cancelled) return
      const { city, tape } = synthesizeToyRun(residents)
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
  residents: number
  minutes: number
}) {
  return (
    <main>
      <StageHost
        tier={props.tier}
        backend={props.backend}
        hud={props.hud}
        onReady={(stage) => loadSyntheticRun(stage, props.residents, props.minutes)}
      />
    </main>
  )
}
