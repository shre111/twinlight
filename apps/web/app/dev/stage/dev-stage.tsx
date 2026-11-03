'use client'

import type { BackendPreference, Stage, TierOption } from '@twinlight/stage'
import { StageHost } from '../../../components/stage-host'

declare global {
  interface Window {
    /** The stage once its city is loaded; shots and the perf report wait for it. */
    __twinlightStage?: Stage
  }
}

function loadSyntheticCity(stage: Stage, residents: number) {
  let cancelled = false
  import('@twinlight/stage/synthetic')
    .then(({ synthesizeToyRun }) => {
      if (cancelled) return
      const { city } = synthesizeToyRun(residents)
      return stage.loadCity(city)
    })
    .then(() => {
      if (!cancelled) window.__twinlightStage = stage
    })
    .catch(console.error)
  return () => {
    cancelled = true
    if (window.__twinlightStage === stage) window.__twinlightStage = undefined
  }
}

export function DevStage(props: {
  tier?: TierOption
  backend?: BackendPreference
  hud: boolean
  residents: number
}) {
  return (
    <main>
      <StageHost
        tier={props.tier}
        backend={props.backend}
        hud={props.hud}
        onReady={(stage) => loadSyntheticCity(stage, props.residents)}
      />
    </main>
  )
}
