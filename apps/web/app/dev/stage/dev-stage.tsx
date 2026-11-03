'use client'

import type { BackendPreference, Stage, TierOption } from '@twinlight/stage'
import { StageHost } from '../../../components/stage-host'

declare global {
  interface Window {
    __twinlightStage?: Stage
  }
}

function expose(stage: Stage) {
  window.__twinlightStage = stage
  return () => {
    if (window.__twinlightStage === stage) window.__twinlightStage = undefined
  }
}

export function DevStage(props: { tier?: TierOption; backend?: BackendPreference }) {
  return (
    <main>
      <StageHost tier={props.tier} backend={props.backend} onReady={expose} />
    </main>
  )
}
