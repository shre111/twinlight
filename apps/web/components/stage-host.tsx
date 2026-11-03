'use client'

import type { BackendPreference, Stage, TierOption } from '@twinlight/stage'
import { useEffect, useEffectEvent, useRef, useState } from 'react'
import styles from './stage-host.module.css'

interface StageHostProps {
  tier?: TierOption
  backend?: BackendPreference
  onReady?: (stage: Stage) => void
}

export function StageHost({ tier = 'auto', backend = 'auto', onReady }: StageHostProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const ready = useEffectEvent((stage: Stage) => onReady?.(stage))
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    // A fresh canvas per mount, because a canvas keeps the first GPU context it hands out.
    const canvas = document.createElement('canvas')
    host.append(canvas)
    let stage: Stage | undefined
    let disposed = false

    import('@twinlight/stage')
      .then(({ Stage }) => Stage.create(canvas, { tier, backend }))
      .then((created) => {
        if (disposed) {
          created.dispose()
          return
        }
        stage = created
        ready(created)
      })
      .catch((error: unknown) => {
        console.error(error)
        if (!disposed) setFailed(true)
      })

    return () => {
      disposed = true
      stage?.dispose()
      canvas.remove()
    }
  }, [tier, backend])

  return (
    <div ref={hostRef} className={styles.host}>
      {failed && (
        <p className={styles.failure} role="alert">
          This view needs WebGPU or WebGL 2. Try a recent Chrome, Edge, Safari or Firefox.
        </p>
      )}
    </div>
  )
}
