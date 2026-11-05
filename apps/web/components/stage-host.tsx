'use client'

import type { BackendPreference, Stage, TierOption } from '@twinlight/stage'
import { useEffect, useEffectEvent, useRef, useState } from 'react'
import styles from './stage-host.module.css'

interface StageHostProps {
  tier?: TierOption
  backend?: BackendPreference
  hud?: boolean
  /** Holds wall-clock animation for repeatable screenshots. */
  still?: boolean
  /** Why what this stage was meant to show could not load. */
  failure?: string | null
  /** Runs once the stage is live; a returned function runs before the stage is disposed. */
  onReady?: (stage: Stage) => (() => void) | undefined
}

export function StageHost({
  tier = 'auto',
  backend = 'auto',
  hud = false,
  still = false,
  failure,
  onReady,
}: StageHostProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const ready = useEffectEvent((stage: Stage) => onReady?.(stage))
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    setFailed(false)
    // A fresh canvas per mount, because a canvas keeps the first GPU context it hands out.
    const canvas = document.createElement('canvas')
    canvas.setAttribute('role', 'img')
    canvas.setAttribute('aria-label', 'Live city view')
    host.append(canvas)
    let stage: Stage | undefined
    let release: (() => void) | undefined
    let disposed = false

    import('@twinlight/stage')
      .then(({ Stage }) => Stage.create(canvas, { tier, backend, hud, still }))
      .then((created) => {
        if (disposed) {
          created.dispose()
          return
        }
        stage = created
        release = ready(created)
      })
      .catch((error: unknown) => {
        console.error(error)
        if (!disposed) setFailed(true)
      })

    return () => {
      disposed = true
      release?.()
      stage?.dispose()
      canvas.remove()
    }
  }, [tier, backend, hud, still])

  const message = failed
    ? 'This view needs WebGPU or WebGL 2. Try a recent Chrome, Edge, Safari or Firefox.'
    : failure
  return (
    <div ref={hostRef} className={styles.host}>
      {message && (
        <p className={styles.failure} role="alert">
          {message}
        </p>
      )}
    </div>
  )
}
