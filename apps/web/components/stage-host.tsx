'use client'

import type { BackendPreference, Stage, TierOption } from '@twinlight/stage'
import { useEffect, useEffectEvent, useRef, useState } from 'react'
import { browserHasGraphics, startFailureMessage, withOneRetry } from '../lib/stage-start'
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
  const [startFailure, setStartFailure] = useState<string | null>(null)

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    setStartFailure(null)
    let canvas: HTMLCanvasElement | undefined
    let stage: Stage | undefined
    let release: (() => void) | undefined
    let disposed = false

    const start = async () => {
      const { Stage } = await import('@twinlight/stage')
      if (disposed) return undefined
      canvas?.remove()
      // A fresh canvas per attempt, because a canvas keeps the first GPU context it hands out.
      canvas = document.createElement('canvas')
      canvas.setAttribute('role', 'img')
      canvas.setAttribute('aria-label', 'Live city view')
      host.append(canvas)
      return Stage.create(canvas, { tier, backend, hud, still })
    }

    withOneRetry(start)
      .then((created) => {
        if (disposed) {
          created?.dispose()
          return
        }
        stage = created
        if (created) release = ready(created)
      })
      .catch(async (error: unknown) => {
        console.error(error)
        const hasGraphics = await browserHasGraphics()
        if (!disposed) setStartFailure(startFailureMessage(hasGraphics))
      })

    return () => {
      disposed = true
      release?.()
      stage?.dispose()
      canvas?.remove()
    }
  }, [tier, backend, hud, still])

  const message = startFailure ?? failure
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
