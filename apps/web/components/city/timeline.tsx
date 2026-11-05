'use client'

import type { Stage, StageTape } from '@twinlight/stage'
import { SPEEDS } from '@twinlight/stage/playback'
import { useCallback, useEffect, useSyncExternalStore } from 'react'
import { simClockLabel } from '../../lib/timeline-track'
import styles from './timeline.module.css'

const SPACE_KEEPS_ITS_MEANING =
  'button, a, select, summary, textarea, input:not([type="range"]), [role="button"], [contenteditable="true"]'

/** The docked timeline: play and pause, speed and the sim clock. */
export function Timeline({ stage, tape }: { stage: Stage; tape: StageTape }) {
  const clock = useSyncExternalStore(
    stage.subscribeClock,
    () => stage.clock,
    () => stage.clock,
  )
  const nextSpeed = SPEEDS[(SPEEDS.indexOf(clock.speed) + 1) % SPEEDS.length] ?? SPEEDS[0]

  const togglePlay = useCallback(() => {
    if (stage.clock.playing) stage.pause()
    else stage.play()
  }, [stage])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code !== 'Space' || event.repeat) return
      if (event.ctrlKey || event.metaKey || event.altKey) return
      const target = event.target as HTMLElement | null
      if (target?.closest(SPACE_KEEPS_ITS_MEANING)) return
      event.preventDefault()
      togglePlay()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [togglePlay])

  return (
    <footer className={styles.timeline}>
      <button
        type="button"
        className={styles.play}
        aria-label={clock.playing ? 'Pause' : 'Play'}
        onClick={togglePlay}
      >
        <svg aria-hidden="true" width="16" height="16" viewBox="0 0 16 16">
          {clock.playing ? (
            <path d="M3 2h3.5v12H3zM9.5 2H13v12H9.5z" />
          ) : (
            <path d="M4 2.5v11l9.5-5.5z" />
          )}
        </svg>
      </button>

      <fieldset className={styles.speeds}>
        <legend className={styles.visuallyHidden}>Speed</legend>
        {SPEEDS.map((speed) => (
          <button
            key={speed}
            type="button"
            aria-pressed={clock.speed === speed}
            aria-label={`${speed}× speed`}
            onClick={() => stage.setSpeed(speed)}
          >
            {speed}×
          </button>
        ))}
      </fieldset>

      <span className={styles.time}>{simClockLabel(clock.minutes, tape.simMinutes)}</span>

      <button
        type="button"
        className={styles.speedCycle}
        aria-label={`Speed ${clock.speed}×, change to ${nextSpeed}×`}
        onClick={() => stage.setSpeed(nextSpeed)}
      >
        {clock.speed}×
      </button>
    </footer>
  )
}
