'use client'

import type { Stage, StageTape } from '@twinlight/stage'
import { SPEEDS } from '@twinlight/stage/playback'
import {
  type KeyboardEvent,
  type PointerEvent,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useSyncExternalStore,
} from 'react'
import {
  buildTrack,
  simClockLabel,
  simDayLabel,
  type TimelineEvent,
  TRACK_WIDTH,
  trackX,
} from '../../lib/timeline-track'
import styles from './timeline.module.css'

const HOUR = 60
const DAY = 24 * HOUR
const DAY_LABEL_EVERY = { desktop: 5, phone: 10 }
const LONG_TICK_EVERY = 5
// Past this point on the track a marker's label ends at its position, so it stays on screen.
const MARKER_FLIP_X = TRACK_WIDTH * 0.85
const SPACE_KEEPS_ITS_MEANING =
  'button, a, select, summary, textarea, input:not([type="range"]), [role="button"], [contenteditable="true"]'

function markerStyle(x: number) {
  return {
    left: `${(x / TRACK_WIDTH) * 100}%`,
    ...(x > MARKER_FLIP_X && { transform: 'translateX(-100%)' }),
  }
}

/** The docked timeline: play and pause, speed, the sim clock and a scrubber over the run. */
export function Timeline({
  stage,
  tape,
  events,
}: {
  stage: Stage
  tape: StageTape
  events: TimelineEvent[]
}) {
  const clock = useSyncExternalStore(
    stage.subscribeClock,
    () => stage.clock,
    () => stage.clock,
  )
  const { simMinutes } = tape
  const track = useMemo(() => buildTrack(tape), [tape])
  const days = useMemo(() => Array.from({ length: track.days + 1 }, (_, day) => day), [track.days])
  const clipId = useId()
  const scrub = useRef<{ pointer: number; resume: boolean } | null>(null)
  const label = simClockLabel(clock.minutes, simMinutes)
  const nowX = trackX(clock.minutes, simMinutes)

  const nextSpeed = SPEEDS[(SPEEDS.indexOf(clock.speed) + 1) % SPEEDS.length] ?? SPEEDS[0]

  const togglePlay = useCallback(() => {
    if (scrub.current) return
    if (stage.clock.playing) stage.pause()
    else stage.play()
  }, [stage])

  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
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

  const onScrubKey = (event: KeyboardEvent<HTMLInputElement>) => {
    const step = {
      ArrowLeft: -HOUR,
      ArrowDown: -HOUR,
      ArrowRight: HOUR,
      ArrowUp: HOUR,
      PageDown: -DAY,
      PageUp: DAY,
    }[event.key]
    if (step !== undefined) stage.setTime(stage.time + step)
    else if (event.key === 'Home') stage.setTime(0)
    else if (event.key === 'End') stage.setTime(simMinutes)
    else return
    event.preventDefault()
  }

  const startScrub = (event: PointerEvent<HTMLInputElement>) => {
    if (event.button !== 0 || scrub.current) return
    event.currentTarget.setPointerCapture(event.pointerId)
    scrub.current = { pointer: event.pointerId, resume: stage.clock.playing }
    stage.pause()
  }
  const endScrub = (event: PointerEvent<HTMLInputElement>) => {
    if (scrub.current?.pointer !== event.pointerId) return
    const { resume } = scrub.current
    scrub.current = null
    // play() restarts a finished run, which a drag to the end should not do.
    if (resume && stage.time < simMinutes) stage.play()
  }

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

      <span className={styles.time}>{label}</span>

      <button
        type="button"
        className={styles.speedCycle}
        aria-label={`Speed ${clock.speed}×, change to ${nextSpeed}×`}
        onClick={() => stage.setSpeed(nextSpeed)}
      >
        {clock.speed}×
      </button>

      <div className={styles.track}>
        <div aria-hidden="true" className={styles.markers}>
          {events.map((event) => (
            <span key={event.label} style={markerStyle(trackX(event.at, simMinutes))}>
              <span className={styles.full}>{event.label}</span>
              <span className={styles.short}>{event.short}</span>
            </span>
          ))}
          {track.rain.map((span) => (
            <span key={span.from} className={styles.rainLabel} style={markerStyle(span.from)}>
              <span className={styles.full}>Monsoon</span>
              <span className={styles.short}>Rain</span>
            </span>
          ))}
        </div>

        <svg
          aria-hidden="true"
          className={styles.graph}
          viewBox={`0 0 ${TRACK_WIDTH} 64`}
          preserveAspectRatio="none"
        >
          <clipPath id={clipId}>
            <rect width={nowX} height="64" />
          </clipPath>
          {track.rain.map((span) => (
            <rect
              key={span.from}
              className={styles.rain}
              x={span.from}
              y="18"
              width={span.to - span.from}
              height="28"
            />
          ))}
          <path className={styles.spark} d={track.spark} />
          <path className={styles.played} d={track.spark} clipPath={`url(#${clipId})`} />
          <rect className={styles.rail} y="45" width={TRACK_WIDTH} height="2" />
          <rect className={styles.progress} y="45" width={nowX} height="2" />
          {track.stockouts.map((x) => (
            <rect key={x} className={styles.stockout} x={x} y="40" width="4" height="12" rx="1" />
          ))}
          {days.map((day) => (
            <rect
              key={day}
              className={styles.tick}
              x={(day / track.days) * TRACK_WIDTH}
              y="49"
              width="1"
              height={day % LONG_TICK_EVERY === 0 ? 6 : 3}
            />
          ))}
        </svg>

        <input
          className={styles.scrubber}
          type="range"
          min={0}
          max={simMinutes}
          step={1}
          value={Math.round(clock.minutes)}
          aria-label="Day of the launch"
          aria-valuetext={clock.playing ? simDayLabel(clock.minutes, simMinutes) : label}
          onChange={(event) => stage.setTime(Number(event.target.value))}
          onKeyDown={onScrubKey}
          onPointerDown={startScrub}
          onPointerUp={endScrub}
          onPointerCancel={endScrub}
          onLostPointerCapture={endScrub}
        />

        <div aria-hidden="true" className={styles.days}>
          {days
            .filter((day) => day % DAY_LABEL_EVERY.desktop === 0)
            .map((day) => (
              <span
                key={day}
                className={day % DAY_LABEL_EVERY.phone === 0 ? undefined : styles.full}
                style={{ left: `${(day / track.days) * 100}%` }}
              >
                {day === 0 ? 'Launch' : `Day ${day}`}
              </span>
            ))}
        </div>
      </div>
    </footer>
  )
}
