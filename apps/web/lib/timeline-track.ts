import type { StageTape } from '@twinlight/stage'

/** The track is drawn in a viewBox this wide, stretched to the timeline's width. */
export const TRACK_WIDTH = 1000
const BASELINE = 45
const SPARK_HEIGHT = 26
const MAX_SPARK_POINTS = 400
const MINUTES_PER_DAY = 24 * 60

export interface TimelineEvent {
  label: string
  /** Shown instead of `label` on phones. */
  short: string
  at: number
}

export interface Track {
  days: number
  /** The area under adoption across the whole run. */
  spark: string
  /** Where stores ran out, as x positions, one per distinct spot on the track. */
  stockouts: number[]
  rain: { from: number; to: number }[]
}

export function trackX(minutes: number, simMinutes: number): number {
  if (simMinutes <= 0) return 0
  return (Math.min(Math.max(minutes, 0), simMinutes) / simMinutes) * TRACK_WIDTH
}

function heldInsideRun(minutes: number, simMinutes: number): number {
  return Math.max(Math.min(minutes, simMinutes - 1), 0)
}

/** "Day 10", held inside the run. */
export function simDayLabel(minutes: number, simMinutes: number): string {
  return `Day ${Math.floor(heldInsideRun(minutes, simMinutes) / MINUTES_PER_DAY) + 1}`
}

/** "Day 10, 9:40 pm", in ten-minute steps like the prototype, held inside the run. */
export function simClockLabel(minutes: number, simMinutes: number): string {
  const t = heldInsideRun(minutes, simMinutes)
  const ofDay = Math.floor((t % MINUTES_PER_DAY) / 10) * 10
  const hour = Math.floor(ofDay / 60)
  const clock = `${hour % 12 === 0 ? 12 : hour % 12}:${String(ofDay % 60).padStart(2, '0')}`
  return `${simDayLabel(t, simMinutes)}, ${clock} ${hour < 12 ? 'am' : 'pm'}`
}

export function buildTrack(tape: StageTape): Track {
  const { simMinutes } = tape
  return {
    days: Math.max(1, Math.round(simMinutes / MINUTES_PER_DAY)),
    spark: adoptionArea(tape),
    stockouts: stockoutSpots(tape),
    rain: rainSpans(tape),
  }
}

function adoptionArea({ aggregates, tickMinutes, simMinutes }: StageTape): string {
  const adopted = aggregates.adopted
  let peak = 0
  for (const count of adopted) peak = Math.max(peak, count)
  const step = Math.max(1, Math.ceil(adopted.length / MAX_SPARK_POINTS))
  const points = [`M0 ${BASELINE}`]
  for (let tick = 0; tick < adopted.length; tick += step) {
    const x = trackX((tick + 1) * tickMinutes, simMinutes)
    const height = peak > 0 ? ((adopted[tick] ?? 0) / peak) * SPARK_HEIGHT : 0
    points.push(`L${x.toFixed(1)} ${(BASELINE - height).toFixed(1)}`)
  }
  points.push(`L${TRACK_WIDTH} ${BASELINE} Z`)
  return points.join(' ')
}

function stockoutSpots({ stores, simMinutes }: StageTape): number[] {
  const spots = new Set<number>()
  for (let i = 0; i < stores.stockoutIntervals.length; i += 2) {
    spots.add(Math.round(trackX(stores.stockoutIntervals[i] ?? 0, simMinutes)))
  }
  return [...spots].sort((a, b) => a - b)
}

function rainSpans({ aggregates, tickMinutes, simMinutes }: StageTape): Track['rain'] {
  const spans: Track['rain'] = []
  const ticks = aggregates.rainIntensity.length
  let start = -1
  for (let tick = 0; tick <= ticks; tick++) {
    const raining = tick < ticks && (aggregates.rainIntensity[tick] ?? 0) > 0
    if (raining && start < 0) start = tick
    if (!raining && start >= 0) {
      spans.push({
        from: trackX(start * tickMinutes, simMinutes),
        to: trackX(tick * tickMinutes, simMinutes),
      })
      start = -1
    }
  }
  return spans
}
