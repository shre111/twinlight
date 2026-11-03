import type { StageTape } from './inputs'

/** The city is fully warm once this share of residents has ordered. */
export const WARM_ADOPTED_SHARE = 0.2

export function clampTime(simMinutes: number, tape: Pick<StageTape, 'simMinutes'>): number {
  if (!Number.isFinite(simMinutes)) return 0
  return Math.min(Math.max(simMinutes, 0), tape.simMinutes)
}

export function tickAt(simMinutes: number, tape: Pick<StageTape, 'tickMinutes'>, ticks: number) {
  return Math.min(Math.max(Math.floor(simMinutes / tape.tickMinutes), 0), ticks - 1)
}

export function warmFraction(
  simMinutes: number,
  tape: Pick<StageTape, 'tickMinutes' | 'aggregates'>,
  residents: number,
): number {
  const adopted = tape.aggregates.adopted
  if (adopted.length === 0 || residents === 0) return 0
  const tick = tickAt(simMinutes, tape, adopted.length)
  return Math.min(1, (adopted[tick] ?? 0) / (residents * WARM_ADOPTED_SHARE))
}
