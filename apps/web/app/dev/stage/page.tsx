import type { BackendPreference, TierOption } from '@twinlight/stage'
import { notFound } from 'next/navigation'
import { DevStage } from './dev-stage'

const TIER_OPTIONS: readonly TierOption[] = ['auto', 'high', 'medium', 'low']
const BACKEND_OPTIONS: readonly BackendPreference[] = ['auto', 'webgl2']
const DEFAULT_RESIDENTS = 20_000
const MAX_RESIDENTS = 100_000
// The prototype's still hours for its day 1, 6 and 13 renders.
const STILL_HOURS: Record<string, number> = { '1': 20, '6': 140, '13': 308 }
const DEFAULT_DAY = '6'

function pick<T extends string>(value: unknown, options: readonly T[]): T | undefined {
  return options.find((option) => option === value)
}

function residentCount(value: unknown): number {
  const count = Number(value)
  return Number.isInteger(count) && count > 0 ? Math.min(count, MAX_RESIDENTS) : DEFAULT_RESIDENTS
}

function startMinutes(query: { t?: unknown; day?: unknown }): number {
  const minutes = Number(query.t)
  if (typeof query.t === 'string' && Number.isFinite(minutes) && minutes >= 0) return minutes
  const hours = STILL_HOURS[String(query.day)] ?? STILL_HOURS[DEFAULT_DAY] ?? 0
  return hours * 60
}

export default async function DevStagePage(props: PageProps<'/dev/stage'>) {
  if (process.env.NODE_ENV === 'production') notFound()
  const query = await props.searchParams
  return (
    <DevStage
      tier={pick(query.tier, TIER_OPTIONS)}
      backend={pick(query.backend, BACKEND_OPTIONS)}
      hud={query.hud !== '0'}
      residents={residentCount(query.residents)}
      minutes={startMinutes(query)}
    />
  )
}
