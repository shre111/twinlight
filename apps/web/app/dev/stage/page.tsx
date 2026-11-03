import type { BackendPreference, TierOption } from '@twinlight/stage'
import { notFound } from 'next/navigation'
import { DevStage, type RunSource } from './dev-stage'

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

const PACK_NAME = /^[a-z0-9-]+$/

function runSource(query: { pack?: unknown; residents?: unknown }): RunSource {
  return typeof query.pack === 'string' && PACK_NAME.test(query.pack)
    ? { pack: query.pack }
    : { residents: residentCount(query.residents) }
}

function startMinutes(query: { t?: unknown; day?: unknown }): number {
  const minutes = Number(query.t)
  if (typeof query.t === 'string' && query.t !== '' && Number.isFinite(minutes) && minutes >= 0) {
    return minutes
  }
  const day = String(query.day)
  const hours = Object.hasOwn(STILL_HOURS, day) ? STILL_HOURS[day] : STILL_HOURS[DEFAULT_DAY]
  return (hours ?? 0) * 60
}

export default async function DevStagePage(props: PageProps<'/dev/stage'>) {
  if (process.env.NODE_ENV === 'production') notFound()
  const query = await props.searchParams
  return (
    <DevStage
      tier={pick(query.tier, TIER_OPTIONS)}
      backend={pick(query.backend, BACKEND_OPTIONS)}
      hud={query.hud !== '0'}
      source={runSource(query)}
      minutes={startMinutes(query)}
    />
  )
}
