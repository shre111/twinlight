import type { BackendPreference, TierOption } from '@twinlight/stage'
import { notFound } from 'next/navigation'
import { DevStage } from './dev-stage'

const TIER_OPTIONS: readonly TierOption[] = ['auto', 'high', 'medium', 'low']
const BACKEND_OPTIONS: readonly BackendPreference[] = ['auto', 'webgl2']

function pick<T extends string>(value: unknown, options: readonly T[]): T | undefined {
  return options.find((option) => option === value)
}

export default async function DevStagePage(props: PageProps<'/dev/stage'>) {
  if (process.env.NODE_ENV === 'production') notFound()
  const query = await props.searchParams
  return (
    <DevStage
      tier={pick(query.tier, TIER_OPTIONS)}
      backend={pick(query.backend, BACKEND_OPTIONS)}
    />
  )
}
