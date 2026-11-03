export type TierName = 'high' | 'medium' | 'low'
export type TierOption = TierName | 'auto'

export interface Tier {
  residents: number
  bloomScale: number
  rainParticles: boolean
  maxPixelRatio: number
}

export const TIERS: Record<TierName, Tier> = {
  high: { residents: 100_000, bloomScale: 1, rainParticles: true, maxPixelRatio: 1.75 },
  medium: { residents: 60_000, bloomScale: 0.5, rainParticles: true, maxPixelRatio: 1.5 },
  low: { residents: 25_000, bloomScale: 0, rainParticles: false, maxPixelRatio: 1.25 },
}

export function initialTier(coarsePointer: boolean): TierName {
  return coarsePointer ? 'medium' : 'high'
}
