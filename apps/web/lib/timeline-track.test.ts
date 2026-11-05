import type { StageTape } from '@twinlight/stage'
import { describe, expect, it } from 'vitest'
import { buildTrack, simClockLabel, TRACK_WIDTH, trackX } from './timeline-track'

const SIM_MINUTES = 30 * 24 * 60

function tape(overrides: {
  adopted?: number[]
  rain?: number[]
  stockouts?: number[]
  tickMinutes?: number
}): StageTape {
  const tickMinutes = overrides.tickMinutes ?? 60
  const ticks = SIM_MINUTES / tickMinutes
  const fill = (values: number[] | undefined) =>
    Array.from({ length: ticks }, (_, i) => values?.[i] ?? 0)
  return {
    simMinutes: SIM_MINUTES,
    tickMinutes,
    residents: {
      awareAt: new Uint32Array(0),
      firstOrderAt: new Uint32Array(0),
      orderCount: new Uint16Array(0),
    },
    stores: {
      stockoutOffsets: Uint32Array.of(0, (overrides.stockouts?.length ?? 0) / 2),
      stockoutIntervals: Uint32Array.from(overrides.stockouts ?? []),
    },
    aggregates: {
      aware: new Uint32Array(ticks),
      adopted: Uint32Array.from(fill(overrides.adopted)),
      orders: new Uint32Array(ticks),
      storesOut: new Uint16Array(ticks),
      rainIntensity: Float32Array.from(fill(overrides.rain)),
    },
  }
}

describe('simClockLabel', () => {
  it('reads like the design, in ten-minute steps', () => {
    expect(simClockLabel(0, SIM_MINUTES)).toBe('Day 1, 12:00 am')
    expect(simClockLabel(9 * 24 * 60 + 21 * 60 + 47, SIM_MINUTES)).toBe('Day 10, 9:40 pm')
    expect(simClockLabel(12 * 60, SIM_MINUTES)).toBe('Day 1, 12:00 pm')
    expect(simClockLabel(13 * 60 + 5, SIM_MINUTES)).toBe('Day 1, 1:00 pm')
  })

  it('stays inside the run at both ends', () => {
    expect(simClockLabel(-30, SIM_MINUTES)).toBe('Day 1, 12:00 am')
    expect(simClockLabel(SIM_MINUTES, SIM_MINUTES)).toBe('Day 30, 11:50 pm')
    expect(simClockLabel(Number.POSITIVE_INFINITY, SIM_MINUTES)).toBe('Day 30, 11:50 pm')
  })
})

describe('trackX', () => {
  it('maps the run onto the track and clamps outside it', () => {
    expect(trackX(0, SIM_MINUTES)).toBe(0)
    expect(trackX(SIM_MINUTES / 2, SIM_MINUTES)).toBe(TRACK_WIDTH / 2)
    expect(trackX(-5, SIM_MINUTES)).toBe(0)
    expect(trackX(SIM_MINUTES * 2, SIM_MINUTES)).toBe(TRACK_WIDTH)
  })
})

describe('buildTrack', () => {
  it('counts the days of the run', () => {
    expect(buildTrack(tape({})).days).toBe(30)
  })

  it('draws adoption as a closed area rising from the baseline to the peak', () => {
    const { spark } = buildTrack(tape({ adopted: [0, 50, 100, 100] }))
    expect(spark.startsWith('M0 45 ')).toBe(true)
    expect(spark.endsWith(`L${TRACK_WIDTH} 45 Z`)).toBe(true)
    expect(spark).toContain('L4.2 19.0')
  })

  it('stays flat when nobody has ordered', () => {
    const { spark } = buildTrack(tape({}))
    const heights = [...spark.matchAll(/L[\d.]+ ([\d.]+)/g)].map((match) => Number(match[1]))
    expect(new Set(heights)).toEqual(new Set([45]))
  })

  it('keeps the path to a bounded size for long fine-grained runs', () => {
    const { spark } = buildTrack(tape({ tickMinutes: 15 }))
    expect(spark.split('L').length).toBeLessThanOrEqual(402)
  })

  it('marks each distinct stockout start once, in order', () => {
    const day = 24 * 60
    const { stockouts } = buildTrack(
      tape({ stockouts: [10 * day, 10 * day + 60, 5 * day, 6 * day, 10 * day, 11 * day] }),
    )
    expect(stockouts).toEqual([Math.round((5 / 30) * 1000), Math.round((10 / 30) * 1000)])
  })

  it('merges rainy ticks into spans, including one running to the end', () => {
    const rain = Array.from({ length: 720 }, (_, hour) =>
      (hour >= 264 && hour < 336) || hour >= 700 ? 1 : 0,
    )
    const spans = buildTrack(tape({ rain })).rain
    expect(spans).toHaveLength(2)
    expect(spans[0]?.from).toBeCloseTo((264 / 720) * 1000)
    expect(spans[0]?.to).toBeCloseTo((336 / 720) * 1000)
    expect(spans[1]?.to).toBe(TRACK_WIDTH)
  })
})
