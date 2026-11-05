import type { StageTape } from '@twinlight/stage'
import { describe, expect, it } from 'vitest'
import { figuresAt } from './run-figures'

const DAY = 24 * 60
const SIM_MINUTES = 3 * DAY

function tape(overrides: {
  residents?: number
  orders?: (tick: number) => number
  stockouts?: number[][]
  tickMinutes?: number
}): StageTape {
  const tickMinutes = overrides.tickMinutes ?? 60
  const ticks = SIM_MINUTES / tickMinutes
  const residents = overrides.residents ?? 1000
  const byStore = overrides.stockouts ?? [[], []]
  const offsets = new Uint32Array(byStore.length + 1)
  byStore.forEach((intervals, store) => {
    offsets[store + 1] = (offsets[store] ?? 0) + intervals.length / 2
  })
  return {
    simMinutes: SIM_MINUTES,
    tickMinutes,
    residents: {
      awareAt: new Uint32Array(residents),
      firstOrderAt: new Uint32Array(residents),
      orderCount: new Uint16Array(residents),
    },
    stores: {
      stockoutOffsets: offsets,
      stockoutIntervals: Uint32Array.from(byStore.flat()),
    },
    aggregates: {
      aware: Uint32Array.from({ length: ticks }, (_, tick) => (tick + 1) * 4),
      adopted: Uint32Array.from({ length: ticks }, (_, tick) => tick + 1),
      orders: Uint32Array.from({ length: ticks }, (_, tick) => overrides.orders?.(tick) ?? 1),
      storesOut: new Uint16Array(ticks),
      rainIntensity: new Float32Array(ticks),
    },
  }
}

describe('figuresAt', () => {
  it('shows nothing before the first tick has finished', () => {
    const figures = figuresAt(tape({}), 59)
    expect(figures).toMatchObject({ ordered: 0, heardShare: 0, ordersToday: 0 })
  })

  it('reads the last finished tick, never the one in progress', () => {
    expect(figuresAt(tape({}), 60).ordered).toBe(1)
    expect(figuresAt(tape({}), 119).ordered).toBe(1)
    expect(figuresAt(tape({}), 120).ordered).toBe(2)
  })

  it('reports heard as a share of residents', () => {
    expect(figuresAt(tape({ residents: 1000 }), 250 * 15).heardShare).toBeCloseTo((62 * 4) / 1000)
  })

  it('reports no share for a run without residents', () => {
    expect(figuresAt(tape({ residents: 0 }), DAY).heardShare).toBe(0)
  })

  it('sums the orders of the day so far and starts again at midnight', () => {
    const run = tape({ orders: () => 2 })
    expect(figuresAt(run, 3 * 60).ordersToday).toBe(6)
    expect(figuresAt(run, DAY - 1).ordersToday).toBe(23 * 2)
    expect(figuresAt(run, DAY).ordersToday).toBe(0)
    expect(figuresAt(run, DAY + 60).ordersToday).toBe(2)
  })

  it('stays on the last day at the end of the run', () => {
    const figures = figuresAt(tape({ orders: () => 2 }), SIM_MINUTES)
    expect(figures.ordersToday).toBe(24 * 2)
    expect(figures.ordered).toBe(72)
  })

  it('holds the figures inside the run past either end', () => {
    expect(figuresAt(tape({}), -30).ordered).toBe(0)
    expect(figuresAt(tape({}), SIM_MINUTES * 2).ordered).toBe(72)
  })

  it('counts days the same way with finer ticks', () => {
    const run = tape({ tickMinutes: 15, orders: () => 1 })
    expect(figuresAt(run, DAY + 30).ordersToday).toBe(2)
    expect(figuresAt(run, SIM_MINUTES).ordered).toBe(288)
  })

  it('counts stores out at the exact minute, end exclusive', () => {
    const run = tape({ stockouts: [[100, 200], [150, 300], []] })
    expect(figuresAt(run, 99)).toMatchObject({ storesOut: 0, stores: 3 })
    expect(figuresAt(run, 100).storesOut).toBe(1)
    expect(figuresAt(run, 150).storesOut).toBe(2)
    expect(figuresAt(run, 200).storesOut).toBe(1)
    expect(figuresAt(run, 300).storesOut).toBe(0)
  })
})
