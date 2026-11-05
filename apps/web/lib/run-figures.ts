import type { StageTape } from '@twinlight/stage'
import { isOutOfStock } from '@twinlight/stage/stockouts'

const MINUTES_PER_DAY = 24 * 60

export interface RunFigures {
  ordered: number
  heardShare: number
  ordersToday: number
  storesOut: number
  stores: number
}

/**
 * The live figures at a sim minute. Per-tick aggregates count a tick once it has finished, so
 * they are never ahead of the city; stores are read from their intervals to match the beacons.
 */
export function figuresAt(tape: StageTape, minutes: number): RunFigures {
  const { aggregates, tickMinutes, simMinutes, residents, stores } = tape
  const inRun = Math.max(Math.min(minutes, simMinutes), 0)
  const lastTick = Math.min(Math.floor(inRun / tickMinutes), aggregates.adopted.length) - 1
  const residentCount = residents.awareAt.length
  const storeCount = stores.stockoutOffsets.length - 1
  // The clock holds the last minute inside the run, so "today" does too.
  const clockDay = Math.floor(Math.min(inRun, simMinutes - 1) / MINUTES_PER_DAY)
  const dayFirstTick = Math.max(0, (clockDay * MINUTES_PER_DAY) / tickMinutes)
  let ordersToday = 0
  for (let tick = dayFirstTick; tick <= lastTick; tick++)
    ordersToday += aggregates.orders[tick] ?? 0
  let storesOut = 0
  for (let store = 0; store < storeCount; store++) {
    if (isOutOfStock(stores, store, inRun)) storesOut++
  }
  return {
    ordered: aggregates.adopted[lastTick] ?? 0,
    heardShare: residentCount > 0 ? (aggregates.aware[lastTick] ?? 0) / residentCount : 0,
    ordersToday,
    storesOut,
    stores: storeCount,
  }
}
