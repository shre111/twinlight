import { NEVER, type StageCity, type StageTape } from './inputs'

const RUN_MINUTES = 30 * 24 * 60
const TICK_MINUTES = 15

/** A run in which nobody has heard of the product yet, to show a city pack before P-014's tapes. */
export function idleTape(city: StageCity, simMinutes = RUN_MINUTES): StageTape {
  const residents = city.residents.count
  const ticks = Math.ceil(simMinutes / TICK_MINUTES)
  return {
    simMinutes,
    tickMinutes: TICK_MINUTES,
    residents: {
      awareAt: new Uint32Array(residents).fill(NEVER),
      firstOrderAt: new Uint32Array(residents).fill(NEVER),
      orderCount: new Uint16Array(residents),
    },
    stores: {
      stockoutOffsets: new Uint32Array(city.stores.count + 1),
      stockoutIntervals: new Uint32Array(0),
    },
    aggregates: {
      aware: new Uint32Array(ticks),
      adopted: new Uint32Array(ticks),
      orders: new Uint32Array(ticks),
      storesOut: new Uint16Array(ticks),
      rainIntensity: new Float32Array(ticks),
    },
  }
}
