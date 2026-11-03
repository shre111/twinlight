import type { StageTape } from './inputs'

type Stockouts = StageTape['stores']

/** Whether `store` is out of stock at `minute`: inside an interval, from `start` up to but not including `end`. */
export function isOutOfStock(stockouts: Stockouts, store: number, minute: number): boolean {
  const { stockoutOffsets: offsets, stockoutIntervals: intervals } = stockouts
  const first = offsets[store] ?? 0
  let lo = first
  let hi = Math.max(first, offsets[store + 1] ?? first)
  // Intervals are sorted by start; find the first one starting after `minute`.
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if ((intervals[mid * 2] ?? 0) <= minute) lo = mid + 1
    else hi = mid
  }
  return lo > first && minute < (intervals[(lo - 1) * 2 + 1] ?? 0)
}

/** Writes 1 for each store out of stock at `minute` and 0 otherwise; returns whether any changed. */
export function updateStockouts(stockouts: Stockouts, minute: number, out: Float32Array): boolean {
  let changed = false
  for (let store = 0; store < out.length; store++) {
    const state = isOutOfStock(stockouts, store, minute) ? 1 : 0
    if (out[store] !== state) {
      out[store] = state
      changed = true
    }
  }
  return changed
}
