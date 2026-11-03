import { describe, expect, it } from 'vitest'
import type { StageTape } from '../src/inputs'
import { isOutOfStock, updateStockouts } from '../src/stockouts'

// Store 0: out 100 to 200. Store 1: never out. Store 2: out 50 to 60, 300 to 400 and 400 to 450.
const tape: StageTape['stores'] = {
  stockoutOffsets: Uint32Array.of(0, 1, 1, 4),
  stockoutIntervals: Uint32Array.of(100, 200, 50, 60, 300, 400, 400, 450),
}

describe('isOutOfStock', () => {
  it('is out from the start minute up to but not including the end minute', () => {
    expect(isOutOfStock(tape, 0, 99)).toBe(false)
    expect(isOutOfStock(tape, 0, 100)).toBe(true)
    expect(isOutOfStock(tape, 0, 150)).toBe(true)
    expect(isOutOfStock(tape, 0, 199)).toBe(true)
    expect(isOutOfStock(tape, 0, 200)).toBe(false)
  })

  it('reads fractional minutes between whole ones', () => {
    expect(isOutOfStock(tape, 0, 99.5)).toBe(false)
    expect(isOutOfStock(tape, 0, 199.5)).toBe(true)
  })

  it('keeps a store with no intervals in stock', () => {
    for (const minute of [0, 50, 100, 1e6]) expect(isOutOfStock(tape, 1, minute)).toBe(false)
  })

  it('finds the right interval among several, including back-to-back ones', () => {
    const at = (minute: number) => isOutOfStock(tape, 2, minute)
    expect([0, 49, 50, 59, 60, 61].map(at)).toEqual([false, false, true, true, false, false])
    expect([299, 300, 399, 400, 449, 450, 1e6].map(at)).toEqual([
      false,
      true,
      true,
      true,
      true,
      false,
      false,
    ])
  })

  it('does not read past the offsets for a store the tape lacks', () => {
    expect(isOutOfStock(tape, 3, 100)).toBe(false)
    expect(isOutOfStock(tape, 99, 100)).toBe(false)
  })
})

describe('updateStockouts', () => {
  it('handles a city with no stores', () => {
    const empty: StageTape['stores'] = {
      stockoutOffsets: Uint32Array.of(0),
      stockoutIntervals: new Uint32Array(0),
    }
    expect(updateStockouts(empty, 100, new Float32Array(0))).toBe(false)
  })

  it('writes each store state and reports a change only when one flips', () => {
    const out = new Float32Array(3)
    expect(updateStockouts(tape, 10, out)).toBe(false)
    expect(Array.from(out)).toEqual([0, 0, 0])

    expect(updateStockouts(tape, 55, out)).toBe(true)
    expect(Array.from(out)).toEqual([0, 0, 1])
    expect(updateStockouts(tape, 58, out)).toBe(false)

    expect(updateStockouts(tape, 150, out)).toBe(true)
    expect(Array.from(out)).toEqual([1, 0, 0])

    expect(updateStockouts(tape, 399, out)).toBe(true)
    expect(Array.from(out)).toEqual([0, 0, 1])
    expect(updateStockouts(tape, 400, out)).toBe(false)

    expect(updateStockouts(tape, 120, out)).toBe(true)
    expect(Array.from(out)).toEqual([1, 0, 0])
  })
})
