import { describe, expect, it } from 'vitest'
import type { StageTape } from '../src/inputs'
import { clampTime, tickAt, warmFraction } from '../src/time'

const RESIDENTS = 1000
const tape: Pick<StageTape, 'simMinutes' | 'tickMinutes' | 'aggregates'> = {
  simMinutes: 240,
  tickMinutes: 15,
  aggregates: {
    aware: new Uint32Array(16),
    // Adopters per tick: 20% of residents is 200, reached from tick 3.
    adopted: Uint32Array.of(
      0,
      50,
      100,
      200,
      300,
      300,
      300,
      300,
      300,
      300,
      300,
      300,
      300,
      300,
      300,
      300,
    ),
    orders: new Uint32Array(16),
    storesOut: new Uint16Array(16),
    rainIntensity: new Float32Array(16),
  },
}

describe('clampTime', () => {
  it('keeps minutes inside the run', () => {
    expect(clampTime(-5, tape)).toBe(0)
    expect(clampTime(100, tape)).toBe(100)
    expect(clampTime(1e9, tape)).toBe(240)
  })

  it('treats NaN and infinities as the start', () => {
    expect(clampTime(Number.NaN, tape)).toBe(0)
    expect(clampTime(Number.POSITIVE_INFINITY, tape)).toBe(0)
  })
})

describe('tickAt', () => {
  it('floors to the tick and stays inside the aggregates', () => {
    expect(tickAt(0, tape, 16)).toBe(0)
    expect(tickAt(14, tape, 16)).toBe(0)
    expect(tickAt(15, tape, 16)).toBe(1)
    expect(tickAt(240, tape, 16)).toBe(15)
    expect(tickAt(-1, tape, 16)).toBe(0)
  })
})

describe('warmFraction', () => {
  it('reads the adopted share at the tick and saturates at a fifth of residents', () => {
    expect(warmFraction(0, tape, RESIDENTS)).toBe(0)
    expect(warmFraction(20, tape, RESIDENTS)).toBeCloseTo(0.25)
    expect(warmFraction(30, tape, RESIDENTS)).toBeCloseTo(0.5)
    expect(warmFraction(45, tape, RESIDENTS)).toBe(1)
    expect(warmFraction(240, tape, RESIDENTS)).toBe(1)
  })

  it('is zero without aggregates or residents', () => {
    const empty = { ...tape, aggregates: { ...tape.aggregates, adopted: new Uint32Array(0) } }
    expect(warmFraction(30, empty, RESIDENTS)).toBe(0)
    expect(warmFraction(30, tape, 0)).toBe(0)
  })
})
