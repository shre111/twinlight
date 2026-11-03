import type { BufferAttribute } from 'three/webgpu'
import { describe, expect, it } from 'vitest'
import { NEVER, type StageCity, type StageTape } from '../src/inputs'
import { Fog } from '../src/layers/fog'
import {
  eventMinutes,
  ResidentsLayer,
  residentPositions,
  residentSubset,
} from '../src/layers/residents'

const residents: StageCity['residents'] = {
  count: 3,
  x: Float32Array.of(10, 20, 30),
  y: Float32Array.of(100, 200, 300),
  z: Float32Array.of(2.5, 5.75, 9),
  building: Uint32Array.of(0, 0, 1),
}

const events: StageTape['residents'] = {
  awareAt: Uint32Array.of(60, NEVER, 120),
  firstOrderAt: Uint32Array.of(NEVER, NEVER, 180),
  orderCount: Uint16Array.of(0, 0, 1),
}

describe('residentSubset', () => {
  it('keeps every resident at or under the cap', () => {
    expect(Array.from(residentSubset(5, 5))).toEqual([0, 1, 2, 3, 4])
    expect(Array.from(residentSubset(3, 100))).toEqual([0, 1, 2])
  })

  it('spreads a capped subset evenly through the list, the same way every time', () => {
    const picks = residentSubset(100_000, 25_000)
    expect(picks.length).toBe(25_000)
    expect(picks[0]).toBe(0)
    expect(picks[1]).toBe(4)
    expect(picks[24_999]).toBe(99_996)
    expect(new Set(picks).size).toBe(25_000)
    expect(Array.from(residentSubset(100_000, 25_000))).toEqual(Array.from(picks))
  })

  it('handles an empty city and a zero cap', () => {
    expect(residentSubset(0, 100).length).toBe(0)
    expect(residentSubset(100, 0).length).toBe(0)
  })
})

describe('residentPositions', () => {
  it('maps pack (x, y, z) to three.js (x, z, -y) for the picked residents', () => {
    const picks = Uint32Array.of(2, 0)
    expect(Array.from(residentPositions(residents, picks))).toEqual([30, 9, -300, 10, 2.5, -100])
  })
})

describe('eventMinutes', () => {
  it('uploads minutes as floats and keeps never far beyond any run', () => {
    const minutes = eventMinutes(events.awareAt, Uint32Array.of(0, 1, 2))
    expect(minutes[0]).toBe(60)
    expect(minutes[2]).toBe(120)
    expect(minutes[1]).toBeGreaterThan(365 * 24 * 60)
    expect(Number.isFinite(minutes[1])).toBe(true)
  })
})

describe('ResidentsLayer', () => {
  const layer = new ResidentsLayer(residents, events, 2, new Fog())
  const attribute = (name: string) => layer.object.geometry.getAttribute(name) as BufferAttribute

  it('draws one instanced sprite per picked resident', () => {
    expect(layer.count).toBe(2)
    expect(layer.object.count).toBe(2)
    expect(layer.object.frustumCulled).toBe(false)
  })

  it('uploads positions and event times once as instanced attributes', () => {
    expect(attribute('home').count).toBe(2)
    expect(Array.from(attribute('home').array)).toEqual([10, 2.5, -100, 20, 5.75, -200])
    expect(Array.from(attribute('awareAt').array)).toEqual([60, 2 ** 32])
    expect(Array.from(attribute('orderAt').array)).toEqual([2 ** 32, 2 ** 32])
    for (const name of ['home', 'awareAt', 'orderAt']) {
      expect(
        (attribute(name) as { isInstancedBufferAttribute?: boolean }).isInstancedBufferAttribute,
      ).toBe(true)
    }
  })

  it('exposes time as a uniform only', () => {
    layer.time.value = 4321
    expect(layer.time.value).toBe(4321)
  })
})
