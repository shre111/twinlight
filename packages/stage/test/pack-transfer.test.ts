import { describe, expect, it } from 'vitest'
import type { StageCity } from '../src/inputs'
import { cityTransfers } from '../src/pack-transfer'

// Columns sharing one file buffer, as the pack reader returns them.
function city(): StageCity {
  const buildingsFile = new ArrayBuffer(6 * 8)
  const column = (i: number) => new Float32Array(buildingsFile, i * 8, 2)
  const residentsFile = new ArrayBuffer(32)
  return {
    buildings: {
      count: 2,
      x: column(0),
      y: column(1),
      width: column(2),
      depth: column(3),
      height: column(4),
      rotation: column(5),
      tileStart: Uint32Array.of(0, 2),
    },
    roads: { segments: Float32Array.of(0, 0, 1, 1), kind: Uint8Array.of(0) },
    residents: {
      count: 1,
      x: new Float32Array(residentsFile, 0, 1),
      y: new Float32Array(residentsFile, 8, 1),
      z: new Float32Array(residentsFile, 16, 1),
      building: new Uint32Array(residentsFile, 24, 1),
    },
    stores: { count: 0, x: new Float32Array(), y: new Float32Array(), radius: new Float32Array() },
    bounds: { minX: 0, minY: 0, maxX: 1, maxY: 1 },
  }
}

describe('cityTransfers', () => {
  it('names each buffer once, so a city crosses to the page without a copy', () => {
    const original = city()
    original.buildings.x.set([3, 4])
    const transfers = cityTransfers(original)
    expect(new Set(transfers).size).toBe(transfers.length)
    expect(transfers).toContain(original.buildings.x.buffer)
    expect(transfers).toContain(original.residents.building.buffer)

    const moved = structuredClone(original, { transfer: transfers })
    expect(Array.from(moved.buildings.x)).toEqual([3, 4])
    expect(moved.buildings.y.buffer).toBe(moved.buildings.x.buffer)
    expect(moved.bounds).toEqual(original.bounds)
    expect(original.buildings.x.buffer.byteLength).toBe(0)
  })

  it('is needed: a buffer named twice cannot be transferred', () => {
    const shared = city()
    const twice = [shared.buildings.x.buffer, shared.buildings.y.buffer]
    expect(() => structuredClone(shared, { transfer: twice })).toThrow()
  })
})
