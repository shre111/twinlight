import { describe, expect, it } from 'vitest'
import { cityBounds, extentOf, fitPoint, TOY_BOUNDS, TOY_EXTENT } from '../src/extent'
import type { StageCity } from '../src/inputs'

function cityWith(x: number[], y: number[], bounds?: StageCity['bounds']): StageCity {
  const f = (values: number[]) => Float32Array.from(values)
  const n = x.length
  return {
    buildings: {
      count: n,
      x: f(x),
      y: f(y),
      width: f(x.map(() => 10)),
      depth: f(x.map(() => 10)),
      height: f(x.map(() => 10)),
      rotation: f(x.map(() => 0)),
    },
    roads: { segments: new Float32Array(0), kind: new Uint8Array(0) },
    residents: {
      count: 0,
      x: f([]),
      y: f([]),
      z: f([]),
      building: new Uint32Array(0),
    },
    stores: { count: 0, x: f([]), y: f([]), radius: f([]) },
    ...(bounds && { bounds }),
  }
}

describe('city extent', () => {
  it('leaves the toy city exactly where the stage was tuned', () => {
    expect(TOY_EXTENT.scale).toBe(1)
    expect(fitPoint(-560, -260, TOY_EXTENT)).toEqual([-560, -260])
  })

  it('prefers the pack bbox, and falls back to the buildings', () => {
    const bounds = { minX: -10, minY: -20, maxX: 30, maxY: 40 }
    expect(cityBounds(cityWith([0], [0], bounds))).toBe(bounds)
    expect(cityBounds(cityWith([-5, 15], [8, -2]))).toEqual({
      minX: -5,
      minY: -2,
      maxX: 15,
      maxY: 8,
    })
    expect(cityBounds(cityWith([], []))).toBe(TOY_BOUNDS)
  })

  it('scales by the longer side and maps pack north to three.js -z', () => {
    const extent = extentOf({ minX: -13848, minY: -12178, maxX: 13832, maxY: 12186 })
    expect(extent.size).toBe(27680)
    expect(extent.scale).toBeCloseTo(27680 / 7450)
    expect(extent.centreX).toBe(-8)
    expect(extent.centreZ).toBe(-4)
  })

  it('moves a toy-tuned point with the city centre and stretches it with the scale', () => {
    const doubled = extentOf({ minX: -7472, minY: -6200, maxX: 7428, maxY: 6920 })
    expect(doubled.scale).toBeCloseTo(2)
    const [x, z] = fitPoint(TOY_EXTENT.centreX + 100, TOY_EXTENT.centreZ - 50, doubled)
    expect(x).toBeCloseTo(doubled.centreX + 200)
    expect(z).toBeCloseTo(doubled.centreZ - 100)
  })
})
