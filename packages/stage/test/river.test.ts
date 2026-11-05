import { uniform } from 'three/tsl'
import { describe, expect, it } from 'vitest'
import { Fog } from '../src/layers/fog'
import { createRiver, waterMesh } from '../src/layers/river'

const square = {
  offsets: Uint32Array.of(0, 4),
  x: Float32Array.of(0, 10, 10, 0),
  y: Float32Array.of(0, 0, 20, 20),
}

describe('waterMesh', () => {
  it('lays a ring flat at (x, 0, -y) and fills it', () => {
    const { positions, indices } = waterMesh(square)
    expect(Array.from(positions)).toEqual([0, 0, -0, 10, 0, -0, 10, 0, -20, 0, 0, -20])
    expect(indices).toHaveLength(6)
  })

  it('keeps each polygon on its own vertices', () => {
    const { positions, indices } = waterMesh({
      offsets: Uint32Array.of(0, 4, 7),
      x: Float32Array.of(0, 10, 10, 0, 50, 60, 55),
      y: Float32Array.of(0, 0, 20, 20, 5, 5, 15),
    })
    expect(positions).toHaveLength(7 * 3)
    expect(Array.from(indices.slice(6)).sort()).toEqual([4, 5, 6])
  })

  it('does not count a closing point that repeats the first', () => {
    const { positions, indices } = waterMesh({
      offsets: Uint32Array.of(0, 5),
      x: Float32Array.of(0, 10, 10, 0, 0),
      y: Float32Array.of(0, 0, 20, 20, 0),
    })
    expect(positions).toHaveLength(4 * 3)
    expect(Math.max(...indices)).toBe(3)
  })

  it('draws nothing for a ring too short to enclose water', () => {
    const { indices } = waterMesh({
      offsets: Uint32Array.of(0, 2),
      x: Float32Array.of(0, 10),
      y: Float32Array.of(0, 0),
    })
    expect(indices).toHaveLength(0)
  })

  it('fills a bending ribbon without gaps', () => {
    const samples = 40
    const x = new Float32Array(samples * 2)
    const y = new Float32Array(samples * 2)
    for (let s = 0; s < samples; s++) {
      const centre = 200 * Math.sin(s / 6)
      x[s] = centre - 90
      x[samples * 2 - 1 - s] = centre + 90
      y[s] = -s * 30
      y[samples * 2 - 1 - s] = -s * 30
    }
    const { positions, indices } = waterMesh({ offsets: Uint32Array.of(0, samples * 2), x, y })
    expect(indices).toHaveLength((samples * 2 - 2) * 3)
    let area = 0
    for (let i = 0; i < indices.length; i += 3) {
      const [a, b, c] = [indices[i] ?? 0, indices[i + 1] ?? 0, indices[i + 2] ?? 0]
      const ax = positions[a * 3] ?? 0
      const az = positions[a * 3 + 2] ?? 0
      const ux = (positions[b * 3] ?? 0) - ax
      const uz = (positions[b * 3 + 2] ?? 0) - az
      const vx = (positions[c * 3] ?? 0) - ax
      const vz = (positions[c * 3 + 2] ?? 0) - az
      area += Math.abs(ux * vz - uz * vx) / 2
    }
    expect(area).toBeCloseTo(180 * 30 * (samples - 1), 0)
  })
})

describe('waterMesh with a longest edge', () => {
  const wide = {
    offsets: Uint32Array.of(0, 4),
    x: Float32Array.of(0, 200, 200, 0),
    y: Float32Array.of(0, 0, 1000, 1000),
  }
  const { positions, banks, indices } = waterMesh(wide, 40)
  const at = (vertex: number) => [positions[vertex * 3] ?? 0, positions[vertex * 3 + 2] ?? 0]

  function eachEdge(visit: (a: number, b: number) => void) {
    for (let i = 0; i < indices.length; i += 3) {
      const [a = 0, b = 0, c = 0] = indices.subarray(i, i + 3)
      visit(a, b)
      visit(b, c)
      visit(c, a)
    }
  }

  it('leaves no edge too long', () => {
    eachEdge((a, b) => {
      const [ax = 0, az = 0] = at(a)
      const [bx = 0, bz = 0] = at(b)
      expect(Math.hypot(ax - bx, az - bz)).toBeLessThanOrEqual(40)
    })
  })

  it('leaves no cracks: an edge on one triangle only lies on the bank', () => {
    const uses = new Map<number, number>()
    eachEdge((a, b) => {
      const key = Math.min(a, b) * 2 ** 26 + Math.max(a, b)
      uses.set(key, (uses.get(key) ?? 0) + 1)
    })
    for (const [key, count] of uses) {
      expect(count).toBeLessThanOrEqual(2)
      if (count === 1) {
        const a = Math.floor(key / 2 ** 26)
        const b = key % 2 ** 26
        expect(banks[a]).toBeCloseTo(0)
        expect(banks[b]).toBeCloseTo(0)
      }
    }
  })

  it('still covers the same water', () => {
    let area = 0
    for (let i = 0; i < indices.length; i += 3) {
      const [a = 0, b = 0, c = 0] = indices.subarray(i, i + 3)
      const [ax = 0, az = 0] = at(a)
      const [bx = 0, bz = 0] = at(b)
      const [cx = 0, cz = 0] = at(c)
      area += Math.abs((bx - ax) * (cz - az) - (bz - az) * (cx - ax)) / 2
    }
    expect(area).toBeCloseTo(200 * 1000, 0)
  })

  it('gives each vertex its distance to the nearest bank', () => {
    let deepest = 0
    banks.forEach((bank, vertex) => {
      const [x = 0, z = 0] = at(vertex)
      const expected = Math.min(x, 200 - x, -z, 1000 + z)
      expect(bank).toBeCloseTo(expected, 3)
      deepest = Math.max(deepest, bank)
    })
    expect(deepest).toBeGreaterThan(80)
  })

  it('keeps the ring on the bank', () => {
    expect(Array.from(banks.subarray(0, 4))).toEqual([0, 0, 0, 0])
  })
})

describe('createRiver', () => {
  const inputs = [new Fog(), uniform(0), uniform(0), false] as const

  it('makes no layer without water', () => {
    expect(createRiver(undefined, ...inputs)).toHaveLength(0)
    const empty = { offsets: Uint32Array.of(0), x: new Float32Array(0), y: new Float32Array(0) }
    expect(createRiver(empty, ...inputs)).toHaveLength(0)
  })

  it('makes one layer for any number of polygons', () => {
    expect(createRiver(square, ...inputs)).toHaveLength(1)
  })
})
