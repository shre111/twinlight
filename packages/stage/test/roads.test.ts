import type { BufferGeometry, LineSegments } from 'three/webgpu'
import { describe, expect, it } from 'vitest'
import { ROAD_KIND, type StageCity } from '../src/inputs'
import { Fog } from '../src/layers/fog'
import { createRoads } from '../src/layers/roads'

const roads: StageCity['roads'] = {
  segments: Float32Array.of(10, 20, 30, 40, -5, 100, 5, 120),
  kind: Uint8Array.of(ROAD_KIND.street, ROAD_KIND.bridge),
}

describe('createRoads', () => {
  const layers = createRoads(roads, new Fog())

  it('makes one layer per road class present and none for absent ones', () => {
    expect(layers).toHaveLength(2)
  })

  it('places segment ends at (x, lift, -y) with bridges lifted above streets', () => {
    const [street, bridge] = layers.map((layer) => {
      const geometry = (layer.object as LineSegments).geometry as BufferGeometry
      return Array.from(geometry.getAttribute('position').array)
    })
    expect(street?.filter((_, i) => i % 3 !== 1)).toEqual([10, -20, 30, -40])
    expect(bridge?.filter((_, i) => i % 3 !== 1)).toEqual([-5, -100, 5, -120])
    expect(bridge?.[1] ?? 0).toBeGreaterThan(street?.[1] ?? 0)
  })
})
