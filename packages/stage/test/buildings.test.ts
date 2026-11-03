import { Matrix4, Vector3 } from 'three/webgpu'
import { describe, expect, it } from 'vitest'
import type { StageCity } from '../src/inputs'
import { buildingMatrix } from '../src/layers/buildings'

// One building 40 m wide and 10 m deep, turned 30 degrees counter-clockwise from east.
const ROTATION = Math.PI / 6
const buildings: StageCity['buildings'] = {
  count: 1,
  x: Float32Array.of(100),
  y: Float32Array.of(200),
  width: Float32Array.of(40),
  depth: Float32Array.of(10),
  height: Float32Array.of(25),
  rotation: Float32Array.of(ROTATION),
}

function toPack(point: Vector3): { x: number; y: number; z: number } {
  return { x: point.x, y: -point.z, z: point.y }
}

describe('buildingMatrix', () => {
  const matrix = buildingMatrix(buildings, 0, new Matrix4())

  it('stands the box on the ground at the pack position, north towards -z', () => {
    const base = toPack(new Vector3(0, 0, 0).applyMatrix4(matrix))
    expect(base.x).toBeCloseTo(100)
    expect(base.y).toBeCloseTo(200)
    expect(base.z).toBeCloseTo(0)
    const top = toPack(new Vector3(0, 1, 0).applyMatrix4(matrix))
    expect(top.z).toBeCloseTo(25)
  })

  it('turns the width axis counter-clockwise from east in pack axes', () => {
    const east = toPack(new Vector3(0.5, 0, 0).applyMatrix4(matrix))
    expect(east.x).toBeCloseTo(100 + 20 * Math.cos(ROTATION))
    expect(east.y).toBeCloseTo(200 + 20 * Math.sin(ROTATION))
    const north = toPack(new Vector3(0, 0, -0.5).applyMatrix4(matrix))
    expect(north.x).toBeCloseTo(100 - 5 * Math.sin(ROTATION))
    expect(north.y).toBeCloseTo(200 + 5 * Math.cos(ROTATION))
  })
})
