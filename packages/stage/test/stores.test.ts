import {
  type BufferAttribute,
  type BufferGeometry,
  Group,
  type InstancedBufferGeometry,
  type Material,
  type Mesh,
} from 'three/webgpu'
import { describe, expect, it } from 'vitest'
import type { StageCity, StageTape } from '../src/inputs'
import { Fog } from '../src/layers/fog'
import { beamShape, StoresLayer, storeCentres, unitRing } from '../src/layers/stores'

const stores: StageCity['stores'] = {
  count: 2,
  x: Float32Array.of(100, -300),
  y: Float32Array.of(200, 50),
  radius: Float32Array.of(900, 1200),
}

const stockouts: StageTape['stores'] = {
  stockoutOffsets: Uint32Array.of(0, 1, 2),
  stockoutIntervals: Uint32Array.of(60, 120, 90, 200),
}

function layer() {
  return new StoresLayer(stores, new Fog())
}

const geometries = (stores: StoresLayer) =>
  stores.object.children.map((child) => (child as Mesh).geometry as InstancedBufferGeometry)

const stockoutAttribute = (stores: StoresLayer) =>
  geometries(stores)[0]?.getAttribute('stockout') as BufferAttribute
const outOfStock = (stores: StoresLayer) => Array.from(stockoutAttribute(stores).array)

describe('storeCentres', () => {
  it('maps pack (x, y) to three.js (x, 0, -y)', () => {
    expect(Array.from(storeCentres(stores))).toEqual([100, 0, -200, -300, 0, -50])
  })
})

describe('beamShape', () => {
  it('is an open tube standing on the ground, with no caps', () => {
    const tube = beamShape()
    tube.computeBoundingBox()
    expect(tube.boundingBox?.min.y).toBeCloseTo(0)
    expect(tube.boundingBox?.max.y).toBeCloseTo(520)
    const normals = tube.getAttribute('normal') as BufferAttribute
    for (let i = 0; i < normals.count; i++) expect(normals.getY(i)).toBeCloseTo(0)
  })
})

describe('unitRing', () => {
  it('is a closed unit circle of segment pairs at the ring lift', () => {
    const position = unitRing(4).getAttribute('position') as BufferAttribute
    expect(position.count).toBe(8)
    const points = Array.from({ length: 8 }, (_, i) => [
      position.getX(i),
      position.getY(i),
      position.getZ(i),
    ])
    for (const [x = 0, y, z = 0] of points) {
      expect(Math.hypot(x, z)).toBeCloseTo(1)
      expect(y).toBe(points[0]?.[1])
    }
    expect(points[7]?.[0]).toBeCloseTo(points[0]?.[0] ?? Number.NaN)
    expect(points[7]?.[2]).toBeCloseTo(points[0]?.[2] ?? Number.NaN)
  })
})

describe('StoresLayer', () => {
  it('draws beams, base glows and rings as three instanced draws, one instance per store', () => {
    const drawn = geometries(layer())
    expect(drawn).toHaveLength(3)
    for (const geometry of drawn) {
      expect(geometry.isInstancedBufferGeometry).toBe(true)
      expect(geometry.instanceCount).toBe(2)
    }
  })

  it('shares one set of store attributes across the three draws', () => {
    const [beams, bases, rings] = geometries(layer())
    for (const name of ['centre', 'stockout', 'radius']) {
      const attribute = beams?.getAttribute(name)
      expect(
        (attribute as { isInstancedBufferAttribute?: boolean }).isInstancedBufferAttribute,
      ).toBe(true)
      expect(bases?.getAttribute(name)).toBe(attribute)
      expect(rings?.getAttribute(name)).toBe(attribute)
    }
    const radius = beams?.getAttribute('radius') as BufferAttribute | undefined
    expect(Array.from(radius?.array ?? [])).toEqual([900, 1200])
  })

  it('starts with every store in stock', () => {
    expect(outOfStock(layer())).toEqual([0, 0])
  })

  it('reuploads the stockout attribute only when a store changes state', () => {
    const stores = layer()
    const uploads = () => stockoutAttribute(stores).version
    const before = uploads()
    stores.showStockouts(stockouts, 30)
    expect(uploads()).toBe(before)
    stores.showStockouts(stockouts, 60)
    expect(outOfStock(stores)).toEqual([1, 0])
    expect(uploads()).toBe(before + 1)
    stores.showStockouts(stockouts, 80)
    expect(uploads()).toBe(before + 1)
    stores.showStockouts(stockouts, 100)
    expect(outOfStock(stores)).toEqual([1, 1])
    stores.showStockouts(stockouts, 120)
    expect(outOfStock(stores)).toEqual([0, 1])
    expect(uploads()).toBe(before + 3)
  })

  it('frees its geometries and materials and leaves the scene on dispose', () => {
    const stores = layer()
    const freed: string[] = []
    for (const child of stores.object.children) {
      const { geometry, material } = child as Mesh<BufferGeometry, Material>
      geometry.addEventListener('dispose', () => freed.push('geometry'))
      material.addEventListener('dispose', () => freed.push('material'))
    }
    new Group().add(stores.object)
    stores.dispose()
    expect(freed.filter((kind) => kind === 'geometry')).toHaveLength(3)
    expect(freed.filter((kind) => kind === 'material')).toHaveLength(3)
    expect(stores.object.parent).toBeNull()
  })
})
