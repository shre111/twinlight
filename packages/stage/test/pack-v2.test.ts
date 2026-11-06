import { describe, expect, it } from 'vitest'
import { decodePack, fetchPack, PACK_FILES, type PackFiles, type PackManifest } from '../src/pack'

// Encoded by hand from ARCHITECTURE.md section 5, so the reader is not checking itself. The
// cross-language goldens check it against the Python writer.

type Column = Float32Array | Uint32Array | Uint16Array | Uint8Array

function file(...columns: Column[]): ArrayBuffer {
  const padded = columns.map((column) => Math.ceil(column.byteLength / 8) * 8)
  const out = new Uint8Array(padded.reduce((sum, size) => sum + size, 0))
  let at = 0
  columns.forEach((column, i) => {
    out.set(new Uint8Array(column.buffer, column.byteOffset, column.byteLength), at)
    at += padded[i] ?? 0
  })
  return out.buffer
}

function bytePlanes(values: number[], bytes: number): Uint8Array {
  const out = new Uint8Array(values.length * bytes)
  values.forEach((value, i) => {
    for (let b = 0; b < bytes; b++) out[b * values.length + i] = (value >>> (8 * b)) & 0xff
  })
  return out
}

function spreadBits(value: number): number {
  let v = value & 0xffff
  v = (v | (v << 8)) & 0x00ff00ff
  v = (v | (v << 4)) & 0x0f0f0f0f
  v = (v | (v << 2)) & 0x33333333
  return (v | (v << 1)) & 0x55555555
}

const morton = (qx: number, qy: number) => (spreadBits(qx) | (spreadBits(qy) << 1)) >>> 0

function zigzag(delta: number): number {
  // Wrapped to signed 16 bits first, as the writer does.
  const wrapped = (delta << 16) >> 16
  return ((wrapped << 1) ^ (wrapped >> 31)) & 0xffff
}

const sizeCode = (metres: number) => Math.round(Math.sqrt(metres) * 12)

// Exactly 65535 units across, so a quantised unit is 1 m and positions read off directly.
const BBOX: [number, number, number, number] = [-30000, -20000, 35535, 45535]
const metresX = (q: number) => BBOX[0] + q
const metresY = (q: number) => BBOX[1] + q

const QX = [10, 30000, 65535]
const QY = [20, 100, 65535]
const CODES = QX.map((qx, i) => morton(qx, QY[i] ?? 0))
const ROAD_QX = [0, 100, 65000, 65100]
const ROAD_QY = [0, 50, 60000, 60000]

function roadPlanes(values: number[]): Uint8Array {
  const deltas = values.map((value, i) => zigzag(value - (values[i - 1] ?? 0)))
  return bytePlanes(deltas, 2)
}

const manifest: PackManifest = {
  format: 'twinlight-pack',
  version: 2,
  bbox: BBOX,
  counts: {
    buildings: 3,
    buildingTileLevel: 1,
    residents: 2,
    stores: 1,
    roads: 2,
    roadPoints: 4,
    riverPolygons: 2,
    riverRings: 3,
    riverPoints: 11,
  },
}

function pack(): PackFiles {
  const sorted = [...CODES].sort((a, b) => a - b)
  expect(sorted).toEqual(CODES)
  const deltas = CODES.map((code, i) => (code - (CODES[i - 1] ?? 0)) >>> 0)
  return {
    'buildings.bin': file(
      Uint32Array.of(0, 1, 2, 2, 3),
      bytePlanes(deltas, 4),
      Uint8Array.of(sizeCode(5), sizeCode(100), sizeCode(418)),
      Uint8Array.of(sizeCode(7), sizeCode(20), sizeCode(288)),
      Uint8Array.of(sizeCode(6), sizeCode(9), sizeCode(138)),
      Uint8Array.of(0, 128, 255),
      Uint8Array.of(1, 2, 3),
    ),
    'roads.bin': file(
      Uint16Array.of(2, 2),
      Uint8Array.of(0, 2),
      roadPlanes(ROAD_QX),
      roadPlanes(ROAD_QY),
    ),
    'residents.bin': file(
      Uint16Array.of(10, 65535),
      Uint16Array.of(20, 0),
      Uint8Array.of(0, 3),
      Uint32Array.of(0, 2),
      Uint8Array.of(1, 5),
    ),
    'stores.bin': file(
      Float32Array.of(5),
      Float32Array.of(6),
      Uint32Array.of(0),
      Float32Array.of(900),
      Uint16Array.of(2),
    ),
    'river.bin': file(
      Uint32Array.of(0, 4, 8, 11),
      Float32Array.of(0, 200, 200, 0, 80, 80, 120, 120, 500, 600, 550),
      Float32Array.of(0, 0, 200, 200, 80, 120, 120, 80, 0, 0, 90),
      Uint8Array.of(1, 0),
      Uint16Array.of(2, 1),
    ),
  }
}

describe('decodePack v2', () => {
  const city = decodePack(manifest, pack())

  it('undoes Morton deltas into positions inside the bbox', () => {
    expect(Array.from(city.buildings.x)).toEqual(QX.map(metresX))
    expect(Array.from(city.buildings.y)).toEqual(QY.map(metresY))
  })

  it('reads a full-scale Morton code, whose top bit would go negative as a signed int', () => {
    expect(CODES[2]).toBe(0xffffffff)
    expect(city.buildings.x[2]).toBe(BBOX[2])
    expect(city.buildings.y[2]).toBe(BBOX[3])
  })

  it('decodes sizes from the square-root code, with no clipping at the extremes', () => {
    expect(city.buildings.width[0]).toBeCloseTo(5, 0)
    expect(city.buildings.width[2]).toBeCloseTo(418, -1)
    expect(city.buildings.depth[2]).toBeCloseTo(288, -1)
    expect(city.buildings.height[2]).toBeCloseTo(138, 0)
    const code = sizeCode(100)
    expect(city.buildings.width[1]).toBeCloseTo((code / 12) ** 2, 3)
  })

  it('maps rotation codes across half a turn', () => {
    expect(city.buildings.rotation[0]).toBe(0)
    expect(city.buildings.rotation[1]).toBeCloseTo((128 / 255) * Math.PI, 5)
    expect(city.buildings.rotation[2]).toBeCloseTo(Math.PI, 5)
  })

  it('undoes road deltas across a jump too long for 16 bits, keeping road classes', () => {
    expect(Math.abs((ROAD_QX[2] ?? 0) - (ROAD_QX[1] ?? 0))).toBeGreaterThan(32767)
    expect(Array.from(city.roads.segments)).toEqual([
      metresX(0),
      metresY(0),
      metresX(100),
      metresY(50),
      metresX(65000),
      metresY(60000),
      metresX(65100),
      metresY(60000),
    ])
    expect(Array.from(city.roads.kind)).toEqual([0, 2])
  })

  it('reads residents from quantised positions and floors', () => {
    expect(city.residents.count).toBe(2)
    expect(Array.from(city.residents.x)).toEqual([metresX(10), metresX(65535)])
    expect(Array.from(city.residents.y)).toEqual([metresY(20), metresY(0)])
    expect(city.residents.z[0]).toBeCloseTo(1.6)
    expect(city.residents.z[1]).toBeCloseTo(11.2)
    expect(Array.from(city.residents.building)).toEqual([0, 2])
  })

  it('keeps which rings belong to which water polygon', () => {
    expect(Array.from(city.river?.offsets ?? [])).toEqual([0, 4, 8, 11])
    expect(Array.from(city.river?.rings ?? [])).toEqual([2, 1])
    expect(city.river?.x[8]).toBe(500)
  })

  it('scales quantised positions to a bbox that is neither square nor in whole units', () => {
    const [minX, minY, maxX, maxY] = [-13848, -12181, 13844, 12186]
    const files = pack()
    // 13107 is exactly a fifth of 65535, so the expected metres need no quantisation arithmetic.
    files['residents.bin'] = file(
      Uint16Array.of(13107, 65535),
      Uint16Array.of(52428, 0),
      Uint8Array.of(0, 0),
      Uint32Array.of(0, 0),
      Uint8Array.of(0, 0),
    )
    const scaled = decodePack({ ...manifest, bbox: [minX, minY, maxX, maxY] }, files)
    expect(scaled.residents.x[0]).toBeCloseTo(minX + (maxX - minX) / 5, 2)
    expect(scaled.residents.y[0]).toBeCloseTo(minY + ((maxY - minY) * 4) / 5, 2)
    expect(scaled.residents.x[1]).toBe(maxX)
    expect(scaled.residents.y[1]).toBe(minY)
    expect(scaled.buildings.x[2]).toBe(maxX)
    expect(scaled.buildings.y[2]).toBe(maxY)
  })

  it('carries the bbox as bounds', () => {
    expect(city.bounds).toEqual({ minX: -30000, minY: -20000, maxX: 35535, maxY: 45535 })
  })

  it('reads stores as before', () => {
    expect(city.stores.count).toBe(1)
    expect(city.stores.radius[0]).toBe(900)
  })
})

describe('decodePack v2 rejects what does not add up', () => {
  it('a tile table that does not cover the buildings', () => {
    const files = pack()
    const deltas = CODES.map((code, i) => (code - (CODES[i - 1] ?? 0)) >>> 0)
    files['buildings.bin'] = file(
      Uint32Array.of(0, 1, 2, 2, 2),
      bytePlanes(deltas, 4),
      new Uint8Array(3),
      new Uint8Array(3),
      new Uint8Array(3),
      new Uint8Array(3),
      new Uint8Array(3),
    )
    expect(() => decodePack(manifest, files)).toThrow(/buildings\.bin offsets must run from 0 to 3/)
  })

  it('a road list whose points do not add up', () => {
    const files = pack()
    files['roads.bin'] = file(
      Uint16Array.of(2, 1),
      Uint8Array.of(0, 2),
      roadPlanes(ROAD_QX),
      roadPlanes(ROAD_QY),
    )
    expect(() => decodePack(manifest, files)).toThrow(/roads\.bin offsets must run from 0 to 4/)
  })

  it('polygons that use the wrong number of rings', () => {
    const files = pack()
    files['river.bin'] = file(
      Uint32Array.of(0, 4, 8, 11),
      new Float32Array(11),
      new Float32Array(11),
      Uint8Array.of(1, 0),
      Uint16Array.of(1, 1),
    )
    expect(() => decodePack(manifest, files)).toThrow(/use 2 rings but it has 3/)
  })

  it('a file of the wrong size', () => {
    const files = pack()
    files['stores.bin'] = files['stores.bin'].slice(0, 8)
    expect(() => decodePack(manifest, files)).toThrow(/stores\.bin/)
  })

  it('a v2 manifest with no bbox, since positions cannot be read without one', () => {
    const { bbox: _bbox, ...noBox } = manifest
    expect(() => decodePack(noBox, pack())).toThrow(/Unsupported city pack/)
  })

  it('a version it does not know', () => {
    expect(() => decodePack({ ...manifest, version: 3 }, pack())).toThrow(/v3/)
  })
})

describe('fetchPack with compression', () => {
  const served = (compression: PackManifest['compression']) => {
    const files = pack()
    return (async (url: string) => {
      const name = url.split('/').at(-1) ?? ''
      if (name === 'manifest.json') return Response.json({ ...manifest, compression })
      return new Response(files[name as keyof PackFiles])
    }) as typeof fetch
  }

  it('runs every file through the decoder for a zstd pack', async () => {
    let calls = 0
    const city = await fetchPack('/packs/x', served('zstd'), (compressed) => {
      calls++
      return compressed
    })
    expect(calls).toBe(PACK_FILES.length)
    expect(city.buildings.count).toBe(3)
  })

  it('says so when a zstd pack has no decoder', async () => {
    await expect(fetchPack('/packs/x', served('zstd'))).rejects.toThrow(/zstd/)
  })

  it('takes a pack with no compression as it is', async () => {
    const city = await fetchPack('/packs/x', served(undefined))
    expect(city.buildings.count).toBe(3)
  })
})
