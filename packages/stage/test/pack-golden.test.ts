import { readFileSync } from 'node:fs'
import { blake3 } from '@noble/hashes/blake3.js'
import { bytesToHex } from '@noble/hashes/utils.js'
import { decompressZstd } from '@twinlight/tape'
import { beforeAll, describe, expect, it } from 'vitest'
import type { StageCity } from '../src/inputs'
import { waterMesh } from '../src/layers/river'
import {
  decodeDetailBuildings,
  decodePack,
  fetchPack,
  PACK_FILES,
  type PackFiles,
} from '../src/pack'

// Python writes fixtures/pack/v2/small and its probes (`just goldens`); this reader must agree.

interface BuildingProbe {
  index: number
  x_m: number
  y_m: number
  width_m: number
  depth_m: number
  height_m: number
  rotation_rad: number
}

interface Probes {
  pack_hash: string
  tile_start: { tile: number; start: number }[]
  buildings: BuildingProbe[]
  detail: BuildingProbe[]
  roads: { road: number; class: number; points: [number, number][] }[]
  residents: {
    index: number
    qx: number
    qy: number
    z_m: number
    building: number
  }[]
  river: { polygon: number; rings: number[]; points: [number, number][][] }[]
  stores: { index: number; x: number; y: number; radius_m: number }[]
}

interface Manifest {
  format: string
  version: number
  bbox: [number, number, number, number]
  compression: 'zstd'
  files: Record<string, string>
  counts: {
    buildings: number
    buildingTileLevel: number
    detailBuildings: number
    residents: number
    stores: number
    roads: number
    roadPoints: number
    riverPolygons: number
    riverRings: number
    riverPoints: number
  }
}

const FIXTURE = new URL('../../../fixtures/pack/v2/small/', import.meta.url)
const read = (name: string) => new Uint8Array(readFileSync(new URL(name, FIXTURE)))
const parse = <T>(name: string) => JSON.parse(new TextDecoder().decode(read(name))) as T

const manifestBytes = read('manifest.json')
const manifest = parse<Manifest>('manifest.json')
const probes = parse<Probes>('probes.json')
const [minX, minY, maxX, maxY] = manifest.bbox
const metresX = (q: number) => minX + (q / 65535) * (maxX - minX)
const metresY = (q: number) => minY + (q / 65535) * (maxY - minY)

// Float32 positions, a quantised step of about 0.09 m here: well under one step.
const POSITION_DIGITS = 2

const plain = new Map<string, ArrayBuffer>()
let city: StageCity

beforeAll(async () => {
  for (const name of Object.keys(manifest.files)) {
    const bytes = await decompressZstd(read(name))
    plain.set(name, bytes.slice().buffer)
  }
  const files = Object.fromEntries(PACK_FILES.map((name) => [name, plain.get(name)])) as PackFiles
  city = decodePack(manifest, files)
})

function expectBuilding(buildings: StageCity['buildings'], probe: BuildingProbe) {
  const at = probe.index
  expect(buildings.x[at], `x of ${at}`).toBeCloseTo(probe.x_m, POSITION_DIGITS)
  expect(buildings.y[at], `y of ${at}`).toBeCloseTo(probe.y_m, POSITION_DIGITS)
  expect(buildings.width[at], `width of ${at}`).toBeCloseTo(probe.width_m, 3)
  expect(buildings.depth[at], `depth of ${at}`).toBeCloseTo(probe.depth_m, 3)
  expect(buildings.height[at], `height of ${at}`).toBeCloseTo(probe.height_m, 3)
  expect(buildings.rotation[at], `rotation of ${at}`).toBeCloseTo(probe.rotation_rad, 5)
}

describe('the v2 golden pack', () => {
  it('has the file hashes its manifest claims', () => {
    expect(plain.size).toBeGreaterThan(0)
    for (const [name, buffer] of plain) {
      expect(bytesToHex(blake3(new Uint8Array(buffer))), name).toBe(manifest.files[name])
    }
  })

  it('has the pack hash its probes claim', () => {
    expect(bytesToHex(blake3(manifestBytes))).toBe(probes.pack_hash)
  })

  it('agrees on every probed building', () => {
    expect(city.buildings.count).toBe(manifest.counts.buildings)
    expect(probes.buildings.length).toBeGreaterThan(0)
    for (const probe of probes.buildings) expectBuilding(city.buildings, probe)
  })

  it('agrees on the tile table, at the pack own level and with empty tiles', () => {
    const { tileStart } = city.buildings
    expect(tileStart).toHaveLength(4 ** manifest.counts.buildingTileLevel + 1)
    expect(tileStart?.at(-1)).toBe(manifest.counts.buildings)
    expect(probes.tile_start.length).toBeGreaterThan(0)
    for (const { tile, start } of probes.tile_start)
      expect(tileStart?.[tile], `tile ${tile}`).toBe(start)
    const tiles = Array.from(tileStart ?? [])
    expect(tiles.some((start, tile) => start === tiles[tile + 1])).toBe(true)
  })

  it('agrees on every probed detail footprint', () => {
    const detail = decodeDetailBuildings(
      plain.get('buildings_detail.bin') as ArrayBuffer,
      manifest.counts.detailBuildings,
      manifest.bbox,
    )
    expect(detail.count).toBe(manifest.counts.detailBuildings)
    expect(detail.tileStart).toBeUndefined()
    expect(probes.detail.length).toBeGreaterThan(0)
    for (const probe of probes.detail) expectBuilding(detail, probe)
  })

  it('agrees on every probed road, across jumps that wrap', () => {
    // The stage keeps segments, so each road's run is found from the file's own point counts.
    const points = new Uint16Array(plain.get('roads.bin') as ArrayBuffer, 0, manifest.counts.roads)
    let total = 0
    const firstSegment = Array.from(points, (count) => {
      const at = total
      total += Math.max(0, count - 1)
      return at
    })
    expect(city.roads.kind).toHaveLength(total)
    expect(city.roads.segments).toHaveLength(total * 4)
    let wrapped = 0
    let singlePoint = 0
    for (const probe of probes.roads) {
      const start = firstSegment[probe.road] ?? 0
      expect(points[probe.road], `points of road ${probe.road}`).toBe(probe.points.length)
      if (probe.points.length < 2) singlePoint++
      probe.points.forEach(([qx, qy], p) => {
        const prior = probe.points[p - 1]
        if (prior && Math.abs(qx - prior[0]) > 32767) wrapped++
        const segment = start + Math.max(p - 1, 0)
        const offset = p === 0 ? 0 : 2
        if (probe.points.length < 2) return
        const { segments } = city.roads
        expect(segments[segment * 4 + offset], `x ${probe.road}/${p}`).toBeCloseTo(
          metresX(qx),
          POSITION_DIGITS,
        )
        expect(segments[segment * 4 + offset + 1], `y ${probe.road}/${p}`).toBeCloseTo(
          metresY(qy),
          POSITION_DIGITS,
        )
      })
      if (probe.points.length > 1) expect(city.roads.kind[start]).toBe(probe.class)
    }
    expect(wrapped, 'the fixture exercises the wrap').toBeGreaterThan(0)
    expect(singlePoint, 'the fixture has a road of one point').toBeGreaterThan(0)
  })

  it('agrees on every probed resident', () => {
    expect(probes.residents.length).toBeGreaterThan(0)
    for (const probe of probes.residents) {
      const at = probe.index
      expect(city.residents.x[at]).toBeCloseTo(metresX(probe.qx), POSITION_DIGITS)
      expect(city.residents.y[at]).toBeCloseTo(metresY(probe.qy), POSITION_DIGITS)
      expect(city.residents.z[at]).toBeCloseTo(probe.z_m, 4)
      expect(city.residents.building[at]).toBe(probe.building)
    }
  })

  it('agrees exactly on every water ring, holes included', () => {
    const river = city.river
    expect(river?.rings).toHaveLength(manifest.counts.riverPolygons)
    expect(river?.offsets).toHaveLength(manifest.counts.riverRings + 1)
    let ring = 0
    const firstRing = Array.from(river?.rings ?? [], (count) => {
      const at = ring
      ring += count
      return at
    })
    expect(
      probes.river.some((probe) => probe.rings.length > 1),
      'a hole is probed',
    ).toBe(true)
    for (const probe of probes.river) {
      expect(river?.rings?.[probe.polygon]).toBe(probe.rings.length)
      probe.points.forEach((points, r) => {
        const start = river?.offsets[(firstRing[probe.polygon] ?? 0) + r] ?? 0
        expect(points).toHaveLength(probe.rings[r] ?? 0)
        points.forEach(([x, y], p) => {
          expect(river?.x[start + p]).toBe(x)
          expect(river?.y[start + p]).toBe(y)
        })
      })
    }
  })

  it('keeps its open rings open and triangulates them, holes included', () => {
    const river = city.river
    for (let ring = 0; ring < manifest.counts.riverRings; ring++) {
      const first = river?.offsets[ring] ?? 0
      const last = (river?.offsets[ring + 1] ?? 0) - 1
      const closed = river?.x[first] === river?.x[last] && river?.y[first] === river?.y[last]
      expect(closed, `ring ${ring} repeats its first point`).toBe(false)
    }
    expect(waterMesh(river).indices.length).toBeGreaterThan(0)
  })

  it('agrees exactly on every store', () => {
    expect(probes.stores.length).toBe(city.stores.count)
    for (const probe of probes.stores) {
      expect(city.stores.x[probe.index]).toBe(probe.x)
      expect(city.stores.y[probe.index]).toBe(probe.y)
      expect(city.stores.radius[probe.index]).toBe(probe.radius_m)
    }
  })

  it('loads whole through fetchPack with the real decoder', async () => {
    const load = (async (url: string) => {
      const name = url.split('/').at(-1) ?? ''
      return new Response(read(name))
    }) as typeof fetch
    const fetched = await fetchPack('/packs/small', load, async (compressed) => {
      const bytes = await decompressZstd(new Uint8Array(compressed))
      return bytes.slice().buffer
    })
    expect(Array.from(fetched.buildings.x)).toEqual(Array.from(city.buildings.x))
    expect(fetched.bounds).toEqual({ minX, minY, maxX, maxY })
  })
})
