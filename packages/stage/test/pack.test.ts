import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { ROAD_KIND } from '../src/inputs'
import { decodePack, fetchPack, PACK_FILES, type PackFiles, type PackManifest } from '../src/pack'

type Column = Float32Array | Uint32Array | Uint16Array | Uint8Array

// Writes columns as ARCHITECTURE.md section 5 v1 describes: each zero-padded to 8 bytes.
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

const manifest: PackManifest = {
  format: 'twinlight-pack',
  version: 1,
  counts: { buildings: 2, residents: 3, stores: 1, roads: 2, roadPoints: 5 },
}

function tinyPack(): PackFiles {
  return {
    'buildings.bin': file(
      Float32Array.of(10, -20),
      Float32Array.of(30, 40),
      Float32Array.of(12, 8),
      Float32Array.of(6, 9),
      Float32Array.of(25, 7),
      Float32Array.of(0.1, -0.2),
      Uint16Array.of(3, 4),
    ),
    'residents.bin': file(
      Float32Array.of(11, 12, -19),
      Float32Array.of(31, 32, 41),
      Float32Array.of(2.4, 5.7, 2.4),
      Uint32Array.of(0, 0, 1),
      Uint32Array.of(0, 0, 0),
      Uint16Array.of(1, 2, 3),
      Uint32Array.of(0xffffffff, 0xffffffff, 0xffffffff),
      Float32Array.of(1, 1, 1),
    ),
    'stores.bin': file(
      Float32Array.of(5),
      Float32Array.of(6),
      Uint32Array.of(0),
      Float32Array.of(900),
      Uint16Array.of(2),
    ),
    // A three-point arterial and a two-point bridge.
    'roads.bin': file(
      Uint32Array.of(0, 3, 5),
      Float32Array.of(0, 10, 20, 100, 110),
      Float32Array.of(0, 0, 5, -50, -50),
      Uint8Array.of(ROAD_KIND.arterial, ROAD_KIND.bridge),
    ),
  }
}

describe('decodePack', () => {
  const city = decodePack(manifest, tinyPack())

  it('reads buildings, residents and stores as pack-axis columns', () => {
    expect(city.buildings.count).toBe(2)
    expect(Array.from(city.buildings.y)).toEqual([30, 40])
    expect(city.buildings.rotation[1]).toBeCloseTo(-0.2)
    expect(Array.from(city.residents.building)).toEqual([0, 0, 1])
    expect(city.residents.z[1]).toBeCloseTo(5.7)
    expect(city.stores.radius[0]).toBe(900)
  })

  it('splits road polylines into segments that keep their class', () => {
    expect(Array.from(city.roads.segments)).toEqual([0, 0, 10, 0, 10, 0, 20, 5, 100, -50, 110, -50])
    expect(Array.from(city.roads.kind)).toEqual([
      ROAD_KIND.arterial,
      ROAD_KIND.arterial,
      ROAD_KIND.bridge,
    ])
  })

  it('rejects a file whose size does not match its columns', () => {
    const files = tinyPack()
    files['stores.bin'] = files['stores.bin'].slice(0, 32)
    expect(() => decodePack(manifest, files)).toThrow(/stores\.bin/)
    const longer = tinyPack()
    longer['residents.bin'] = file(Float32Array.of(...new Float32Array(40)))
    expect(() => decodePack(manifest, longer)).toThrow(/residents\.bin/)
  })

  it('draws nothing for roads with one point or none', () => {
    const files = tinyPack()
    files['roads.bin'] = file(
      Uint32Array.of(0, 0, 1, 3),
      Float32Array.of(5, 0, 10),
      Float32Array.of(5, 0, 0),
      Uint8Array.of(ROAD_KIND.street, ROAD_KIND.street, ROAD_KIND.bridge),
    )
    const roads = decodePack(
      { ...manifest, counts: { ...manifest.counts, roads: 3, roadPoints: 3 } },
      files,
    ).roads
    expect(Array.from(roads.segments)).toEqual([0, 0, 10, 0])
    expect(Array.from(roads.kind)).toEqual([ROAD_KIND.bridge])
  })

  it('rejects road offsets that do not cover the points in order', () => {
    const withOffsets = (offsets: Uint32Array) => {
      const files = tinyPack()
      files['roads.bin'] = file(
        offsets,
        Float32Array.of(0, 10, 20, 100, 110),
        Float32Array.of(0, 0, 5, -50, -50),
        Uint8Array.of(ROAD_KIND.arterial, ROAD_KIND.bridge),
      )
      return () => decodePack(manifest, files)
    }
    expect(withOffsets(Uint32Array.of(1, 3, 5))).toThrow(/from 0 to 5/)
    expect(withOffsets(Uint32Array.of(0, 3, 4))).toThrow(/from 0 to 5/)
    expect(withOffsets(Uint32Array.of(0, 6, 5))).toThrow(/backwards at road 1/)
  })

  it('rejects other formats and versions', () => {
    expect(() => decodePack({ ...manifest, version: 2 }, tinyPack())).toThrow(/v2/)
  })
})

describe('fetchPack', () => {
  it('fetches the manifest and the columns under one base url', async () => {
    const files = tinyPack()
    const requested: string[] = []
    const load = (async (url: string) => {
      requested.push(url)
      const name = url.split('/').at(-1) ?? ''
      if (name === 'manifest.json') return Response.json(manifest)
      return new Response(files[name as keyof PackFiles])
    }) as typeof fetch
    const city = await fetchPack('/packs/toy/', load)
    expect(city.residents.count).toBe(3)
    expect(requested.sort()).toEqual(
      ['manifest.json', ...PACK_FILES].map((name) => `/packs/toy/${name}`).sort(),
    )
  })
})

// A generated pack (`uv run python -m twl_engine.fixtures --out <dir>`); skipped unless provided.
const generated = process.env.TWL_TEST_PACK_DIR
describe.skipIf(!generated)('a generated toy pack', () => {
  it('decodes, and every resident stands on its building footprint', () => {
    const read = (name: string) => {
      const bytes = readFileSync(`${generated}/${name}`)
      return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    }
    const files = Object.fromEntries(PACK_FILES.map((name) => [name, read(name)])) as PackFiles
    const city = decodePack(JSON.parse(readFileSync(`${generated}/manifest.json`, 'utf8')), files)
    const { buildings: b, residents: r } = city
    for (let i = 0; i < r.count; i++) {
      const k = r.building[i] ?? 0
      const dx = (r.x[i] ?? 0) - (b.x[k] ?? 0)
      const dy = (r.y[i] ?? 0) - (b.y[k] ?? 0)
      const rot = b.rotation[k] ?? 0
      expect(Math.abs(dx * Math.cos(rot) + dy * Math.sin(rot))).toBeLessThanOrEqual(
        (b.width[k] ?? 0) / 2 + 1,
      )
      expect(Math.abs(-dx * Math.sin(rot) + dy * Math.cos(rot))).toBeLessThanOrEqual(
        (b.depth[k] ?? 0) / 2 + 1,
      )
    }
    expect(city.roads.kind.length).toBeGreaterThan(0)
  })
})
