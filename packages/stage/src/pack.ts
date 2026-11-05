import type { StageCity } from './inputs'

/** The parts of a v1 city pack manifest (ARCHITECTURE.md section 5) the stage reads. */
export interface PackManifest {
  format: string
  version: number
  /** minX, minY, maxX, maxY in local metres. */
  bbox?: [number, number, number, number]
  counts: {
    buildings: number
    residents: number
    stores: number
    roads: number
    roadPoints: number
    riverPolygons: number
    riverPoints: number
  }
}

export const PACK_FILES = [
  'buildings.bin',
  'roads.bin',
  'river.bin',
  'residents.bin',
  'stores.bin',
] as const
export type PackFiles = Record<(typeof PACK_FILES)[number], ArrayBuffer>

export function decodePack(manifest: PackManifest, files: PackFiles): StageCity {
  if (manifest.format !== 'twinlight-pack' || manifest.version !== 1) {
    throw new Error(`Unsupported city pack ${manifest.format} v${manifest.version}`)
  }
  const { counts, bbox } = manifest
  return {
    buildings: decodeBuildings(files['buildings.bin'], counts.buildings),
    roads: decodeRoads(files['roads.bin'], counts.roads, counts.roadPoints),
    river: decodeRiver(files['river.bin'], counts.riverPolygons, counts.riverPoints),
    residents: decodeResidents(files['residents.bin'], counts.residents),
    stores: decodeStores(files['stores.bin'], counts.stores),
    ...(bbox && { bounds: { minX: bbox[0], minY: bbox[1], maxX: bbox[2], maxY: bbox[3] } }),
  }
}

export async function fetchPack(baseUrl: string, load: typeof fetch = fetch): Promise<StageCity> {
  const base = baseUrl.replace(/\/$/, '')
  const get = async (file: string) => {
    const response = await load(`${base}/${file}`)
    if (!response.ok) throw new Error(`Could not load ${file} from ${base} (${response.status})`)
    return response
  }
  const [manifest, ...buffers] = await Promise.all([
    get('manifest.json').then((response) => response.json() as Promise<PackManifest>),
    ...PACK_FILES.map((file) => get(file).then((response) => response.arrayBuffer())),
  ])
  const files = Object.fromEntries(PACK_FILES.map((file, i) => [file, buffers[i]])) as PackFiles
  return decodePack(manifest, files)
}

// Every column starts on an 8-byte boundary and is zero-padded to a multiple of 8, so each one
// is a typed-array view into the file, and the file size is the sum of the padded columns.
class Columns {
  readonly #buffer: ArrayBuffer
  readonly #file: string
  #offset = 0

  constructor(buffer: ArrayBuffer, file: string) {
    this.#buffer = buffer
    this.#file = file
  }

  f32(n: number): Float32Array {
    return new Float32Array(this.#buffer, this.#take(n * 4), n)
  }

  u32(n: number): Uint32Array {
    return new Uint32Array(this.#buffer, this.#take(n * 4), n)
  }

  u16(n: number): Uint16Array {
    return new Uint16Array(this.#buffer, this.#take(n * 2), n)
  }

  u8(n: number): Uint8Array {
    return new Uint8Array(this.#buffer, this.#take(n), n)
  }

  end(): void {
    if (this.#offset !== this.#buffer.byteLength) {
      throw new Error(
        `${this.#file} is ${this.#buffer.byteLength} bytes but its columns take ${this.#offset}`,
      )
    }
  }

  #take(bytes: number): number {
    const at = this.#offset
    this.#offset += Math.ceil(bytes / 8) * 8
    if (this.#offset > this.#buffer.byteLength) throw new Error(`${this.#file} is truncated`)
    return at
  }
}

function decodeBuildings(buffer: ArrayBuffer, count: number): StageCity['buildings'] {
  const c = new Columns(buffer, 'buildings.bin')
  const x = c.f32(count)
  const y = c.f32(count)
  const width = c.f32(count)
  const depth = c.f32(count)
  const height = c.f32(count)
  const rotation = c.f32(count)
  c.u16(count) // district
  c.end()
  return { count, x, y, width, depth, height, rotation }
}

function decodeResidents(buffer: ArrayBuffer, count: number): StageCity['residents'] {
  const c = new Columns(buffer, 'residents.bin')
  const x = c.f32(count)
  const y = c.f32(count)
  const z = c.f32(count)
  const building = c.u32(count)
  c.u32(count) // hex
  c.u16(count) // segment
  c.u32(count) // persona_ref
  c.f32(count) // weight
  c.end()
  return { count, x, y, z, building }
}

function decodeStores(buffer: ArrayBuffer, count: number): StageCity['stores'] {
  const c = new Columns(buffer, 'stores.bin')
  const x = c.f32(count)
  const y = c.f32(count)
  c.u32(count) // hex
  const radius = c.f32(count)
  c.u16(count) // district
  c.end()
  return { count, x, y, radius }
}

// Roads are CSR polylines; the stage draws each pair of consecutive points as a segment.
function decodeRoads(buffer: ArrayBuffer, roads: number, points: number): StageCity['roads'] {
  const c = new Columns(buffer, 'roads.bin')
  const offsets = c.u32(roads + 1)
  const x = c.f32(points)
  const y = c.f32(points)
  const roadClass = c.u8(roads)
  c.end()
  checkOffsets('roads.bin', 'road', offsets, points)
  let total = 0
  for (let r = 0; r < roads; r++) {
    total += Math.max(0, (offsets[r + 1] ?? 0) - (offsets[r] ?? 0) - 1)
  }
  const segments = new Float32Array(total * 4)
  const kind = new Uint8Array(total)
  let s = 0
  for (let r = 0; r < roads; r++) {
    const last = (offsets[r + 1] ?? 0) - 1
    for (let p = offsets[r] ?? 0; p < last; p++, s++) {
      segments[s * 4] = x[p] ?? 0
      segments[s * 4 + 1] = y[p] ?? 0
      segments[s * 4 + 2] = x[p + 1] ?? 0
      segments[s * 4 + 3] = y[p + 1] ?? 0
      kind[s] = roadClass[r] ?? 0
    }
  }
  return { segments, kind }
}

function decodeRiver(
  buffer: ArrayBuffer,
  polygons: number,
  points: number,
): NonNullable<StageCity['river']> {
  const c = new Columns(buffer, 'river.bin')
  const offsets = c.u32(polygons + 1)
  const x = c.f32(points)
  const y = c.f32(points)
  c.u8(polygons) // kind
  c.end()
  checkOffsets('river.bin', 'river', offsets, points)
  return { offsets, x, y }
}

function checkOffsets(file: string, what: string, offsets: Uint32Array, points: number): void {
  const count = offsets.length - 1
  if (offsets[0] !== 0 || offsets[count] !== points) {
    throw new Error(`${file} offsets must run from 0 to ${points} ${what} points`)
  }
  for (let i = 0; i < count; i++) {
    if ((offsets[i + 1] ?? 0) < (offsets[i] ?? 0)) {
      throw new Error(`${file} offsets go backwards at ${what} ${i}`)
    }
  }
}
