import type { StageCity } from './inputs'
import { Columns, checkOffsets, decodeStores, polylineSegments } from './pack-columns'
import { decodePackV1, type PackCountsV1 } from './pack-v1'

type Bbox = [number, number, number, number]

interface PackCountsV2 {
  buildings: number
  buildingTileLevel: number
  residents: number
  stores: number
  roads: number
  roadPoints: number
  riverPolygons: number
  riverRings: number
  riverPoints: number
}

/** The parts of a city pack manifest (ARCHITECTURE.md section 5) the stage reads. */
export interface PackManifest {
  format: string
  version: number
  /** minX, minY, maxX, maxY in local metres. */
  bbox?: Bbox
  counts: PackCountsV1 | PackCountsV2
  compression?: 'zstd'
}

export const PACK_FILES = [
  'buildings.bin',
  'roads.bin',
  'river.bin',
  'residents.bin',
  'stores.bin',
] as const
export type PackFiles = Record<(typeof PACK_FILES)[number], ArrayBuffer>

export type Decompress = (compressed: ArrayBuffer) => ArrayBuffer | Promise<ArrayBuffer>

export function decodePack(manifest: PackManifest, files: PackFiles): StageCity {
  if (manifest.format === 'twinlight-pack' && manifest.version === 1) {
    return { ...decodePackV1(manifest.counts as PackCountsV1, files), ...boundsOf(manifest.bbox) }
  }
  if (manifest.format === 'twinlight-pack' && manifest.version === 2 && manifest.bbox) {
    return decodePackV2(manifest.counts as PackCountsV2, manifest.bbox, files)
  }
  throw new Error(`Unsupported city pack ${manifest.format} v${manifest.version}`)
}

function boundsOf(bbox: Bbox | undefined): Pick<StageCity, 'bounds'> {
  return bbox ? { bounds: { minX: bbox[0], minY: bbox[1], maxX: bbox[2], maxY: bbox[3] } } : {}
}

export async function fetchPack(
  baseUrl: string,
  load: typeof fetch = fetch,
  decompress?: Decompress,
): Promise<StageCity> {
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
  if (manifest.compression === 'zstd' && !decompress) {
    throw new Error('This city pack is zstd compressed and no decoder was given')
  }
  const plain =
    manifest.compression === 'zstd' && decompress
      ? await Promise.all(buffers.map((buffer) => decompress(buffer)))
      : buffers
  const files = Object.fromEntries(PACK_FILES.map((file, i) => [file, plain[i]])) as PackFiles
  return decodePack(manifest, files)
}

const QUANTISED_MAX = 65535
const FLOOR_METRES = 3.2

function decodePackV2(counts: PackCountsV2, bbox: Bbox, files: PackFiles): StageCity {
  const quantised = quantisedAxes(bbox)
  return {
    buildings: decodeBuildings(
      files['buildings.bin'],
      counts.buildings,
      counts.buildingTileLevel,
      quantised,
    ),
    roads: decodeRoads(files['roads.bin'], counts.roads, counts.roadPoints, quantised),
    river: decodeRiver(
      files['river.bin'],
      counts.riverPolygons,
      counts.riverRings,
      counts.riverPoints,
    ),
    residents: decodeResidents(files['residents.bin'], counts.residents, quantised),
    stores: decodeStores(files['stores.bin'], counts.stores),
    ...boundsOf(bbox),
  }
}

type QuantisedAxes = ReturnType<typeof quantisedAxes>

function quantisedAxes([minX, minY, maxX, maxY]: Bbox) {
  const stepX = (maxX - minX) / QUANTISED_MAX
  const stepY = (maxY - minY) / QUANTISED_MAX
  return {
    x: (q: number) => minX + q * stepX,
    y: (q: number) => minY + q * stepY,
  }
}

// Every other bit of a Morton code: x is on the even bits and y on the odd ones.
function evenBits(code: number): number {
  let v = code & 0x55555555
  v = (v | (v >>> 1)) & 0x33333333
  v = (v | (v >>> 2)) & 0x0f0f0f0f
  v = (v | (v >>> 4)) & 0x00ff00ff
  return (v | (v >>> 8)) & 0x0000ffff
}

const SIZE_METRES = Float32Array.from({ length: 256 }, (_, value) => (value / 12) ** 2)
const ROTATION_RADIANS = Float32Array.from({ length: 256 }, (_, value) => (value / 255) * Math.PI)

function fromTable(values: Uint8Array, table: Float32Array): Float32Array {
  return Float32Array.from(values, (value) => table[value] ?? 0)
}

function decodeBuildings(
  buffer: ArrayBuffer,
  count: number,
  tileLevel: number,
  quantised: QuantisedAxes,
): StageCity['buildings'] {
  const c = new Columns(buffer, 'buildings.bin')
  const tileStart = c.u32(4 ** tileLevel + 1)
  const buildings = readBuildingRows(c, count, quantised)
  checkOffsets('buildings.bin', 'tile', tileStart, count)
  return { ...buildings, tileStart }
}

/** The optional detail file: the same columns as `buildings.bin`, with no tile table. */
export function decodeDetailBuildings(
  buffer: ArrayBuffer,
  count: number,
  bbox: Bbox,
): StageCity['buildings'] {
  const c = new Columns(buffer, 'buildings_detail.bin')
  return readBuildingRows(c, count, quantisedAxes(bbox))
}

function readBuildingRows(
  c: Columns,
  count: number,
  quantised: QuantisedAxes,
): StageCity['buildings'] {
  const deltas = c.u8(count * 4)
  const width = c.u8(count)
  const depth = c.u8(count)
  const height = c.u8(count)
  const rotation = c.u8(count)
  c.u8(count) // district
  c.end()
  const x = new Float32Array(count)
  const y = new Float32Array(count)
  let code = 0
  for (let i = 0; i < count; i++) {
    const delta =
      (deltas[i] ?? 0) |
      ((deltas[count + i] ?? 0) << 8) |
      ((deltas[2 * count + i] ?? 0) << 16) |
      ((deltas[3 * count + i] ?? 0) << 24)
    code = (code + delta) >>> 0
    x[i] = quantised.x(evenBits(code))
    y[i] = quantised.y(evenBits(code >>> 1))
  }
  return {
    count,
    x,
    y,
    width: fromTable(width, SIZE_METRES),
    depth: fromTable(depth, SIZE_METRES),
    height: fromTable(height, SIZE_METRES),
    rotation: fromTable(rotation, ROTATION_RADIANS),
  }
}

// The zigzag delta is wrapped to 16 bits by the writer, so positions add back modulo 65536.
function undoDeltas(planes: Uint8Array, count: number, toMetres: (q: number) => number) {
  const out = new Float32Array(count)
  let q = 0
  for (let i = 0; i < count; i++) {
    const zigzag = (planes[i] ?? 0) | ((planes[count + i] ?? 0) << 8)
    q = (q + ((zigzag >>> 1) ^ -(zigzag & 1))) & QUANTISED_MAX
    out[i] = toMetres(q)
  }
  return out
}

function decodeRoads(
  buffer: ArrayBuffer,
  roads: number,
  points: number,
  quantised: QuantisedAxes,
): StageCity['roads'] {
  const c = new Columns(buffer, 'roads.bin')
  const pointsPerRoad = c.u16(roads)
  const roadClass = c.u8(roads)
  const dx = c.u8(points * 2)
  const dy = c.u8(points * 2)
  c.end()
  const offsets = new Uint32Array(roads + 1)
  for (let r = 0; r < roads; r++) offsets[r + 1] = (offsets[r] ?? 0) + (pointsPerRoad[r] ?? 0)
  checkOffsets('roads.bin', 'road', offsets, points)
  return polylineSegments(
    offsets,
    undoDeltas(dx, points, quantised.x),
    undoDeltas(dy, points, quantised.y),
    roadClass,
  )
}

function decodeResidents(
  buffer: ArrayBuffer,
  count: number,
  quantised: QuantisedAxes,
): StageCity['residents'] {
  const c = new Columns(buffer, 'residents.bin')
  const x = c.u16(count)
  const y = c.u16(count)
  const floor = c.u8(count)
  const building = c.u32(count)
  c.u8(count) // segment
  c.end()
  return {
    count,
    x: Float32Array.from(x, quantised.x),
    y: Float32Array.from(y, quantised.y),
    z: Float32Array.from(floor, (f) => (f + 0.5) * FLOOR_METRES),
    building,
  }
}

function decodeRiver(
  buffer: ArrayBuffer,
  polygons: number,
  ringCount: number,
  points: number,
): NonNullable<StageCity['river']> {
  const c = new Columns(buffer, 'river.bin')
  const offsets = c.u32(ringCount + 1)
  const x = c.f32(points)
  const y = c.f32(points)
  c.u8(polygons) // kind
  const rings = c.u16(polygons)
  c.end()
  checkOffsets('river.bin', 'ring', offsets, points)
  const ringsUsed = rings.reduce((sum, count) => sum + count, 0)
  if (ringsUsed !== ringCount) {
    throw new Error(`river.bin polygons use ${ringsUsed} rings but it has ${ringCount}`)
  }
  return { offsets, x, y, rings }
}
