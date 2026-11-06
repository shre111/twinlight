import type { StageCity } from './inputs'
import { Columns, checkOffsets, decodeStores, polylineSegments } from './pack-columns'

export interface PackCountsV1 {
  buildings: number
  residents: number
  stores: number
  roads: number
  roadPoints: number
  riverPolygons: number
  riverPoints: number
}

type FilesV1 = Record<
  'buildings.bin' | 'roads.bin' | 'river.bin' | 'residents.bin' | 'stores.bin',
  ArrayBuffer
>

/** Toy packs until the P-025 writer lands; goes away with it. */
export function decodePackV1(counts: PackCountsV1, files: FilesV1): StageCity {
  return {
    buildings: decodeBuildings(files['buildings.bin'], counts.buildings),
    roads: decodeRoads(files['roads.bin'], counts.roads, counts.roadPoints),
    river: decodeRiver(files['river.bin'], counts.riverPolygons, counts.riverPoints),
    residents: decodeResidents(files['residents.bin'], counts.residents),
    stores: decodeStores(files['stores.bin'], counts.stores),
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

function decodeRoads(buffer: ArrayBuffer, roads: number, points: number): StageCity['roads'] {
  const c = new Columns(buffer, 'roads.bin')
  const offsets = c.u32(roads + 1)
  const x = c.f32(points)
  const y = c.f32(points)
  const roadClass = c.u8(roads)
  c.end()
  checkOffsets('roads.bin', 'road', offsets, points)
  return polylineSegments(offsets, x, y, roadClass)
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
