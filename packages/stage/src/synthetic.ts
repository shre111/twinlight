import { buildWorld, HOURS } from '../../../design/prototype/toy-sim.mjs'
import { NEVER, ROAD_KIND, type StageCity, type StageTape } from './inputs'

// Stand-in for a real city pack and tape until P-010 and P-014 land. The toy sim works in
// three.js axes (negative z is north) and float hours, so this converts to pack axes and minutes.

type ToyWorld = ReturnType<typeof buildWorld>

// Constants inside toy-sim.mjs that it does not export.
const TOY_DELIVERY_RADIUS_M = 900
const TOY_RAIN_HOURS = { from: 264, to: 336 }
const TOY_NEVER_HOURS = 1e9

export function synthesizeToyRun(residents: number): { city: StageCity; tape: StageTape } {
  const world = buildWorld(residents)
  return { city: toyCity(world), tape: toyTape(world) }
}

function toyCity(world: ToyWorld): StageCity {
  const { buildings: b, roads, arterials, bridges } = world.city
  const res = world.res
  const stores = world.stores
  return {
    buildings: {
      count: b.count,
      x: b.x,
      y: negate(b.z),
      width: b.w,
      depth: b.d,
      height: b.h,
      // The prototype turns buildings by -rot about +Y, which is clockwise seen from above.
      rotation: negate(b.rot),
    },
    roads: roadSegments([
      [roads, ROAD_KIND.street],
      [arterials, ROAD_KIND.arterial],
      [bridges, ROAD_KIND.bridge],
    ]),
    residents: {
      count: res.count,
      x: res.x,
      y: negate(res.z),
      z: res.y,
      building: Uint32Array.from(res.b),
    },
    stores: {
      count: stores.length,
      x: Float32Array.from(stores, (s) => s.x),
      y: Float32Array.from(stores, (s) => -s.z),
      radius: new Float32Array(stores.length).fill(TOY_DELIVERY_RADIUS_M),
    },
  }
}

function toyTape(world: ToyWorld): StageTape {
  const { sim, res } = world
  const orderCount = new Uint16Array(res.count)
  for (const resident of sim.deliveries.res) orderCount[resident] = (orderCount[resident] ?? 0) + 1
  return {
    simMinutes: HOURS * 60,
    tickMinutes: 60,
    residents: {
      awareAt: Uint32Array.from(sim.awareT, toMinutes),
      firstOrderAt: Uint32Array.from(sim.adoptT, toMinutes),
      orderCount,
    },
    stores: stockouts(sim.outIntervals as number[][][]),
    aggregates: {
      aware: Uint32Array.from(sim.awareCount),
      adopted: Uint32Array.from(sim.adoptCount),
      orders: Uint32Array.from(sim.ordersH),
      storesOut: Uint16Array.from(sim.outH),
      rainIntensity: Float32Array.from({ length: HOURS }, (_, hour) =>
        hour >= TOY_RAIN_HOURS.from && hour < TOY_RAIN_HOURS.to ? 1 : 0,
      ),
    },
  }
}

function toMinutes(hours: number): number {
  return hours >= TOY_NEVER_HOURS ? NEVER : Math.round(hours * 60)
}

function negate(values: Float32Array): Float32Array {
  return values.map((v) => -v)
}

function roadSegments(groups: [Float32Array, number][]): StageCity['roads'] {
  const total = groups.reduce((n, [segments]) => n + segments.length, 0)
  const segments = new Float32Array(total)
  const kind = new Uint8Array(total / 4)
  let offset = 0
  for (const [source, roadKind] of groups) {
    for (let i = 0; i < source.length; i += 2) {
      segments[offset + i] = source[i] ?? 0
      segments[offset + i + 1] = -(source[i + 1] ?? 0)
    }
    kind.fill(roadKind, offset / 4, (offset + source.length) / 4)
    offset += source.length
  }
  return { segments, kind }
}

function stockouts(intervalsByStore: number[][][]): StageTape['stores'] {
  const offsets = new Uint32Array(intervalsByStore.length + 1)
  const flat: number[] = []
  intervalsByStore.forEach((intervals, store) => {
    for (const [startHour = 0, endHour = 0] of intervals) flat.push(startHour * 60, endHour * 60)
    offsets[store + 1] = flat.length / 2
  })
  return { stockoutOffsets: offsets, stockoutIntervals: Uint32Array.from(flat) }
}
