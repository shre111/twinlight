import { buildWorld, HOURS } from '../../../design/prototype/toy-sim.mjs'
import { TOY_BOUNDS } from './extent'
import { NEVER, ROAD_KIND, type StageCity, type StageTape } from './inputs'

// The prototype's toy run for /dev/stage. The toy sim works in three.js axes (negative z is
// north) and float hours, so this converts to pack axes and minutes.

type ToyWorld = ReturnType<typeof buildWorld>
type ToySim = ToyWorld['sim']

const SIM_MINUTES = HOURS * 60

// Constants inside toy-sim.mjs that it does not export.
const TOY_DELIVERY_RADIUS_M = 900
const TOY_RAIN_HOURS = { from: 264, to: 336 }
const TOY_NEVER_HOURS = 1e9
const TOY_RIVER_STEP_M = 30
const TOY_RIVER_REACH_M = 6000
const toyRiverX = (z: number) =>
  520 + 230 * Math.sin(z / 1250 + 0.6) + 110 * Math.sin(z / 470 + 2.1)
const toyRiverHalfWidth = (z: number) => 88 + 18 * Math.sin(z / 800 + 1.0)

/** The toy run's campaign moments, in sim minutes, as the timeline names them. */
export const TOY_EVENTS = [
  { label: 'Instagram ad', short: 'Ad', at: 9 * 60 },
  { label: 'Creator campaign', short: 'Creators', at: 186 * 60 },
]

export function synthesizeToyRun(residents: number): { city: StageCity; tape: StageTape } {
  const world = buildWorld(residents)
  return { city: toyCity(world), tape: toyTape(world.sim, world.res.count) }
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
    river: toyRiver(),
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
    bounds: TOY_BOUNDS,
  }
}

// One ring: the west bank north to south, then the east bank back north.
function toyRiver(): NonNullable<StageCity['river']> {
  const samples = (2 * TOY_RIVER_REACH_M) / TOY_RIVER_STEP_M + 1
  const x = new Float32Array(samples * 2)
  const y = new Float32Array(samples * 2)
  for (let s = 0; s < samples; s++) {
    const z = s * TOY_RIVER_STEP_M - TOY_RIVER_REACH_M
    const back = samples * 2 - 1 - s
    x[s] = toyRiverX(z) - toyRiverHalfWidth(z)
    x[back] = toyRiverX(z) + toyRiverHalfWidth(z)
    y[s] = -z
    y[back] = -z
  }
  return { offsets: Uint32Array.of(0, samples * 2), x, y }
}

function toyTape(sim: ToySim, residents: number): StageTape {
  const firstOrderAt = Uint32Array.from(sim.adoptT, toMinutes)
  return {
    simMinutes: SIM_MINUTES,
    // The toy sim ticks hourly; engine tapes use 15-minute ticks.
    tickMinutes: 60,
    residents: {
      awareAt: Uint32Array.from(sim.awareT, toMinutes),
      firstOrderAt,
      orderCount: orderCounts(sim.deliveries.res, firstOrderAt, residents),
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

// Word of mouth can land just past the last hour; an event inside the run stays inside it.
function toMinutes(hours: number): number {
  return hours >= TOY_NEVER_HOURS ? NEVER : Math.min(Math.round(hours * 60), SIM_MINUTES - 1)
}

// The toy sim stops logging deliveries at a fixed cap, so a large run can have adopters
// with no logged order; they still ordered once.
function orderCounts(
  deliveryResidents: Int32Array,
  firstOrderAt: Uint32Array,
  residents: number,
): Uint16Array {
  const counts = new Uint16Array(residents)
  for (const resident of deliveryResidents) counts[resident] = (counts[resident] ?? 0) + 1
  firstOrderAt.forEach((at, resident) => {
    if (at !== NEVER && counts[resident] === 0) counts[resident] = 1
  })
  return counts
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
