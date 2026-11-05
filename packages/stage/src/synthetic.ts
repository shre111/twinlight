import {
  buildTies,
  buildWorld,
  DISTRICTS,
  HOURS,
  mulberry32,
  simulate,
} from '../../../design/prototype/toy-sim.mjs'
import { TOY_BOUNDS } from './extent'
import { NEVER, ROAD_KIND, type StageCity, type StageTape } from './inputs'

// Stand-in for a real tape until P-014 lands. The toy sim works in three.js axes (negative z is
// north) and float hours, so this converts to pack axes and minutes.

type ToyWorld = ReturnType<typeof buildWorld>
type ToySim = ToyWorld['sim']
type ToyDistrict = { name: string; x: number; z: number; i: number; east: boolean }

const TOY_DISTRICTS = DISTRICTS as unknown as ToyDistrict[]

const SIM_MINUTES = HOURS * 60

// Constants inside toy-sim.mjs that it does not export.
const TOY_SEED = 49
const TOY_DELIVERY_RADIUS_M = 900
const TOY_RAIN_HOURS = { from: 264, to: 336 }
const TOY_NEVER_HOURS = 1e9
const TOY_AGE_SHARE_WEST = [0.2, 0.27, 0.27, 0.16, 0.1]
const TOY_AGE_SHARE_EAST = [0.16, 0.22, 0.28, 0.21, 0.13]
const TOY_AGE_AFFINITY = [1.35, 1.2, 0.9, 0.6, 0.38]
const TOY_AGE_AD_WEIGHT = [1.0, 0.9, 0.62, 0.38, 0.2]
const TOY_CREATOR_DISTRICTS = new Set(['Satellite', 'Bodakdev', 'Vastrapur'])

/** The toy run's campaign moments, in sim minutes, as the timeline names them. */
export const TOY_EVENTS = [
  { label: 'Instagram ad', short: 'Ad', at: 9 * 60 },
  { label: 'Creator campaign', short: 'Creators', at: 186 * 60 },
]

export function synthesizeToyRun(residents: number): { city: StageCity; tape: StageTape } {
  const world = buildWorld(residents)
  return { city: toyCity(world), tape: toyTape(world.sim, world.res.count) }
}

/** The toy sim's launch over a generated pack's own residents and stores. */
export function synthesizeTapeFor(city: StageCity): StageTape {
  const rand = mulberry32(TOY_SEED)
  const residents = toyResidents(city, rand)
  const sim = simulate(residents, toyStores(city, residents), buildTies(residents, rand), rand)
  return toyTape(sim, residents.count)
}

// A stand-in, so it reads only what the stage decodes: traits are drawn as the prototype draws them,
// districts go by nearest centre, and ties are rebuilt rather than read from the pack.
function toyResidents(city: StageCity, rand: () => number) {
  const count = city.residents.count
  const x = city.residents.x
  const z = negate(city.residents.y)
  const dist = new Uint8Array(count)
  const aff = new Float32Array(count)
  const adW = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    const district = nearestDistrict(x[i] ?? 0, z[i] ?? 0)
    const band = ageBand(district.east ? TOY_AGE_SHARE_EAST : TOY_AGE_SHARE_WEST, rand())
    const creator = TOY_CREATOR_DISTRICTS.has(district.name)
    dist[i] = district.i
    aff[i] =
      (TOY_AGE_AFFINITY[band] ?? 0) *
      (district.east ? 0.8 : 1.08) *
      (creator ? 1.22 : 1) *
      (0.45 + rand() * 1.1)
    adW[i] = (TOY_AGE_AD_WEIGHT[band] ?? 0) * (district.east ? 0.72 : 1)
  }
  return { count, x, z, dist, aff, adW }
}

function nearestDistrict(x: number, z: number): ToyDistrict {
  const distance = (d: ToyDistrict) => (x - d.x) ** 2 + (z - d.z) ** 2
  return TOY_DISTRICTS.reduce((best, d) => (distance(d) < distance(best) ? d : best))
}

function ageBand(shares: number[], roll: number): number {
  let left = roll
  let band = 0
  while (band < shares.length - 1 && left > (shares[band] ?? 0)) {
    left -= shares[band] ?? 0
    band++
  }
  return band
}

function toyStores(
  city: StageCity,
  residents: { count: number; x: Float32Array; z: Float32Array },
) {
  const { count } = residents
  const storeX = city.stores.x
  const storeZ = negate(city.stores.y)
  const stores = Array.from({ length: city.stores.count }, () => ({ catchment: 0 }))
  const near1 = new Uint8Array(count)
  const near2 = new Uint8Array(count)
  const d2 = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    let first = 0
    let second = 0
    let firstDistance = Infinity
    let secondDistance = Infinity
    for (let s = 0; s < stores.length; s++) {
      const distance = Math.hypot(
        (residents.x[i] ?? 0) - (storeX[s] ?? 0),
        (residents.z[i] ?? 0) - (storeZ[s] ?? 0),
      )
      if (distance < firstDistance) {
        second = first
        secondDistance = firstDistance
        first = s
        firstDistance = distance
      } else if (distance < secondDistance) {
        second = s
        secondDistance = distance
      }
    }
    near1[i] = first
    near2[i] = second
    d2[i] = secondDistance
    const nearest = stores[first]
    if (nearest) nearest.catchment++
  }
  return { stores, near1, near2, d2 }
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
    bounds: TOY_BOUNDS,
  }
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
