import { beforeAll, describe, expect, it } from 'vitest'
import { buildWorld, DISTRICTS } from '../../../design/prototype/toy-sim.mjs'
import { NEVER, ROAD_KIND, type StageCity, type StageTape } from '../src/inputs'
import { synthesizeToyRun } from '../src/synthetic'

const RESIDENTS = 3000
let city: StageCity
let tape: StageTape
let world: ReturnType<typeof buildWorld>

beforeAll(() => {
  ;({ city, tape } = synthesizeToyRun(RESIDENTS))
  world = buildWorld(RESIDENTS)
})

function residentCentre(district: string): { x: number; y: number } {
  const index = DISTRICTS.findIndex((d) => d.name === district)
  let x = 0
  let y = 0
  let n = 0
  for (let i = 0; i < city.residents.count; i++) {
    if (world.res.dist[i] !== index) continue
    x += city.residents.x[i] ?? 0
    y += city.residents.y[i] ?? 0
    n++
  }
  expect(n, `${district} residents`).toBeGreaterThan(20)
  return { x: x / n, y: y / n }
}

describe('synthetic toy run', () => {
  it('keeps Ahmedabad the right way up: Chandkheda north, Maninagar south-east, Thaltej west', () => {
    const north = residentCentre('Chandkheda')
    const southEast = residentCentre('Maninagar')
    const west = residentCentre('Thaltej')
    expect(north.y).toBeGreaterThan(southEast.y)
    expect(southEast.x).toBeGreaterThan(west.x)
    expect(north.y).toBeGreaterThan(0)
    expect(southEast.y).toBeLessThan(0)
  })

  it('flips road segments into pack axes', () => {
    const source = world.city.roads
    expect(city.roads.segments.slice(0, 4)).toEqual(
      Float32Array.of(source[0] ?? 0, -(source[1] ?? 0), source[2] ?? 0, -(source[3] ?? 0)),
    )
  })

  it('places every resident on the footprint of its building in pack axes', () => {
    const { buildings: b, residents: r } = city
    let checked = 0
    for (let i = 0; i < r.count; i++) {
      const k = r.building[i] ?? 0
      const dx = (r.x[i] ?? 0) - (b.x[k] ?? 0)
      const dy = (r.y[i] ?? 0) - (b.y[k] ?? 0)
      const rot = b.rotation[k] ?? 0
      const alongWidth = dx * Math.cos(rot) + dy * Math.sin(rot)
      const alongDepth = -dx * Math.sin(rot) + dy * Math.cos(rot)
      expect(Math.abs(alongWidth)).toBeLessThanOrEqual((b.width[k] ?? 0) / 2 + 1)
      expect(Math.abs(alongDepth)).toBeLessThanOrEqual((b.depth[k] ?? 0) / 2 + 1)
      expect(r.z[i]).toBeLessThan(b.height[k] ?? 0)
      checked++
    }
    expect(checked).toBe(RESIDENTS)
  })

  it('tags streets, arterials and bridges', () => {
    expect(city.roads.segments.length).toBe(city.roads.kind.length * 4)
    for (const kind of Object.values(ROAD_KIND)) expect(city.roads.kind).toContain(kind)
  })

  it('writes event times as minutes within the run, or never', () => {
    expect(tape.simMinutes).toBe(30 * 24 * 60)
    const { awareAt, firstOrderAt, orderCount } = tape.residents
    let adopters = 0
    for (let i = 0; i < RESIDENTS; i++) {
      const aware = awareAt[i] ?? NEVER
      const ordered = firstOrderAt[i] ?? NEVER
      if (aware !== NEVER) expect(aware).toBeLessThan(tape.simMinutes)
      if (ordered === NEVER) {
        expect(orderCount[i]).toBe(0)
        continue
      }
      adopters++
      expect(ordered).toBeGreaterThanOrEqual(aware)
      expect(orderCount[i]).toBeGreaterThan(0)
    }
    expect(adopters).toBeGreaterThan(0)
    expect(tape.aggregates.adopted.at(-1)).toBe(adopters)
  })

  it('has one aggregate per tick and stockouts as CSR over interval pairs', () => {
    const ticks = tape.simMinutes / tape.tickMinutes
    expect(tape.aggregates.aware.length).toBe(ticks)
    expect(tape.aggregates.rainIntensity.length).toBe(ticks)
    const { stockoutOffsets, stockoutIntervals } = tape.stores
    expect(stockoutOffsets.length).toBe(city.stores.count + 1)
    expect(stockoutOffsets.at(-1)).toBe(stockoutIntervals.length / 2)
    for (let s = 0; s < city.stores.count; s++) {
      expect(stockoutOffsets[s + 1]).toBeGreaterThanOrEqual(stockoutOffsets[s] ?? 0)
    }
  })
})
