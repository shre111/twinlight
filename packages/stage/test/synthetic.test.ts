import { beforeAll, describe, expect, it } from 'vitest'
import { buildWorld } from '../../../design/prototype/toy-sim.mjs'
import { NEVER, ROAD_KIND, type StageCity, type StageTape } from '../src/inputs'
import { synthesizeToyRun } from '../src/synthetic'

const RESIDENTS = 3000
let city: StageCity
let tape: StageTape

beforeAll(() => {
  ;({ city, tape } = synthesizeToyRun(RESIDENTS))
})

describe('synthetic toy run', () => {
  it('maps the toy sim north (negative z) to pack north (positive y)', () => {
    const world = buildWorld(RESIDENTS)
    for (const i of [0, 1, 500, 2999]) {
      expect(city.residents.y[i]).toBe(-(world.res.z[i] ?? 0))
      expect(city.residents.z[i]).toBe(world.res.y[i])
      expect(city.buildings.y[i]).toBe(-(world.city.buildings.z[i] ?? 0))
    }
    expect(city.stores.y[0]).toBe(-(world.stores[0]?.z ?? 0))
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
