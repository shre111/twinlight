import { readdirSync } from 'node:fs'
import { crc32 as nodeCrc32 } from 'node:zlib'
import { beforeAll, describe, expect, it } from 'vitest'
import { decodeTape, type Tape, tapeHash } from '../src/index'
import type { ProbeColumn } from './helpers'
import { camel, FIXTURE_DIR, hex, loadFixture, loadProbes, REQUIRED_FIXTURES } from './helpers'

type TypedArray = ArrayBufferView & { BYTES_PER_ELEMENT: number }

const CONSTRUCTORS = { u8: Uint8Array, u16: Uint16Array, u32: Uint32Array, f32: Float32Array }

function discover(): string[] {
  try {
    return readdirSync(FIXTURE_DIR)
      .filter((f) => f.endsWith('.twlt'))
      .map((f) => f.slice(0, -'.twlt'.length))
      .sort()
  } catch {
    return []
  }
}

const names = discover()

function littleEndianBytes(column: ArrayBufferView): Buffer {
  return Buffer.from(column.buffer, column.byteOffset, column.byteLength)
}

function edge(column: ArrayLike<number>, count: number, from: 'head' | 'tail'): number[] {
  const values = Array.from({ length: column.length }, (_, i) => column[i] as number)
  return from === 'head' ? values.slice(0, count) : values.slice(Math.max(0, values.length - count))
}

describe('fixtures', () => {
  it('include every required tape and its probes', () => {
    for (const name of REQUIRED_FIXTURES) expect(names).toContain(name)
    for (const name of names) expect(() => loadProbes(name)).not.toThrow()
  })
})

describe.each(names)('tape %s', (name) => {
  const probes = loadProbes(name)
  const bytes = loadFixture(name)
  let tape: Tape
  beforeAll(async () => {
    tape = await decodeTape(bytes)
  })

  it('has the file size and whole-file hash the Python writer recorded', () => {
    expect(bytes.length).toBe(probes.file.bytes)
    expect(tapeHash(bytes)).toBe(probes.file.blake3)
  })

  it('exposes the footer hash', () => {
    expect(hex(tape.hash)).toBe(probes.footer.blake3)
    expect(bytes.length - 40).toBe(probes.footer.total_length)
  })

  it('reads every header field', () => {
    const h = probes.header
    const t = tape.header
    expect(t.version).toBe(h.version)
    expect(t.flags).toBe(h.flags)
    expect(hex(t.runId)).toBe(h.run_id)
    expect(t.seed).toBe(BigInt(h.seed as string))
    expect(t.engineVersion).toBe(h.engine_version)
    expect(hex(t.cityPackHash)).toBe(h.city_pack_hash)
    expect(hex(t.scenarioHash)).toBe(h.scenario_hash)
    expect(t.simMinutes).toBe(h.sim_minutes)
    expect(t.tickMinutes).toBe(h.tick_minutes)
    expect(t.ticks).toBe((h.sim_minutes as number) / (h.tick_minutes as number))
    expect(t.residents).toBe(h.residents)
    expect(t.stores).toBe(h.stores)
    expect(t.chatterRows).toBe(h.chatter_rows)
    expect(t.deliveryRows).toBe(h.delivery_rows)
    expect(t.stockoutPairs).toBe(h.stockout_pairs)
  })

  it('reads the six table rows', () => {
    expect(tape.sections).toHaveLength(6)
    probes.sections.forEach((row, i) => {
      expect(tape.sections[i]).toMatchObject({
        kind: row.kind,
        offset: row.offset,
        length: row.length,
        rawLength: row.raw_length,
        crc32: row.crc32,
      })
    })
  })

  it('decodes the same set of sections and columns as the probes', () => {
    const decoded = (section: string) => Object.keys(tape[section as keyof Tape] as object).sort()
    const sections = Object.keys(probes.columns).map(camel).sort()
    expect(sections).toEqual(['aggregates', 'chatter', 'deliveries', 'residents', 'stores'])
    for (const [section, columns] of Object.entries(probes.columns)) {
      expect(decoded(camel(section))).toEqual(Object.keys(columns).map(camel).sort())
    }
  })

  const columnCases = Object.entries(loadProbes(name).columns).flatMap(([section, columns]) =>
    Object.entries(columns).map(([column, probe]) => [section, column, probe] as const),
  )
  it.each(columnCases)('column %s.%s matches its probe', (section, column, probe: ProbeColumn) => {
    const values = (
      tape[camel(section) as keyof Tape] as unknown as Record<string, ArrayBufferView>
    )[camel(column)] as unknown as ArrayLike<number> &
      ArrayBufferView & { BYTES_PER_ELEMENT: number }
    expect(values).toBeInstanceOf(CONSTRUCTORS[probe.dtype])
    expect(values.length).toBe(probe.length)
    expect(nodeCrc32(littleEndianBytes(values))).toBe(probe.crc32)
    if (probe.length === 0) expect(probe.crc32).toBe(0)
    if (probe.dtype !== 'f32') {
      let sum = 0n
      for (let i = 0; i < values.length; i++) sum += BigInt(values[i] as number)
      expect(sum.toString()).toBe(probe.sum)
    }
    for (const [edgeName, expected] of [
      ['head', probe.head],
      ['tail', probe.tail],
    ] as const) {
      const actual = edge(values, 8, edgeName)
      expect(actual).toHaveLength(expected.length)
      for (const [i, v] of actual.entries()) {
        expect(Object.is(v, expected[i]), `${edgeName}[${i}]: ${v} vs ${expected[i]}`).toBe(true)
      }
    }
  })

  it('starts every column on an 8-byte boundary of its section buffer', () => {
    for (const section of ['residents', 'chatter', 'deliveries', 'stores', 'aggregates'] as const) {
      for (const column of Object.values(tape[section]) as TypedArray[]) {
        expect(column.byteOffset % 8).toBe(0)
      }
    }
  })
})

describe('decoding', () => {
  it('copies the header fields and hash out of a Node Buffer', async () => {
    const input = Buffer.from(loadFixture('small'))
    const tape = await decodeTape(input)
    const fields = () =>
      [tape.header.runId, tape.header.cityPackHash, tape.header.scenarioHash, tape.hash].map(hex)
    const before = fields()
    input.fill(0xa5)
    expect(fields()).toEqual(before)
    expect(before).toEqual([
      loadProbes('small').header.run_id,
      loadProbes('small').header.city_pack_hash,
      loadProbes('small').header.scenario_hash,
      loadProbes('small').footer.blake3,
    ])
  })

  it('works on a subarray at an odd byte offset and leaves the input untouched', async () => {
    const original = loadFixture('small')
    const backing = new Uint8Array(original.length + 3)
    backing.set(original, 3)
    const input = backing.subarray(3)
    expect(input.byteOffset % 2).toBe(1)
    const tape = await decodeTape(input)
    expect(tape.header.residents).toBe(loadProbes('small').header.residents)
    expect(input).toEqual(original)
    for (const column of Object.values(tape.residents) as TypedArray[]) {
      expect(column.byteOffset % column.BYTES_PER_ELEMENT).toBe(0)
    }
  })
})
