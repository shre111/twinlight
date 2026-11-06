import { describe, expect, it } from 'vitest'
import { decodeTape, TapeError } from '../src/index'
import {
  compressedSection,
  crc,
  FOOTER_SIZE,
  flipByte,
  HEADER_SIZE,
  headerValue,
  loadFixture,
  patchFooterLength,
  patchHeader,
  patchRow,
  rawSection,
  readRowField,
  replaceRawSection,
} from './helpers'

const small = loadFixture('small')
const SECTION = { residents: 0, stores: 3 }
const NO_HASH = { verifyHash: false }
const RUN_ID_AT = 8

async function failure(bytes: Uint8Array, options?: { verifyHash?: boolean }): Promise<Error> {
  const error = await decodeTape(bytes, options).then(
    () => undefined,
    (e: unknown) => e,
  )
  expect(error).toBeInstanceOf(TapeError)
  return error as Error
}

describe('small fixture is intact', () => {
  it('decodes unmodified', async () => {
    await expect(decodeTape(small)).resolves.toBeDefined()
    await expect(decodeTape(small, NO_HASH)).resolves.toBeDefined()
  })
})

describe('header checks', () => {
  it('rejects bad magic', async () => {
    await failure(flipByte(small, 0), NO_HASH)
  })

  it('rejects version 2', async () => {
    await failure(patchHeader(small, 'version', 2), NO_HASH)
  })

  it('rejects flags without the complete bit', async () => {
    await failure(patchHeader(small, 'flags', 0), NO_HASH)
    await failure(patchHeader(small, 'flags', 2), NO_HASH)
  })

  it('rejects a non-zero reserved field', async () => {
    await failure(patchHeader(small, 'reserved', 1), NO_HASH)
  })

  it('rejects tick_minutes of zero', async () => {
    await failure(patchHeader(small, 'tickMinutes', 0), NO_HASH)
  })

  it('rejects sim_minutes that is not a multiple of tick_minutes', async () => {
    await failure(patchHeader(small, 'simMinutes', headerValue(small, 'simMinutes') + 1), NO_HASH)
  })
})

describe('file framing', () => {
  it('rejects a file shorter than header plus footer', async () => {
    await failure(small.slice(0, HEADER_SIZE + FOOTER_SIZE - 1))
    await failure(new Uint8Array(0))
  })

  it('rejects one byte chopped off the end', async () => {
    await failure(small.slice(0, -1))
    await failure(small.slice(0, -1), NO_HASH)
  })

  it('rejects a footer total_length that is off by 8', async () => {
    const total = small.length - FOOTER_SIZE
    await failure(patchFooterLength(small, total + 8), NO_HASH)
    await failure(patchFooterLength(small, total - 8), NO_HASH)
  })
})

describe('integrity', () => {
  const at = readRowField(small, SECTION.residents, 'offset') + 2

  it('catches a flipped section byte by crc32 even without the hash check', async () => {
    expect((await failure(flipByte(small, at), NO_HASH)).message).toMatch(/crc|residents/i)
  })

  it('catches a flipped section byte with default options', async () => {
    await failure(flipByte(small, at))
  })

  it('catches a flipped run_id byte by hash only', async () => {
    const mutated = flipByte(small, RUN_ID_AT)
    await failure(mutated)
    await expect(decodeTape(mutated, NO_HASH)).resolves.toBeDefined()
  })

  it('catches a flipped footer hash byte', async () => {
    await failure(flipByte(small, small.length - 1))
  })
})

describe('section table', () => {
  it('rejects kinds out of order', async () => {
    const swapped = patchRow(patchRow(small, 0, 'kind', 2), 1, 'kind', 1)
    expect((await failure(swapped, NO_HASH)).message).toMatch(/kind/i)
  })

  it('rejects a misaligned offset', async () => {
    const offset = readRowField(small, 1, 'offset')
    await failure(patchRow(small, 1, 'offset', offset + 1), NO_HASH)
  })

  it('rejects an offset inside the header', async () => {
    await failure(patchRow(small, 0, 'offset', HEADER_SIZE - 8), NO_HASH)
  })

  it('rejects a section overlapping the previous one', async () => {
    const previous = readRowField(small, 0, 'offset')
    expect((await failure(patchRow(small, 1, 'offset', previous), NO_HASH)).message).toMatch(
      /overlap|section/i,
    )
  })

  it('rejects a section running past the end', async () => {
    const length = readRowField(small, 4, 'length')
    expect((await failure(patchRow(small, 4, 'length', length + 4096), NO_HASH)).message).toMatch(
      /AGGREGATES|aggregates|end/,
    )
  })

  it('rejects a raw_length that disagrees with the header counts', async () => {
    const raw = readRowField(small, SECTION.residents, 'rawLength')
    const error = await failure(patchRow(small, SECTION.residents, 'rawLength', raw + 8), NO_HASH)
    expect(error.message).toMatch(/residents|raw/i)
  })

  it('rejects a header count changed so columns no longer fit', async () => {
    const residents = headerValue(small, 'residents')
    await failure(patchHeader(small, 'residents', residents + 2), NO_HASH)
    await failure(patchHeader(small, 'residents', residents + 9), NO_HASH)
  })

  it('cannot see a count that moves inside the column padding without the hash', async () => {
    const residents = headerValue(small, 'residents')
    expect(residents % 8).toBe(7)
    const mutated = patchHeader(small, 'residents', residents + 1)
    await failure(mutated)
    expect((await decodeTape(mutated, NO_HASH)).residents.awareAt).toHaveLength(residents + 1)
  })
})

describe('section contents', () => {
  const stores = headerValue(small, 'stores')
  const ticks = headerValue(small, 'simMinutes') / headerValue(small, 'tickMinutes')
  const pad8 = (n: number) => n + ((8 - (n % 8)) % 8)
  const offsetsAt = pad8(4 * stores * ticks) + pad8(2 * stores * ticks)
  const raw = rawSection(small, SECTION.stores)

  function withOffsets(values: number[]): Uint8Array {
    const edited = raw.slice()
    const dv = new DataView(edited.buffer)
    for (const [i, v] of values.entries()) dv.setUint32(offsetsAt + 4 * i, v, true)
    return replaceRawSection(small, SECTION.stores, edited)
  }

  it('is a fixture with at least two stores', () => {
    expect(stores).toBeGreaterThanOrEqual(2)
  })

  it('rejects stockout offsets that do not start at 0', async () => {
    const pairs = headerValue(small, 'stockoutPairs')
    const bad = Array.from({ length: stores + 1 }, () => pairs)
    bad[0] = 1
    expect((await failure(withOffsets(bad), NO_HASH)).message).toMatch(/stockout_offsets|offsets/i)
  })

  it('rejects stockout offsets that decrease', async () => {
    const pairs = headerValue(small, 'stockoutPairs')
    const bad = Array.from({ length: stores + 1 }, () => pairs)
    bad[0] = 0
    bad[1] = pairs + 1
    expect((await failure(withOffsets(bad), NO_HASH)).message).toMatch(/stockout_offsets|offsets/i)
  })

  it('rejects stockout offsets that do not end at stockout_pairs', async () => {
    const pairs = headerValue(small, 'stockoutPairs')
    const bad = Array.from({ length: stores + 1 }, () => 0)
    bad[stores] = pairs + 1
    expect((await failure(withOffsets(bad), NO_HASH)).message).toMatch(/stockout_offsets|offsets/i)
  })

  it('rebuilding a section unchanged still decodes', async () => {
    await expect(
      decodeTape(replaceRawSection(small, SECTION.stores, raw), NO_HASH),
    ).resolves.toBeDefined()
  })

  it('keeps the table crc32 consistent with the compressed bytes', () => {
    expect(crc(compressedSection(small, 0))).toBe(readRowField(small, 0, 'crc32'))
  })
})
