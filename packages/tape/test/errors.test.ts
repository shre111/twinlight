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
const SECTION = { residents: 0, chatter: 1, stores: 3, aggregates: 4 }
const NO_HASH = { verifyHash: false }
const RUN_ID_AT = 8
const ENGINE_VERSION_AT = 32

// Every case names the check that must fire: a looser match would pass on a later check
// (the crc32 or the zstd frame) with the intended one removed.
async function failure(
  bytes: Uint8Array,
  message: RegExp,
  options?: { verifyHash?: boolean },
): Promise<void> {
  const error = await decodeTape(bytes, options).then(
    () => undefined,
    (e: unknown) => e,
  )
  expect(error).toBeInstanceOf(TapeError)
  expect((error as Error).message).toMatch(message)
}

describe('small fixture is intact', () => {
  it('decodes unmodified', async () => {
    await expect(decodeTape(small)).resolves.toBeDefined()
    await expect(decodeTape(small, NO_HASH)).resolves.toBeDefined()
  })
})

describe('header checks', () => {
  it('rejects bad magic', async () => {
    await failure(flipByte(small, 0), /bad magic/, NO_HASH)
    await failure(flipByte(small, 0), /bad magic/)
  })

  it('rejects version 2', async () => {
    await failure(patchHeader(small, 'version', 2), /unsupported tape version 2/, NO_HASH)
  })

  it('rejects flags without the complete bit', async () => {
    await failure(patchHeader(small, 'flags', 0), /not marked complete/, NO_HASH)
    await failure(patchHeader(small, 'flags', 2), /not marked complete/, NO_HASH)
  })

  it('ignores flag bits it does not know', async () => {
    const tape = await decodeTape(patchHeader(small, 'flags', 0x8001), NO_HASH)
    expect(tape.header.flags).toBe(0x8001)
  })

  it('rejects a non-zero reserved field', async () => {
    await failure(patchHeader(small, 'reserved', 1), /reserved/, NO_HASH)
  })

  it('rejects tick_minutes of zero', async () => {
    await failure(patchHeader(small, 'tickMinutes', 0), /multiple of tick_minutes/, NO_HASH)
  })

  it('rejects sim_minutes that is not a multiple of tick_minutes', async () => {
    const simMinutes = headerValue(small, 'simMinutes') + 1
    await failure(patchHeader(small, 'simMinutes', simMinutes), /multiple of tick_minutes/, NO_HASH)
  })

  it('rejects an engine_version that is not ASCII', async () => {
    await failure(flipByte(small, ENGINE_VERSION_AT), /engine_version is not ASCII/, NO_HASH)
  })
})

describe('file framing', () => {
  it('rejects a file shorter than header plus footer', async () => {
    await failure(small.slice(0, HEADER_SIZE + FOOTER_SIZE - 1), /too short/)
    await failure(new Uint8Array(0), /too short/)
  })

  it('rejects one byte chopped off the end', async () => {
    await failure(small.slice(0, -1), /total_length does not match/)
    await failure(small.slice(0, -1), /total_length does not match/, NO_HASH)
  })

  it('rejects a footer total_length that is off by 8', async () => {
    const total = small.length - FOOTER_SIZE
    await failure(patchFooterLength(small, total + 8), /total_length does not match/, NO_HASH)
    await failure(patchFooterLength(small, total - 8), /total_length does not match/, NO_HASH)
  })

  it('rejects bytes between the last section and the footer', async () => {
    const total = small.length - FOOTER_SIZE
    const grown = new Uint8Array(small.length + 8)
    grown.set(small.subarray(0, total))
    grown.set(small.subarray(total), total + 8)
    await failure(patchFooterLength(grown, total + 8), /does not end at total_length/, NO_HASH)
  })
})

describe('integrity', () => {
  const at = readRowField(small, SECTION.residents, 'offset') + 2

  it('catches a flipped section byte by crc32 even without the hash check', async () => {
    await failure(flipByte(small, at), /RESIDENTS section crc32 mismatch/, NO_HASH)
  })

  it('catches a flipped section byte by the hash with default options', async () => {
    await failure(flipByte(small, at), /footer hash does not match/)
  })

  it('reports a frame that is not zstd once its crc32 agrees', async () => {
    const mutated = flipByte(small, at)
    const crc32 = crc(compressedSection(mutated, SECTION.residents))
    const patched = patchRow(mutated, SECTION.residents, 'crc32', crc32)
    await failure(patched, /RESIDENTS section is not a valid zstd frame/, NO_HASH)
  })

  it('catches a flipped run_id byte by hash only', async () => {
    const mutated = flipByte(small, RUN_ID_AT)
    await failure(mutated, /footer hash does not match/)
    await expect(decodeTape(mutated, NO_HASH)).resolves.toBeDefined()
  })

  it('catches a flipped footer hash byte', async () => {
    await failure(flipByte(small, small.length - 1), /footer hash does not match/)
  })
})

describe('section table', () => {
  it('rejects kinds out of order', async () => {
    const swapped = patchRow(patchRow(small, 0, 'kind', 2), 1, 'kind', 1)
    await failure(swapped, /entry 0 is kind 2, expected 1/, NO_HASH)
  })

  it('rejects a misaligned offset', async () => {
    const offset = readRowField(small, SECTION.chatter, 'offset')
    const mutated = patchRow(small, SECTION.chatter, 'offset', offset + 1)
    await failure(mutated, /CHATTER section offset \d+ is misaligned/, NO_HASH)
  })

  it('rejects an offset inside the header', async () => {
    const mutated = patchRow(small, SECTION.residents, 'offset', HEADER_SIZE - 8)
    await failure(mutated, /RESIDENTS section offset 344 is misaligned or inside/, NO_HASH)
  })

  it('rejects a section overlapping the previous one', async () => {
    const previous = readRowField(small, SECTION.residents, 'offset')
    const mutated = patchRow(small, SECTION.chatter, 'offset', previous)
    await failure(mutated, /CHATTER section overlaps or precedes/, NO_HASH)
  })

  it('rejects a section running past the end', async () => {
    const length = readRowField(small, SECTION.aggregates, 'length')
    const mutated = patchRow(small, SECTION.aggregates, 'length', length + 4096)
    await failure(mutated, /AGGREGATES section runs past the end/, NO_HASH)
  })

  it('rejects a table value too large to be a byte offset', async () => {
    const mutated = small.slice()
    new DataView(mutated.buffer).setBigUint64(160 + 4, 2n ** 53n, true)
    await failure(mutated, /RESIDENTS section table entry is out of range/, NO_HASH)
  })

  it('rejects a raw_length that disagrees with the header counts', async () => {
    const raw = readRowField(small, SECTION.residents, 'rawLength')
    const mutated = patchRow(small, SECTION.residents, 'rawLength', raw + 8)
    await failure(mutated, /RESIDENTS section claims \d+ raw bytes/, NO_HASH)
  })

  it('rejects a header count changed so columns no longer fit', async () => {
    const residents = headerValue(small, 'residents')
    const message = /RESIDENTS section claims \d+ raw bytes/
    await failure(patchHeader(small, 'residents', residents + 2), message, NO_HASH)
    await failure(patchHeader(small, 'residents', residents + 9), message, NO_HASH)
  })

  it('cannot see a count that moves inside the column padding without the hash', async () => {
    const residents = headerValue(small, 'residents')
    expect(residents % 8).toBe(7)
    const mutated = patchHeader(small, 'residents', residents + 1)
    await failure(mutated, /footer hash does not match/)
    expect((await decodeTape(mutated, NO_HASH)).residents.awareAt).toHaveLength(residents + 1)
  })
})

describe('section contents', () => {
  const stores = headerValue(small, 'stores')
  const ticks = headerValue(small, 'simMinutes') / headerValue(small, 'tickMinutes')
  const pairs = headerValue(small, 'stockoutPairs')
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
    const bad = Array.from({ length: stores + 1 }, () => pairs)
    bad[0] = 1
    await failure(withOffsets(bad), /stockout_offsets must start at 0/, NO_HASH)
  })

  it('rejects stockout offsets that decrease', async () => {
    const bad = Array.from({ length: stores + 1 }, () => pairs)
    bad[0] = 0
    bad[1] = pairs + 1
    await failure(withOffsets(bad), /stockout_offsets must be non-decreasing/, NO_HASH)
  })

  it('rejects stockout offsets that do not end at stockout_pairs', async () => {
    const bad = Array.from({ length: stores + 1 }, () => 0)
    bad[stores] = pairs + 1
    await failure(withOffsets(bad), /stockout_offsets does not end at stockout_pairs/, NO_HASH)
  })

  it('rejects a section with bytes after its frame', async () => {
    const offset = readRowField(small, SECTION.aggregates, 'offset')
    const length = readRowField(small, SECTION.aggregates, 'length')
    expect(length % 8).not.toBe(0)
    const grown = patchRow(small, SECTION.aggregates, 'length', length + 1)
    const crc32 = crc(grown.subarray(offset, offset + length + 1))
    const patched = patchRow(grown, SECTION.aggregates, 'crc32', crc32)
    await failure(patched, /AGGREGATES section is not a valid zstd frame/, NO_HASH)
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
