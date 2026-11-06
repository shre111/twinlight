import { blake3 } from '@noble/hashes/blake3.js'
import { bytesToHex } from '@noble/hashes/utils.js'
import { crc32 } from './crc32'
import {
  SECTION_NAMES,
  TAPE_VERSION,
  type Tape,
  type TapeAggregates,
  type TapeChatter,
  type TapeDeliveries,
  TapeError,
  type TapeHeader,
  type TapeResidents,
  type TapeSection,
  type TapeStores,
} from './format'
import { decompressZstd } from './zstd'

const MAGIC = 0x544c5754
const FLAG_COMPLETE = 1
const FIXED_HEADER_BYTES = 160
const TABLE_ROW_BYTES = 32
const SECTION_COUNT = SECTION_NAMES.length
const HEADER_BYTES = FIXED_HEADER_BYTES + SECTION_COUNT * TABLE_ROW_BYTES
const HASH_BYTES = 32
const FOOTER_BYTES = 8 + HASH_BYTES
const ALIGN = 8
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1

export interface DecodeOptions {
  /** Check the footer's blake3 over the whole body (default true). Section crc32s always run. */
  verifyHash?: boolean
}

/** Decode a complete v1 tape (ARCHITECTURE.md section 4). Rejects with `TapeError`. */
export async function decodeTape(bytes: Uint8Array, options: DecodeOptions = {}): Promise<Tape> {
  if (!LITTLE_ENDIAN) throw new TapeError('tapes can only be decoded on a little-endian host')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (bytes.length < HEADER_BYTES + FOOTER_BYTES) {
    throw new TapeError('file is too short to be a tape')
  }
  if (view.getUint32(0, true) !== MAGIC) throw new TapeError('not a tape (bad magic)')
  const version = view.getUint16(4, true)
  if (version !== TAPE_VERSION) throw new TapeError(`unsupported tape version ${version}`)

  const bodyLength = bytes.length - FOOTER_BYTES
  if (view.getBigUint64(bodyLength, true) !== BigInt(bodyLength)) {
    throw new TapeError('footer total_length does not match the file size')
  }
  const hash = bytes.slice(bodyLength + 8)
  if (options.verifyHash !== false && !sameBytes(blake3(bytes.subarray(0, bodyLength)), hash)) {
    throw new TapeError('footer hash does not match the file contents')
  }

  const header = readHeader(bytes, view)
  const sections = readTable(view, bodyLength)
  const expected = rawLengths(header)
  const raw: Uint8Array<ArrayBuffer>[] = []
  for (const [index, section] of sections.entries()) {
    raw.push(await readSection(bytes, section, expected[index] ?? 0))
  }
  const [residents, chatter, deliveries, stores, aggregates] = raw as [
    Uint8Array<ArrayBuffer>,
    Uint8Array<ArrayBuffer>,
    Uint8Array<ArrayBuffer>,
    Uint8Array<ArrayBuffer>,
    Uint8Array<ArrayBuffer>,
  ]
  return {
    header,
    sections,
    hash,
    residents: residentsFrom(residents, header),
    chatter: chatterFrom(chatter, header),
    deliveries: deliveriesFrom(deliveries, header),
    stores: storesFrom(stores, header),
    aggregates: aggregatesFrom(aggregates, header),
  }
}

/** blake3 hex of the whole file: the value the goldens pin (`twl_tape.tape_hash`). */
export function tapeHash(bytes: Uint8Array): string {
  return bytesToHex(blake3(bytes))
}

function readHeader(bytes: Uint8Array, view: DataView): TapeHeader {
  const flags = view.getUint16(6, true)
  if (!(flags & FLAG_COMPLETE)) throw new TapeError('tape is not marked complete (flags bit0)')
  const simMinutes = view.getUint32(128, true)
  const tickMinutes = view.getUint32(132, true)
  if (view.getUint32(156, true) !== 0) throw new TapeError('reserved header field must be zero')
  if (tickMinutes === 0 || simMinutes % tickMinutes !== 0) {
    throw new TapeError('sim_minutes must be a positive multiple of tick_minutes')
  }
  return {
    version: TAPE_VERSION,
    flags,
    runId: bytes.slice(8, 24),
    seed: view.getBigUint64(24, true),
    engineVersion: engineVersion(bytes.subarray(32, 64)),
    cityPackHash: bytes.slice(64, 96),
    scenarioHash: bytes.slice(96, 128),
    simMinutes,
    tickMinutes,
    ticks: simMinutes / tickMinutes,
    residents: view.getUint32(136, true),
    stores: view.getUint32(140, true),
    chatterRows: view.getUint32(144, true),
    deliveryRows: view.getUint32(148, true),
    stockoutPairs: view.getUint32(152, true),
  }
}

function engineVersion(field: Uint8Array): string {
  let end = field.length
  while (end > 0 && field[end - 1] === 0) end--
  const text = field.subarray(0, end)
  if (text.some((byte) => byte > 0x7f)) throw new TapeError('engine_version is not ASCII')
  return String.fromCharCode(...text)
}

function readTable(view: DataView, bodyLength: number): TapeSection[] {
  const sections: TapeSection[] = []
  let previousEnd = HEADER_BYTES
  for (const [index, name] of SECTION_NAMES.entries()) {
    const at = FIXED_HEADER_BYTES + index * TABLE_ROW_BYTES
    const kind = view.getUint32(at, true)
    if (kind !== index + 1) {
      throw new TapeError(`section table entry ${index} is kind ${kind}, expected ${index + 1}`)
    }
    const offset = tableNumber(view, at + 4, name)
    const length = tableNumber(view, at + 12, name)
    const rawLength = tableNumber(view, at + 20, name)
    if (offset % ALIGN !== 0 || offset < HEADER_BYTES) {
      throw new TapeError(`${name} section offset ${offset} is misaligned or inside the header`)
    }
    if (offset < previousEnd) {
      throw new TapeError(`${name} section overlaps or precedes the section before it`)
    }
    if (offset + length > bodyLength) {
      throw new TapeError(`${name} section runs past the end of the file`)
    }
    previousEnd = offset + length
    sections.push({ kind, offset, length, rawLength, crc32: view.getUint32(at + 28, true) })
  }
  if (padded(previousEnd) !== bodyLength) {
    throw new TapeError('the last section does not end at total_length')
  }
  return sections
}

function tableNumber(view: DataView, at: number, name: string): number {
  const value = view.getBigUint64(at, true)
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new TapeError(`${name} section table entry is out of range`)
  }
  return Number(value)
}

function rawLengths(header: TapeHeader): number[] {
  const { residents: n, chatterRows: m, deliveryRows: d, stores, ticks, stockoutPairs } = header
  const series = stores * ticks
  return [
    padded(4 * n) * 3 + padded(2 * n) + padded(n) * 2,
    padded(4 * m) * 3 + padded(m),
    padded(2 * d) + padded(4 * d) * 4 + padded(d),
    padded(4 * series) + padded(2 * series) + padded(4 * (stores + 1)) + padded(8 * stockoutPairs),
    padded(4 * ticks) * 3 + padded(2 * ticks) + padded(4 * ticks),
    0,
  ]
}

async function readSection(
  bytes: Uint8Array,
  section: TapeSection,
  expectedRawLength: number,
): Promise<Uint8Array<ArrayBuffer>> {
  const name = SECTION_NAMES[section.kind - 1]
  if (section.rawLength !== expectedRawLength) {
    throw new TapeError(
      `${name} section claims ${section.rawLength} raw bytes; the header counts need ${expectedRawLength}`,
    )
  }
  const compressed = bytes.subarray(section.offset, section.offset + section.length)
  if (crc32(compressed) !== section.crc32) throw new TapeError(`${name} section crc32 mismatch`)
  try {
    return await decompressZstd(compressed, section.rawLength)
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error)
    throw new TapeError(`${name} section is not a valid zstd frame: ${reason}`)
  }
}

function residentsFrom(raw: Uint8Array<ArrayBuffer>, header: TapeHeader): TapeResidents {
  const columns = new Columns(raw)
  const n = header.residents
  return {
    awareAt: columns.u32(n),
    firstOrderAt: columns.u32(n),
    orderCount: columns.u16(n),
    heardFrom: columns.u32(n),
    firstSource: columns.u8(n),
    outcomeFlags: columns.u8(n),
  }
}

function chatterFrom(raw: Uint8Array<ArrayBuffer>, header: TapeHeader): TapeChatter {
  const columns = new Columns(raw)
  const m = header.chatterRows
  return { from: columns.u32(m), to: columns.u32(m), at: columns.u32(m), kind: columns.u8(m) }
}

function deliveriesFrom(raw: Uint8Array<ArrayBuffer>, header: TapeHeader): TapeDeliveries {
  const columns = new Columns(raw)
  const d = header.deliveryRows
  return {
    store: columns.u16(d),
    resident: columns.u32(d),
    orderedAt: columns.u32(d),
    pickedAt: columns.u32(d),
    deliveredAt: columns.u32(d),
    outcome: columns.u8(d),
  }
}

function storesFrom(raw: Uint8Array<ArrayBuffer>, header: TapeHeader): TapeStores {
  const columns = new Columns(raw)
  const series = header.stores * header.ticks
  const stores = {
    stock: columns.f32(series),
    orders: columns.u16(series),
    stockoutOffsets: columns.u32(header.stores + 1),
    stockoutIntervals: columns.u32(2 * header.stockoutPairs),
  }
  checkStockoutOffsets(stores.stockoutOffsets, header.stockoutPairs)
  return stores
}

function aggregatesFrom(raw: Uint8Array<ArrayBuffer>, header: TapeHeader): TapeAggregates {
  const columns = new Columns(raw)
  const ticks = header.ticks
  return {
    aware: columns.u32(ticks),
    adopted: columns.u32(ticks),
    orders: columns.u32(ticks),
    storesOut: columns.u16(ticks),
    rainIntensity: columns.f32(ticks),
  }
}

function checkStockoutOffsets(offsets: Uint32Array, stockoutPairs: number): void {
  if (offsets[0] !== 0) throw new TapeError('stockout_offsets must start at 0')
  for (let store = 1; store < offsets.length; store++) {
    if ((offsets[store] as number) < (offsets[store - 1] as number)) {
      throw new TapeError('stockout_offsets must be non-decreasing')
    }
  }
  if (offsets[offsets.length - 1] !== stockoutPairs) {
    throw new TapeError('stockout_offsets does not end at stockout_pairs')
  }
}

// Columns are padded to 8 bytes, so each is a view into the section's buffer with no copy.
// A section's size has already been checked against the header counts.
class Columns {
  readonly #buffer: ArrayBuffer
  #offset: number

  constructor(raw: Uint8Array<ArrayBuffer>) {
    const aligned = raw.byteOffset % ALIGN === 0 ? raw : raw.slice()
    this.#buffer = aligned.buffer
    this.#offset = aligned.byteOffset
  }

  u8(length: number): Uint8Array<ArrayBuffer> {
    return new Uint8Array(this.#buffer, this.#take(length), length)
  }

  u16(length: number): Uint16Array<ArrayBuffer> {
    return new Uint16Array(this.#buffer, this.#take(2 * length), length)
  }

  u32(length: number): Uint32Array<ArrayBuffer> {
    return new Uint32Array(this.#buffer, this.#take(4 * length), length)
  }

  f32(length: number): Float32Array<ArrayBuffer> {
    return new Float32Array(this.#buffer, this.#take(4 * length), length)
  }

  #take(bytes: number): number {
    const at = this.#offset
    this.#offset += padded(bytes)
    return at
  }
}

function padded(bytes: number): number {
  return Math.ceil(bytes / ALIGN) * ALIGN
}

function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
  return a.length === b.length && a.every((byte, index) => byte === b[index])
}
