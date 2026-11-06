import { readFileSync } from 'node:fs'
import { crc32 as zlibCrc32, zstdCompressSync, zstdDecompressSync } from 'node:zlib'

export const FIXTURE_DIR = new URL('../../../fixtures/tape/v1/', import.meta.url)
export const HEADER_SIZE = 352
export const FOOTER_SIZE = 40
export const REQUIRED_FIXTURES = ['small', 'empty', 'limits', 'engine_supply_small_seed1']

export const HEADER_FIELDS = {
  version: [4, 2],
  flags: [6, 2],
  seed: [24, 8],
  simMinutes: [128, 4],
  tickMinutes: [132, 4],
  residents: [136, 4],
  stores: [140, 4],
  chatterRows: [144, 4],
  deliveryRows: [148, 4],
  stockoutPairs: [152, 4],
  reserved: [156, 4],
} as const

const ROW_FIELDS = {
  kind: [0, 4],
  offset: [4, 8],
  length: [12, 8],
  rawLength: [20, 8],
  crc32: [28, 4],
} as const
const TABLE_START = 160
const ROW_SIZE = 32

export type HeaderField = keyof typeof HEADER_FIELDS
export type RowField = keyof typeof ROW_FIELDS

export function loadFixture(name: string): Uint8Array {
  return new Uint8Array(readFileSync(new URL(`${name}.twlt`, FIXTURE_DIR)))
}

export function loadProbes(name: string): Probes {
  return JSON.parse(readFileSync(new URL(`${name}.probes.json`, FIXTURE_DIR), 'utf8')) as Probes
}

export interface ProbeColumn {
  dtype: 'u8' | 'u16' | 'u32' | 'f32'
  length: number
  crc32: number
  sum?: string
  head: number[]
  tail: number[]
}

export interface Probes {
  file: { bytes: number; blake3: string }
  footer: { total_length: number; blake3: string }
  header: Record<string, number | string>
  sections: {
    kind: number
    name: string
    offset: number
    length: number
    raw_length: number
    crc32: number
  }[]
  columns: Record<string, Record<string, ProbeColumn>>
}

function view(bytes: Uint8Array): DataView {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
}

function write(bytes: Uint8Array, at: number, size: number, value: number | bigint): void {
  const dv = view(bytes)
  if (size === 2) dv.setUint16(at, Number(value), true)
  else if (size === 4) dv.setUint32(at, Number(value), true)
  else dv.setBigUint64(at, BigInt(value), true)
}

export function patchHeader(
  bytes: Uint8Array,
  field: HeaderField,
  value: number | bigint,
): Uint8Array {
  const out = bytes.slice()
  const [at, size] = HEADER_FIELDS[field]
  write(out, at, size, value)
  return out
}

export function readRowField(bytes: Uint8Array, row: number, field: RowField): number {
  const [at, size] = ROW_FIELDS[field]
  const dv = view(bytes)
  const pos = TABLE_START + row * ROW_SIZE + at
  return Number(size === 8 ? dv.getBigUint64(pos, true) : dv.getUint32(pos, true))
}

export function patchRow(
  bytes: Uint8Array,
  row: number,
  field: RowField,
  value: number,
): Uint8Array {
  const out = bytes.slice()
  const [at, size] = ROW_FIELDS[field]
  write(out, TABLE_START + row * ROW_SIZE + at, size, value)
  return out
}

export function patchFooterLength(bytes: Uint8Array, value: number): Uint8Array {
  const out = bytes.slice()
  write(out, out.length - FOOTER_SIZE, 8, value)
  return out
}

export function flipByte(bytes: Uint8Array, at: number): Uint8Array {
  const out = bytes.slice()
  out[at] = (out[at] ?? 0) ^ 0xff
  return out
}

export function headerValue(bytes: Uint8Array, field: HeaderField): number {
  const [at, size] = HEADER_FIELDS[field]
  const dv = view(bytes)
  return size === 2 ? dv.getUint16(at, true) : dv.getUint32(at, true)
}

export function compressedSection(bytes: Uint8Array, row: number): Uint8Array {
  const offset = readRowField(bytes, row, 'offset')
  return bytes.slice(offset, offset + readRowField(bytes, row, 'length'))
}

export function rawSection(bytes: Uint8Array, row: number): Uint8Array {
  return new Uint8Array(zstdDecompressSync(compressedSection(bytes, row)))
}

export function compress(raw: Uint8Array): Uint8Array {
  return new Uint8Array(zstdCompressSync(raw, { pledgedSrcSize: raw.length }))
}

export function crc(bytes: Uint8Array): number {
  return zlibCrc32(bytes)
}

// Rebuilds the file with some sections replaced, a fresh table and footer length; the footer hash is left zero.
export function replaceRawSection(bytes: Uint8Array, row: number, raw: Uint8Array): Uint8Array {
  const frames = Array.from({ length: 6 }, (_, i) =>
    i === row ? compress(raw) : compressedSection(bytes, i),
  )
  const rawLengths = Array.from({ length: 6 }, (_, i) =>
    i === row ? raw.length : readRowField(bytes, i, 'rawLength'),
  )
  const parts: Uint8Array[] = [bytes.slice(0, HEADER_SIZE)]
  let cursor = HEADER_SIZE
  frames.forEach((frame, i) => {
    const head = parts[0] as Uint8Array
    write(head, TABLE_START + i * ROW_SIZE + 4, 8, cursor)
    write(head, TABLE_START + i * ROW_SIZE + 12, 8, frame.length)
    write(head, TABLE_START + i * ROW_SIZE + 20, 8, rawLengths[i] as number)
    write(head, TABLE_START + i * ROW_SIZE + 28, 4, crc(frame))
    const padded = new Uint8Array(frame.length + ((8 - (frame.length % 8)) % 8))
    padded.set(frame)
    parts.push(padded)
    cursor += padded.length
  })
  const out = new Uint8Array(cursor + FOOTER_SIZE)
  let at = 0
  for (const part of parts) {
    out.set(part, at)
    at += part.length
  }
  write(out, cursor, 8, cursor)
  return out
}

export function hex(bytes: Uint8Array): string {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

export function camel(name: string): string {
  return name.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase())
}

export function hasContentSize(frame: Uint8Array): boolean {
  const descriptor = frame[4] ?? 0
  return (descriptor & 0x20) !== 0 || descriptor >> 6 !== 0
}

// A valid single raw-block zstd frame whose header carries no content size.
export function frameWithoutContentSize(data: Uint8Array): Uint8Array {
  if (data.length > 1000) throw new Error('fixture frame is for small data')
  const out = new Uint8Array(4 + 1 + 1 + 3 + data.length)
  out.set([0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x28])
  const block = (data.length << 3) | 1
  out.set([block & 0xff, (block >> 8) & 0xff, (block >> 16) & 0xff], 6)
  out.set(data, 9)
  return out
}
