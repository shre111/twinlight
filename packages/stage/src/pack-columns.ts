import type { StageCity } from './inputs'

// Every column starts on an 8-byte boundary and is zero-padded to a multiple of 8, so each one
// is a typed-array view into the file, and the file size is the sum of the padded columns.
export class Columns {
  readonly #buffer: ArrayBuffer
  readonly #file: string
  #offset = 0

  constructor(buffer: ArrayBuffer, file: string) {
    this.#buffer = buffer
    this.#file = file
  }

  f32(n: number): Float32Array {
    return new Float32Array(this.#buffer, this.#take(n * 4), n)
  }

  u32(n: number): Uint32Array {
    return new Uint32Array(this.#buffer, this.#take(n * 4), n)
  }

  u16(n: number): Uint16Array {
    return new Uint16Array(this.#buffer, this.#take(n * 2), n)
  }

  u8(n: number): Uint8Array {
    return new Uint8Array(this.#buffer, this.#take(n), n)
  }

  end(): void {
    if (this.#offset !== this.#buffer.byteLength) {
      throw new Error(
        `${this.#file} is ${this.#buffer.byteLength} bytes but its columns take ${this.#offset}`,
      )
    }
  }

  #take(bytes: number): number {
    const at = this.#offset
    this.#offset += Math.ceil(bytes / 8) * 8
    if (this.#offset > this.#buffer.byteLength) throw new Error(`${this.#file} is truncated`)
    return at
  }
}

export function checkOffsets(
  file: string,
  what: string,
  offsets: Uint32Array,
  points: number,
): void {
  const count = offsets.length - 1
  if (offsets[0] !== 0 || offsets[count] !== points) {
    throw new Error(`${file} offsets must run from 0 to ${points} ${what} points`)
  }
  for (let i = 0; i < count; i++) {
    if ((offsets[i + 1] ?? 0) < (offsets[i] ?? 0)) {
      throw new Error(`${file} offsets go backwards at ${what} ${i}`)
    }
  }
}

// Roads are polylines; the stage draws each pair of consecutive points as a segment.
export function polylineSegments(
  offsets: Uint32Array,
  x: Float32Array,
  y: Float32Array,
  roadClass: Uint8Array,
): StageCity['roads'] {
  const roads = offsets.length - 1
  let total = 0
  for (let r = 0; r < roads; r++) {
    total += Math.max(0, (offsets[r + 1] ?? 0) - (offsets[r] ?? 0) - 1)
  }
  const segments = new Float32Array(total * 4)
  const kind = new Uint8Array(total)
  let s = 0
  for (let r = 0; r < roads; r++) {
    const last = (offsets[r + 1] ?? 0) - 1
    for (let p = offsets[r] ?? 0; p < last; p++, s++) {
      segments[s * 4] = x[p] ?? 0
      segments[s * 4 + 1] = y[p] ?? 0
      segments[s * 4 + 2] = x[p + 1] ?? 0
      segments[s * 4 + 3] = y[p + 1] ?? 0
      kind[s] = roadClass[r] ?? 0
    }
  }
  return { segments, kind }
}

export function decodeStores(buffer: ArrayBuffer, count: number): StageCity['stores'] {
  const c = new Columns(buffer, 'stores.bin')
  const x = c.f32(count)
  const y = c.f32(count)
  c.u32(count) // hex
  const radius = c.f32(count)
  c.u16(count) // district
  c.end()
  return { count, x, y, radius }
}
