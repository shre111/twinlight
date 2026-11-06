import { crc32 as nodeCrc32, zstdCompressSync } from 'node:zlib'
import { describe, expect, it } from 'vitest'
import { crc32, decompressZstd } from '../src/index'
import { frameWithoutContentSize, hasContentSize } from './helpers'

function sample(length: number, salt: number): Uint8Array {
  return Uint8Array.from({ length }, (_, i) => (i * 31 + salt + (i >> 7)) & 0xff)
}

function frame(data: Uint8Array): Uint8Array {
  const out = new Uint8Array(zstdCompressSync(data, { pledgedSrcSize: data.length }))
  expect(hasContentSize(out)).toBe(true)
  return out
}

describe('decompressZstd', () => {
  const data = sample(50_000, 7)
  const packed = frame(data)

  it('round trips with and without rawLength', async () => {
    expect(await decompressZstd(packed)).toEqual(data)
    expect(await decompressZstd(packed, data.length)).toEqual(data)
  })

  it('gives an empty array for an empty frame', async () => {
    const empty = frame(new Uint8Array(0))
    expect((await decompressZstd(empty)).length).toBe(0)
    expect((await decompressZstd(empty, 0)).length).toBe(0)
  })

  it('rejects a rawLength that disagrees with the frame', async () => {
    await expect(decompressZstd(packed, data.length + 1)).rejects.toThrow()
    await expect(decompressZstd(packed, data.length - 1)).rejects.toThrow()
  })

  it('rejects garbage and a truncated frame', async () => {
    await expect(decompressZstd(sample(64, 3))).rejects.toThrow()
    await expect(decompressZstd(packed.slice(0, packed.length >> 1))).rejects.toThrow()
    await expect(decompressZstd(new Uint8Array(0))).rejects.toThrow()
  })

  it('needs rawLength for a frame without content size', async () => {
    const small = sample(100, 5)
    const bare = frameWithoutContentSize(small)
    expect(hasContentSize(bare)).toBe(false)
    expect(await decompressZstd(bare, small.length)).toEqual(small)
    await expect(decompressZstd(bare)).rejects.toThrow()
  })

  it('resolves two calls in flight with the right bytes each', async () => {
    const other = sample(30_000, 99)
    const [a, b] = await Promise.all([decompressZstd(packed), decompressZstd(frame(other))])
    expect(a).toEqual(data)
    expect(b).toEqual(other)
  })

  it('returns bytes that later decodes do not overwrite', async () => {
    const first = await decompressZstd(packed)
    const snapshot = first.slice()
    await decompressZstd(frame(sample(50_000, 200)))
    expect(first).toEqual(snapshot)
  })
})

describe('crc32', () => {
  it.each([
    ['empty input', new Uint8Array(0)],
    ['a short string', new TextEncoder().encode('twinlight')],
    ['1 MB of patterned bytes', sample(1 << 20, 11)],
  ])('equals zlib crc32 for %s', (_, bytes) => {
    expect(crc32(bytes)).toBe(nodeCrc32(bytes))
    expect(crc32(bytes)).toBeGreaterThanOrEqual(0)
  })
})
