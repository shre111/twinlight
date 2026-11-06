import { constants, crc32 as nodeCrc32, zstdCompressSync } from 'node:zlib'
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

function checksummed(data: Uint8Array): Uint8Array {
  const params = { [constants.ZSTD_c_checksumFlag]: 1 }
  const out = new Uint8Array(zstdCompressSync(data, { params, pledgedSrcSize: data.length }))
  expect((out[4] ?? 0) & 4).toBe(4)
  return out
}

function bare(length: number): Uint8Array {
  return frameWithoutContentSize(sample(length, 1))
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

  it('rejects a header cut off before its content size', async () => {
    await expect(decompressZstd(packed.slice(0, 6))).rejects.toThrow()
  })

  it('rejects a second frame or garbage after the frame', async () => {
    const twice = new Uint8Array([...packed, ...packed])
    await expect(decompressZstd(twice)).rejects.toThrow()
    await expect(decompressZstd(new Uint8Array([...packed, 1, 2, 3, 4, 5, 6, 7]))).rejects.toThrow()
  })

  it('rejects a rawLength of 0 for a frame that holds bytes', async () => {
    await expect(decompressZstd(packed, 0)).rejects.toThrow()
  })

  it('rejects a frame that claims more than the size limit', async () => {
    const huge = new Uint8Array([0x28, 0xb5, 0x2f, 0xfd, 0xe0, 0, 0, 0, 0, 0, 0, 0x20, 0, 1, 0, 0])
    await expect(decompressZstd(huge)).rejects.toThrow(/limit/)
    await expect(decompressZstd(bare(1), 2 ** 28 + 1)).rejects.toThrow(/limit/)
  })

  it('checks the checksum of a frame that carries one', async () => {
    const sealed = checksummed(data)
    expect(await decompressZstd(sealed)).toEqual(data)
    const broken = sealed.slice()
    broken[broken.length - 1] = (broken.at(-1) ?? 0) ^ 0xff
    await expect(decompressZstd(broken)).rejects.toThrow()
  })

  it('checks the checksum of an empty frame', async () => {
    const sealed = checksummed(new Uint8Array(0))
    expect((await decompressZstd(sealed)).length).toBe(0)
    const broken = sealed.slice()
    broken[broken.length - 1] = (broken.at(-1) ?? 0) ^ 0xff
    await expect(decompressZstd(broken)).rejects.toThrow()
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
