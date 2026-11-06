import { ZSTDDecoder } from 'zstddec'

const FRAME_MAGIC = 0xfd2fb528
const MAX_RAW_BYTES = 2 ** 28
const DICTIONARY_ID_BYTES = [0, 1, 2, 4] as const
const EMPTY_LAST_BLOCK = [1, 0, 0]
const EMPTY_CHECKSUM = [0x99, 0xe9, 0xd8, 0x51]

let ready: Promise<ZSTDDecoder> | undefined

function decoder(): Promise<ZSTDDecoder> {
  ready ??= (async () => {
    const zstd = new ZSTDDecoder()
    await zstd.init()
    return zstd
  })()
  return ready
}

interface FrameHeader {
  contentSize: number | undefined
  headerBytes: number
  hasChecksum: boolean
}

// RFC 8878 section 3.1.1.1: magic, descriptor, optional window byte, dictionary id, content size.
function readFrameHeader(bytes: Uint8Array): FrameHeader {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (bytes.length < 6 || view.getUint32(0, true) !== FRAME_MAGIC) {
    throw new Error('not a zstd frame')
  }
  const descriptor = view.getUint8(4)
  const sizeFlag = descriptor >>> 6
  const singleSegment = (descriptor & 0x20) !== 0
  const at = 5 + (singleSegment ? 0 : 1) + (DICTIONARY_ID_BYTES[descriptor & 3] ?? 0)
  const sizeBytes = sizeFlag === 0 ? (singleSegment ? 1 : 0) : 1 << sizeFlag
  if (at + sizeBytes > bytes.length) throw new Error('zstd frame header is truncated')
  let contentSize: number | undefined
  if (sizeBytes === 1) contentSize = view.getUint8(at)
  else if (sizeBytes === 2) contentSize = view.getUint16(at, true) + 256
  else if (sizeBytes === 4) contentSize = view.getUint32(at, true)
  else if (sizeBytes === 8) contentSize = Number(view.getBigUint64(at, true))
  return { contentSize, headerBytes: at + sizeBytes, hasChecksum: (descriptor & 4) !== 0 }
}

// A failed decode and an empty result look the same from zstddec, so an empty frame is checked
// by shape instead: the header, one empty raw last block, and the checksum of nothing if flagged.
function checkEmptyFrame(bytes: Uint8Array, header: FrameHeader): void {
  const expected = header.hasChecksum ? [...EMPTY_LAST_BLOCK, ...EMPTY_CHECKSUM] : EMPTY_LAST_BLOCK
  const rest = bytes.subarray(header.headerBytes)
  if (rest.length !== expected.length || rest.some((byte, index) => byte !== expected[index])) {
    throw new Error('zstd frame is corrupt')
  }
}

/**
 * Decompress one zstd frame. The output size is the frame's content size; `rawLength`, when
 * given, must agree with it, and stands in for it when the frame does not carry one.
 */
export async function decompressZstd(
  bytes: Uint8Array,
  rawLength?: number,
): Promise<Uint8Array<ArrayBuffer>> {
  const header = readFrameHeader(bytes)
  const framed = header.contentSize
  if (framed !== undefined && rawLength !== undefined && framed !== rawLength) {
    throw new Error(`zstd frame holds ${framed} bytes, expected ${rawLength}`)
  }
  const size = framed ?? rawLength
  if (size === undefined) {
    throw new Error('zstd frame has no content size and no raw length was given')
  }
  if (!Number.isSafeInteger(size) || size < 0 || size > MAX_RAW_BYTES) {
    throw new Error(`zstd frame claims ${size} bytes, over the ${MAX_RAW_BYTES} byte limit`)
  }
  if (size === 0) {
    checkEmptyFrame(bytes, header)
    return new Uint8Array(0)
  }
  const zstd = await decoder()
  const raw = zstd.decode(bytes, size)
  if (raw.length !== size) throw new Error('zstd frame is corrupt or truncated')
  return raw as Uint8Array<ArrayBuffer>
}
