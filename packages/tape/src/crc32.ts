const TABLE = new Uint32Array(256)
for (let n = 0; n < 256; n++) {
  let c = n
  for (let bit = 0; bit < 8; bit++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  TABLE[n] = c
}

/** The zlib crc32, as an unsigned 32-bit number. */
export function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff
  for (let i = 0; i < bytes.length; i++) {
    c = (TABLE[(c ^ (bytes[i] as number)) & 0xff] as number) ^ (c >>> 8)
  }
  return (c ^ 0xffffffff) >>> 0
}
