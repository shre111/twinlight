import type { StageCity } from './inputs'

/** Each column's buffer once: columns are views into their file, and a repeated buffer cannot be transferred. */
export function cityTransfers(city: StageCity): ArrayBuffer[] {
  const buffers = new Set<ArrayBuffer>()
  const visit = (value: unknown) => {
    if (ArrayBuffer.isView(value)) buffers.add(value.buffer as ArrayBuffer)
    else if (value && typeof value === 'object') Object.values(value).forEach(visit)
  }
  visit(city)
  return [...buffers]
}
