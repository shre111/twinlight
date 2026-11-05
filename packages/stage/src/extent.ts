import type { StageCity } from './inputs'

export type Bounds = NonNullable<StageCity['bounds']>

/** A city's footprint in three.js axes, and its size relative to the toy city the stage was tuned on. */
export interface CityExtent {
  centreX: number
  centreZ: number
  /** The longer side, in metres. */
  size: number
  /** `size` over the toy city's; every hand-tuned distance in the stage is multiplied by it. */
  scale: number
}

// The prototype's toy city (pack bbox -3747, -3160 to 3703, 3400): poses, clamps and the ground
// glow were tuned by eye against it.
export const TOY_BOUNDS: Bounds = { minX: -3747, minY: -3160, maxX: 3703, maxY: 3400 }

export function cityBounds(city: StageCity): Bounds {
  if (city.bounds) return city.bounds
  const { x, y, count } = city.buildings
  if (count === 0) return TOY_BOUNDS
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (let i = 0; i < count; i++) {
    minX = Math.min(minX, x[i] ?? 0)
    maxX = Math.max(maxX, x[i] ?? 0)
    minY = Math.min(minY, y[i] ?? 0)
    maxY = Math.max(maxY, y[i] ?? 0)
  }
  return { minX, minY, maxX, maxY }
}

const longerSide = (b: Bounds) => Math.max(b.maxX - b.minX, b.maxY - b.minY, 1)
const TOY_SIZE = longerSide(TOY_BOUNDS)

export function extentOf(bounds: Bounds): CityExtent {
  const size = longerSide(bounds)
  return {
    centreX: (bounds.minX + bounds.maxX) / 2,
    centreZ: -(bounds.minY + bounds.maxY) / 2,
    size,
    scale: size / TOY_SIZE,
  }
}

export const TOY_EXTENT = extentOf(TOY_BOUNDS)

/** A toy-tuned point (three.js x, z), moved and scaled onto another city's extent. */
export function fitPoint(x: number, z: number, extent: CityExtent): [number, number] {
  return [
    extent.centreX + (x - TOY_EXTENT.centreX) * extent.scale,
    extent.centreZ + (z - TOY_EXTENT.centreZ) * extent.scale,
  ]
}
