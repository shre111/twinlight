/** Event time meaning the event never happened (u32::MAX in the tape). */
export const NEVER = 0xffff_ffff

export const ROAD_KIND = { street: 0, arterial: 1, bridge: 2 } as const

/**
 * A city in pack coordinates (ARCHITECTURE.md section 5): local metres from the bbox centre,
 * x east, y north, z up. Rotations are radians counter-clockwise from east.
 */
export interface StageCity {
  buildings: {
    count: number
    x: Float32Array
    y: Float32Array
    width: Float32Array
    depth: Float32Array
    height: Float32Array
    rotation: Float32Array
    /** v2 packs: tile i holds buildings tileStart[i] to tileStart[i + 1], in Morton order. */
    tileStart?: Uint32Array
  }
  /** Segments as x0, y0, x1, y1, with one `ROAD_KIND` per segment. */
  roads: { segments: Float32Array; kind: Uint8Array }
  /**
   * Water polygons. Ring r owns points offsets[r] to offsets[r + 1]. Polygon p takes its next
   * rings[p] rings, the first its exterior and the rest holes; without `rings`, each ring is a polygon.
   */
  river?: { offsets: Uint32Array; x: Float32Array; y: Float32Array; rings?: Uint16Array }
  residents: {
    count: number
    x: Float32Array
    y: Float32Array
    z: Float32Array
    building: Uint32Array
  }
  stores: { count: number; x: Float32Array; y: Float32Array; radius: Float32Array }
  /** The pack manifest's bbox; the stage fits its camera and ground to it (or to the buildings). */
  bounds?: { minX: number; minY: number; maxX: number; maxY: number }
}

/** The subset of a decoded tape (ARCHITECTURE.md section 4) that the stage draws. Times in minutes. */
export interface StageTape {
  simMinutes: number
  tickMinutes: number
  residents: { awareAt: Uint32Array; firstOrderAt: Uint32Array; orderCount: Uint16Array }
  /** Store s owns intervals stockoutOffsets[s] to stockoutOffsets[s + 1], each a (start, end) pair. */
  stores: { stockoutOffsets: Uint32Array; stockoutIntervals: Uint32Array }
  aggregates: {
    aware: Uint32Array
    adopted: Uint32Array
    orders: Uint32Array
    storesOut: Uint16Array
    rainIntensity: Float32Array
  }
}
