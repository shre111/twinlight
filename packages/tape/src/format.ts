export const TAPE_VERSION = 1
export const NEVER = 0xffffffff

export const CHATTER_HEARD = 1
export const CHATTER_NUDGED = 2

export const SOURCE_NONE = 0
export const SOURCE_PAID = 1
export const SOURCE_WOM = 2
export const SOURCE_SEEN = 3

export const FLAG_STOCKOUT = 1
export const FLAG_SPILLED = 2
export const FLAG_GAVE_UP = 4

export const DELIVERY_LATE = 1
export const DELIVERY_SPILLED = 2

export const SECTION_NAMES = [
  'RESIDENTS',
  'CHATTER',
  'DELIVERIES',
  'STORES',
  'AGGREGATES',
  'CHUNKS',
] as const

/** The bytes are not a well-formed, intact, complete tape. */
export class TapeError extends Error {
  override name = 'TapeError'
}

type U8 = Uint8Array<ArrayBuffer>
type U16 = Uint16Array<ArrayBuffer>
type U32 = Uint32Array<ArrayBuffer>
type F32 = Float32Array<ArrayBuffer>

export interface TapeHeader {
  version: number
  flags: number
  runId: Uint8Array
  seed: bigint
  engineVersion: string
  cityPackHash: Uint8Array
  scenarioHash: Uint8Array
  simMinutes: number
  tickMinutes: number
  ticks: number
  residents: number
  stores: number
  chatterRows: number
  deliveryRows: number
  stockoutPairs: number
}

export interface TapeSection {
  kind: number
  offset: number
  length: number
  rawLength: number
  crc32: number
}

/** Times are sim minutes; `NEVER` marks one that did not happen. In city pack order. */
export interface TapeResidents {
  awareAt: U32
  firstOrderAt: U32
  orderCount: U16
  heardFrom: U32
  firstSource: U8
  outcomeFlags: U8
}

/** Sorted by `at`. */
export interface TapeChatter {
  from: U32
  to: U32
  at: U32
  kind: U8
}

/** One row per served order, sorted by `orderedAt`. */
export interface TapeDeliveries {
  store: U16
  resident: U32
  orderedAt: U32
  pickedAt: U32
  deliveredAt: U32
  outcome: U8
}

/**
 * `stock` and `orders` are row-major `[store * ticks + tick]`. Store s owns the (start, end)
 * pairs `stockoutOffsets[s]` to `stockoutOffsets[s + 1]` of `stockoutIntervals`, end-exclusive.
 */
export interface TapeStores {
  stock: F32
  orders: U16
  stockoutOffsets: U32
  stockoutIntervals: U32
}

/** One value per tick. */
export interface TapeAggregates {
  aware: U32
  adopted: U32
  orders: U32
  storesOut: U16
  rainIntensity: F32
}

/** Each section's columns are views into one buffer of their own, so a section can be transferred. */
export interface Tape {
  header: TapeHeader
  sections: TapeSection[]
  /** The blake3 of the tape's body, as stored in its footer. */
  hash: Uint8Array
  residents: TapeResidents
  chatter: TapeChatter
  deliveries: TapeDeliveries
  stores: TapeStores
  aggregates: TapeAggregates
}
