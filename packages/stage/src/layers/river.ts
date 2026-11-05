import {
  abs,
  attribute,
  cameraPosition,
  clamp,
  color,
  Fn,
  float,
  normalize,
  oneMinus,
  positionWorld,
  pow,
  sin,
  smoothstep,
} from 'three/tsl'
import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  Mesh,
  MeshBasicNodeMaterial,
  type Node,
  ShapeUtils,
  Vector2,
} from 'three/webgpu'
import type { StageCity } from '../inputs'
import { cityGlow, palette } from '../palette'
import { AMBIENT_RATE } from '../time'
import { type Fog, viewDepth } from './fog'
import type { Layer } from './layer'

// Toy-tuned: the sheen fades out over this margin, about a fifth of the toy river's width, which
// leaves a dark line along each bank.
const BANK_MARGIN_M = 40
// Long bands drifting along the water, at whole multiples of the ambient rate.
const SHEEN_RATE = 10 * AMBIENT_RATE
const SHEEN_SWELL_RATE = 5 * AMBIENT_RATE
const MIDPOINT_KEY_BASE = 2 ** 26
/** A ring's triangles stay within four times this, so a very large water body cannot stall load. */
export const RING_TRIANGLE_TARGET = 50_000

export function createRiver(
  river: StageCity['river'],
  fog: Fog,
  warm: Node<'float'>,
  clock: Node<'float'>,
  reducedMotion: boolean,
  scale = 1,
): Layer[] {
  const margin = BANK_MARGIN_M * scale
  const { positions, banks, indices } = waterMesh(river, margin)
  if (indices.length === 0) return []
  const geometry = new BufferGeometry()
    .setAttribute('position', new BufferAttribute(positions, 3))
    .setAttribute('bank', new BufferAttribute(banks, 1))
    .setIndex(new BufferAttribute(indices, 1))
  const material = new MeshBasicNodeMaterial({ depthWrite: false, side: DoubleSide })
  const time = reducedMotion ? float(0) : clock
  material.colorNode = Fn(() => {
    const { x, z } = positionWorld
    const drift = z
      .mul(0.018 / scale)
      .add(time.mul(SHEEN_RATE))
      .add(sin(x.mul(0.04 / scale)).mul(1.5))
    const swell = z.mul(0.0061 / scale).sub(time.mul(SHEEN_SWELL_RATE))
    const sheen = sin(drift).mul(sin(swell)).mul(0.5).add(0.5)
    const edge = smoothstep(0, margin, attribute<'float'>('bank'))
    const fogged = fog.factor(viewDepth)
    const clear = oneMinus(fogged)
    const grazing = pow(
      clamp(oneMinus(abs(normalize(cameraPosition.sub(positionWorld)).y)), 0, 1),
      3,
    )
    const water = cityGlow(warm)
      .mul(0.035)
      .mul(edge)
      .add(color(palette.riverSheen).mul(sheen).mul(0.16).mul(edge).mul(clear))
      .add(color(palette.riverBand).mul(grazing).mul(0.3).mul(clear).mul(clear))
      .add(color(palette.riverDeep))
    return fog.mix(water, fogged)
  })()
  const mesh = new Mesh(geometry, material)
  mesh.renderOrder = -4
  mesh.frustumCulled = false
  return [
    {
      object: mesh,
      dispose: () => {
        geometry.dispose()
        material.dispose()
      },
    },
  ]
}

/**
 * Water rings triangulated flat on the ground in three.js axes, no edge longer than `maxEdge`, with
 * each vertex's distance to its bank. A ring's own points all sit on the bank, so without the extra
 * vertices that distance could not vary across the water.
 */
export function waterMesh(
  river: StageCity['river'],
  maxEdge = Infinity,
): { positions: Float32Array; banks: Float32Array; indices: Uint32Array } {
  const positions: number[] = []
  const banks: number[] = []
  const indices: number[] = []
  const polygons = (river?.offsets.length ?? 1) - 1
  for (let p = 0; p < polygons && river; p++) {
    const ring: Vector2[] = []
    for (let i = river.offsets[p] ?? 0; i < (river.offsets[p + 1] ?? 0); i++) {
      ring.push(new Vector2(river.x[i] ?? 0, river.y[i] ?? 0))
    }
    // Drops a closing point that repeats the first, so read the ring only after this.
    const faces = ShapeUtils.triangulateShape(ring, [])
    const coarsest = Math.sqrt((2 * Math.abs(ShapeUtils.area(ring))) / RING_TRIANGLE_TARGET)
    const refined = refine(ring, faces, Math.max(maxEdge, coarsest))
    const first = positions.length / 3
    refined.x.forEach((x, i) => {
      positions.push(x, 0, -(refined.y[i] ?? 0))
      banks.push(refined.banks[i] ?? 0)
    })
    for (const corner of refined.triangles) indices.push(first + corner)
  }
  return {
    positions: Float32Array.from(positions),
    banks: Float32Array.from(banks),
    indices: Uint32Array.from(indices),
  }
}

// Splits each triangle's longest edge until none is too long. Every edge that is too long ends up
// split, and both of its triangles share that midpoint, so the mesh has no cracks.
function refine(ring: Vector2[], faces: number[][], maxEdge: number) {
  const x = ring.map((point) => point.x)
  const y = ring.map((point) => point.y)
  const banks = ring.map(() => 0)
  const triangles: number[] = []
  const midpoints = new Map<number, number>()
  const limit = maxEdge * maxEdge
  const squaredLength = (a: number, b: number) =>
    ((x[a] ?? 0) - (x[b] ?? 0)) ** 2 + ((y[a] ?? 0) - (y[b] ?? 0)) ** 2
  const midpointOf = (a: number, b: number) => {
    const key = Math.min(a, b) * MIDPOINT_KEY_BASE + Math.max(a, b)
    const known = midpoints.get(key)
    if (known !== undefined) return known
    const mx = ((x[a] ?? 0) + (x[b] ?? 0)) / 2
    const my = ((y[a] ?? 0) + (y[b] ?? 0)) / 2
    const created = x.length
    x.push(mx)
    y.push(my)
    banks.push(distanceToRing(mx, my, ring))
    midpoints.set(key, created)
    return created
  }
  const pending = [...faces]
  while (pending.length > 0) {
    const [a = 0, b = 0, c = 0] = pending.pop() ?? []
    const ab = squaredLength(a, b)
    const bc = squaredLength(b, c)
    const ca = squaredLength(c, a)
    const longest = Math.max(ab, bc, ca)
    if (longest <= limit) {
      triangles.push(a, b, c)
      continue
    }
    const [p, q, r] = longest === ab ? [a, b, c] : longest === bc ? [b, c, a] : [c, a, b]
    const m = midpointOf(p, q)
    pending.push([p, m, r], [m, q, r])
  }
  return { x, y, banks, triangles }
}

function distanceToRing(x: number, y: number, ring: Vector2[]): number {
  let nearest = Infinity
  let from = ring.at(-1)
  for (const to of ring) {
    if (from) {
      const dx = to.x - from.x
      const dy = to.y - from.y
      const along = ((x - from.x) * dx + (y - from.y) * dy) / (dx * dx + dy * dy || 1)
      const t = Math.min(Math.max(along, 0), 1)
      nearest = Math.min(nearest, (x - from.x - t * dx) ** 2 + (y - from.y - t * dy) ** 2)
    }
    from = to
  }
  return Math.sqrt(nearest)
}
