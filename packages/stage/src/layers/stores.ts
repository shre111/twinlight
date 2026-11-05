import {
  abs,
  attribute,
  clamp,
  color,
  dot,
  exp,
  float,
  fract,
  instanceIndex,
  length,
  max,
  mix,
  modelViewMatrix,
  normalView,
  oneMinus,
  positionGeometry,
  positionViewDirection,
  sin,
  smoothstep,
  uniform,
  uv,
  vec3,
  vec4,
  vertexStage,
  viewportSize,
} from 'three/tsl'
import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  CylinderGeometry,
  Group,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  LineBasicNodeMaterial,
  LineSegments,
  Mesh,
  MeshBasicNodeMaterial,
  type Node,
  type NodeMaterial,
  type PerspectiveCamera,
  PlaneGeometry,
} from 'three/webgpu'
import type { StageCity, StageTape } from '../inputs'
import { palette } from '../palette'
import { updateStockouts } from '../stockouts'
import { AMBIENT_RATE } from '../time'
import { type Fog, viewDepth } from './fog'

type Stores = StageCity['stores']

const BEAM_RADIUS = 7
const BEAM_HEIGHT = 520
const BEAM_SEGMENTS = 16
// Beams never draw thinner than this in physical pixels, so far ones dim instead of breaking up.
const MIN_BEAM_PX = 1.5
const BASE_SIZE = 260
const BASE_LIFT = 2
const RING_LIFT = 1.5
const RING_SEGMENTS = 128
const RING_GAIN = 0.085
const OUT_GAIN = 1.15
// Out-of-stock beams pulse by this much around their gain, roughly once a second.
const FLICKER_DEPTH = 0.28
const FLICKER_RATE = 100 * AMBIENT_RATE
const FLICKER_PHASE_STEP = 1.7
// Reduced motion holds the flicker, so out-of-stock beams break into dashes instead: a still
// cue that is not colour alone. Four dashes with dark gaps stay legible at the default view.
const SEGMENT_METRES = 130
const SEGMENT_GAP_GAIN = 0

/** Dark stores as tube beams over a ground glow and a delivery-radius ring, three draws in all. */
export class StoresLayer {
  readonly object = new Group()
  readonly count: number
  readonly #focal = uniform(1)
  readonly #stockout: InstancedBufferAttribute
  readonly #geometries: BufferGeometry[] = []
  readonly #materials: NodeMaterial[] = []

  /** `clock` is wall-clock seconds for the flicker, which reduced motion holds steady. */
  /** `heightScale` stretches the toy-tuned beam height for larger cities; widths stay in metres. */
  constructor(
    stores: Stores,
    fog: Fog,
    clock: Node<'float'>,
    reducedMotion: boolean,
    heightScale = 1,
  ) {
    this.count = stores.count
    this.#stockout = new InstancedBufferAttribute(new Float32Array(stores.count), 1)
    const centre = new InstancedBufferAttribute(storeCentres(stores), 3)
    const radius = new InstancedBufferAttribute(stores.radius.slice(0, stores.count), 1)
    const shared = { centre, stockout: this.#stockout, radius }

    const out = vertexStage(attribute('stockout', 'float'))
    const tint = mix(color(palette.tube), color(palette.sindoor), out)
    const flicker = reducedMotion ? float(1) : flickerAt(clock)
    const gain = vertexStage(mix(float(1), flicker.mul(OUT_GAIN), out))
    const clear = (amount: number) => vertexStage(oneMinus(fog.factor(viewDepth).mul(amount)))
    const segmented = reducedMotion ? out : null

    this.#mount(this.#beams(shared, tint.mul(gain).mul(clear(0.8)), segmented), 2)
    this.#mount(this.#bases(shared, tint.mul(gain).mul(clear(1))), 1)
    this.#mount(this.#rings(shared, tint.mul(clear(1)).mul(RING_GAIN)), 1)
    this.object.scale.y = heightScale
  }

  update(camera: PerspectiveCamera): void {
    this.#focal.value = camera.projectionMatrix.elements[5] ?? 1
  }

  /** Marks each store out of stock or not at `minute`; the GPU copy changes only when a store does. */
  showStockouts(stockouts: StageTape['stores'], minute: number): void {
    if (updateStockouts(stockouts, minute, this.#stockout.array as Float32Array)) {
      this.#stockout.needsUpdate = true
    }
  }

  dispose(): void {
    for (const geometry of this.#geometries) geometry.dispose()
    for (const material of this.#materials) material.dispose()
    this.object.removeFromParent()
  }

  #beams(shared: SharedAttributes, shade: Node<'vec3'>, segmented: Node<'float'> | null): Mesh {
    const geometry = instanced(beamShape(), shared, this.count)
    const material = additive(new MeshBasicNodeMaterial())
    const centre = attribute('centre', 'vec3')
    const depth = max(modelViewMatrix.mul(vec4(centre.add(positionGeometry), 1)).z.negate(), 1)
    const pixelsPerMetre = this.#focal.mul(viewportSize.y).mul(0.5).div(depth)
    const widen = max(1, float(MIN_BEAM_PX).div(max(pixelsPerMetre.mul(BEAM_RADIUS * 2), 1e-4)))
    material.positionNode = centre.add(positionGeometry.mul(vec3(widen, 1, widen)))
    const thin = vertexStage(float(1).div(widen))
    const facing = clamp(abs(dot(normalView, positionViewDirection)), 0, 1)
    const up = uv().y
    const fall = oneMinus(up)
    const alpha = fall
      .mul(fall)
      .mul(facing.mul(facing).mul(0.85).add(0.08))
      .add(exp(up.mul(-30)).mul(0.6))
    const lit = shade.mul(alpha).mul(thin).mul(0.95)
    material.colorNode = segmented ? lit.mul(mix(float(1), beamSegments(up), segmented)) : lit
    return new Mesh(geometry, material)
  }

  #bases(shared: SharedAttributes, shade: Node<'vec3'>): Mesh {
    const plane = new PlaneGeometry(BASE_SIZE, BASE_SIZE).rotateX(-Math.PI / 2)
    const geometry = instanced(plane, shared, this.count)
    const material = additive(new MeshBasicNodeMaterial())
    material.positionNode = attribute('centre', 'vec3')
      .add(positionGeometry)
      .add(vec3(0, BASE_LIFT, 0))
    const r = length(uv().sub(0.5)).mul(2)
    const r2 = r.mul(r)
    const alpha = exp(r2.mul(-9))
      .mul(0.55)
      .add(exp(r2.mul(-160)).mul(1.2))
    material.colorNode = shade.mul(alpha)
    return new Mesh(geometry, material)
  }

  #rings(shared: SharedAttributes, shade: Node<'vec3'>): LineSegments {
    const geometry = instanced(unitRing(), shared, this.count)
    const material = additive(new LineBasicNodeMaterial())
    const radius = attribute('radius', 'float')
    material.positionNode = attribute('centre', 'vec3').add(
      positionGeometry.mul(vec3(radius, 1, radius)),
    )
    material.colorNode = shade
    return new LineSegments(geometry, material)
  }

  #mount(object: Mesh | LineSegments, renderOrder: number): void {
    object.frustumCulled = false
    object.renderOrder = renderOrder
    this.#geometries.push(object.geometry)
    this.#materials.push(object.material as NodeMaterial)
    this.object.add(object)
  }
}

function flickerAt(clock: Node<'float'>) {
  const phase = clock.mul(FLICKER_RATE).add(instanceIndex.toFloat().mul(FLICKER_PHASE_STEP))
  return sin(phase)
    .mul(FLICKER_DEPTH)
    .add(1 - FLICKER_DEPTH)
}

// A triangle wave along the beam, 0 at each segment's middle and 1 at the gaps between them,
// eased so the gaps have soft ends; the first segment is centred on the foot.
function beamSegments(up: Node<'float'>) {
  const wave = abs(fract(up.mul(BEAM_HEIGHT / SEGMENT_METRES).add(0.5)).sub(0.5)).mul(2)
  return mix(float(1), float(SEGMENT_GAP_GAIN), smoothstep(0.42, 0.55, wave))
}

interface SharedAttributes {
  centre: InstancedBufferAttribute
  stockout: InstancedBufferAttribute
  radius: InstancedBufferAttribute
}

/** Pack axes (x east, y north) to three.js (x, 0, -y). */
export function storeCentres(stores: Stores): Float32Array {
  const out = new Float32Array(stores.count * 3)
  for (let s = 0; s < stores.count; s++) {
    out[s * 3] = stores.x[s] ?? 0
    out[s * 3 + 2] = -(stores.y[s] ?? 0)
  }
  return out
}

// An open tube standing on the ground; caps would draw as bright discs.
export function beamShape(): BufferGeometry {
  const openEnded = true
  const tube = new CylinderGeometry(
    BEAM_RADIUS,
    BEAM_RADIUS,
    BEAM_HEIGHT,
    BEAM_SEGMENTS,
    1,
    openEnded,
  )
  return tube.translate(0, BEAM_HEIGHT / 2, 0)
}

// A unit circle at the ring's lift as line-segment pairs; the radius scales it per store.
export function unitRing(segments = RING_SEGMENTS): BufferGeometry {
  const positions = new Float32Array(segments * 6)
  for (let i = 0; i < segments; i++) {
    const a0 = (i / segments) * Math.PI * 2
    const a1 = ((i + 1) / segments) * Math.PI * 2
    positions.set(
      [Math.cos(a0), RING_LIFT, Math.sin(a0), Math.cos(a1), RING_LIFT, Math.sin(a1)],
      i * 6,
    )
  }
  return new BufferGeometry().setAttribute('position', new BufferAttribute(positions, 3))
}

// One instance per store over the shape's own vertices; all three draws share the store attributes.
function instanced(
  shape: BufferGeometry,
  shared: SharedAttributes,
  count: number,
): InstancedBufferGeometry {
  const geometry = new InstancedBufferGeometry()
  geometry.setIndex(shape.index)
  for (const [name, values] of Object.entries({ ...shape.attributes, ...shared })) {
    geometry.setAttribute(name, values)
  }
  geometry.instanceCount = count
  return geometry
}

function additive<M extends NodeMaterial>(material: M): M {
  material.transparent = true
  material.depthWrite = false
  material.blending = AdditiveBlending
  return material
}
