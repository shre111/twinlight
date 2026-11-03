import { city as cityTokens } from '@twinlight/ui/tokens'
import {
  attribute,
  cameraPosition,
  clamp,
  color,
  exp,
  Fn,
  hash,
  instanceIndex,
  length,
  max,
  mix,
  modelViewMatrix,
  normalize,
  oneMinus,
  screenDPR,
  smoothstep,
  uniform,
  uv,
  vec4,
  vertexStage,
  viewportSize,
} from 'three/tsl'
import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  InstancedBufferAttribute,
  type PerspectiveCamera,
  PointsNodeMaterial,
  Sprite,
} from 'three/webgpu'
import { NEVER, type StageCity, type StageTape } from '../inputs'
import { palette } from '../palette'
import type { Fog } from './fog'

type Residents = StageCity['residents']
type Events = StageTape['residents']

const AWARE_RAMP_MINUTES = 36
const ORDER_RAMP_MINUTES = 15
const FLARE_MINUTES = cityTokens.flareSimHours * 60
// A point is this wide in metres, then held between the two pixel sizes (logical px, scaled by DPR).
const POINT_METRES = 9.5
const MIN_PX = 1.3
const MAX_PX = 7.5
// A fresh flare's quad is this many cores wide to carry its soft glow (a stand-in for bloom
// until P-045); steady points draw only the core.
const FLARE_SPAN = 2

/**
 * Residents as one instanced quad per resident: WebGPU draws point primitives at 1 px, so
 * a camera-facing sprite stands in. State is evaluated in the shader from `time` (D-004).
 */
export class ResidentsLayer {
  readonly object: Sprite
  readonly count: number
  readonly time = uniform(0)
  readonly #focal = uniform(1)
  readonly #geometry: BufferGeometry
  readonly #material: PointsNodeMaterial

  constructor(residents: Residents, events: Events, cap: number, fog: Fog) {
    const picks = residentSubset(residents.count, cap)
    this.count = picks.length
    this.#geometry = quad()
    this.#geometry.setAttribute(
      'home',
      new InstancedBufferAttribute(residentPositions(residents, picks), 3),
    )
    this.#geometry.setAttribute(
      'awareAt',
      new InstancedBufferAttribute(eventMinutes(events.awareAt, picks), 1),
    )
    this.#geometry.setAttribute(
      'orderAt',
      new InstancedBufferAttribute(eventMinutes(events.firstOrderAt, picks), 1),
    )
    this.#material = this.#createMaterial(fog)
    const sprite = new Sprite(this.#material)
    sprite.geometry = this.#geometry
    sprite.count = this.count
    sprite.frustumCulled = false
    this.object = sprite
  }

  update(camera: PerspectiveCamera): void {
    this.#focal.value = camera.projectionMatrix.elements[5] ?? 1
  }

  dispose(): void {
    this.#geometry.dispose()
    this.#material.dispose()
    this.object.removeFromParent()
  }

  #createMaterial(fog: Fog): PointsNodeMaterial {
    const material = new PointsNodeMaterial({
      blending: AdditiveBlending,
      depthWrite: false,
      transparent: true,
    })
    material.sizeAttenuation = false
    material.alphaToCoverage = false

    const home = attribute('home', 'vec3')
    const seed = hash(instanceIndex)
    const sinceAware = this.time.sub(attribute('awareAt', 'float'))
    const sinceOrder = this.time.sub(attribute('orderAt', 'float'))
    const aware = clamp(sinceAware.div(AWARE_RAMP_MINUTES), 0, 1)
    const ordered = clamp(sinceOrder.div(ORDER_RAMP_MINUTES), 0, 1)
    const flare = ordered.mul(oneMinus(smoothstep(0, FLARE_MINUTES, sinceOrder)))

    const unaware = color(palette.tube).mul(cityTokens.residentUnawareIntensity)
    const heard = color(palette.chatter).mul(cityTokens.residentAwareIntensity)
    const lit = color(palette.lamp).mul(cityTokens.residentOrderedIntensity)
    const tint = mix(mix(unaware, heard, aware), lit, ordered)
      .mul(seed.mul(0.3).add(0.85))
      .mul(flare.mul(1.2).add(1))
    const growth = mix(mix(1, 1.2, aware), 2.2, ordered).mul(flare.mul(1.6).add(1))

    // Perspective size in logical pixels; below MIN_PX the point dims instead of shrinking,
    // so far districts fade rather than turn to noise.
    const depth = max(modelViewMatrix.mul(vec4(home, 1)).z.negate(), 1)
    const pixelsPerMetre = this.#focal.mul(viewportSize.y).div(screenDPR).mul(0.5).div(depth)
    const px = growth.mul(seed.mul(0.36).add(0.82)).mul(POINT_METRES).mul(pixelsPerMetre)
    const dim = clamp(px.div(MIN_PX), 0.25, 1)
    const span = flare.mul(FLARE_SPAN - 1).add(1)
    const size = clamp(px, MIN_PX, growth.mul(MAX_PX)).mul(span)
    material.sizeNode = size

    // The quad is depth-tested at its own radius in front of the resident, like the near side
    // of a glowing sphere, so the wall it sits on cannot cut it while buildings in front still hide it.
    const radius = size.div(pixelsPerMetre).mul(0.5)
    material.positionNode = home.add(normalize(cameraPosition.sub(home)).mul(radius))

    const glow = vertexStage(tint.mul(dim).mul(oneMinus(fog.factor(depth))))
    const haloSpan = vertexStage(span)
    const haloGain = vertexStage(flare.mul(0.45))
    material.colorNode = Fn(() => {
      // d is 0 at the centre, 1 at the core's edge and haloSpan at the quad's edge.
      const d = length(uv().sub(0.5)).mul(2).mul(haloSpan)
      const core = oneMinus(smoothstep(0, 1, d))
      const window = oneMinus(smoothstep(haloSpan.mul(0.7), haloSpan, d))
      const halo = exp(d.mul(d).mul(-0.9)).mul(haloGain).mul(window)
      return glow.mul(core.mul(core).mul(0.85).add(core.mul(0.35)).add(halo))
    })()
    return material
  }
}

// Every (count / cap)-th resident, so each district keeps its share of the drawn points.
export function residentSubset(count: number, cap: number): Uint32Array {
  const drawn = Math.max(0, Math.min(count, Math.floor(cap)))
  const picks = new Uint32Array(drawn)
  for (let j = 0; j < drawn; j++) picks[j] = Math.floor((j * count) / drawn)
  return picks
}

/** Pack axes (x east, y north, z up) to three.js (x, z, -y). */
export function residentPositions(residents: Residents, picks: Uint32Array): Float32Array {
  const out = new Float32Array(picks.length * 3)
  picks.forEach((i, j) => {
    out[j * 3] = residents.x[i] ?? 0
    out[j * 3 + 1] = residents.z[i] ?? 0
    out[j * 3 + 2] = -(residents.y[i] ?? 0)
  })
  return out
}

// NEVER (2^32 - 1) rounds to 2^32 as a float, far past any run, so `time - at` stays negative.
export function eventMinutes(times: Uint32Array, picks: Uint32Array): Float32Array {
  return Float32Array.from(picks, (i) => times[i] ?? NEVER)
}

function quad(): BufferGeometry {
  const geometry = new BufferGeometry()
  geometry.setAttribute(
    'position',
    new BufferAttribute(Float32Array.of(-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0), 3),
  )
  geometry.setAttribute('uv', new BufferAttribute(Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1), 2))
  geometry.setIndex([0, 1, 2, 0, 2, 3])
  return geometry
}
