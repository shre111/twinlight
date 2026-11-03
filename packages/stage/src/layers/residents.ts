import { city as cityTokens } from '@twinlight/ui/tokens'
import {
  attribute,
  cameraNear,
  cameraPosition,
  clamp,
  color,
  exp,
  Fn,
  float,
  hash,
  instanceIndex,
  length,
  max,
  min,
  mix,
  modelViewMatrix,
  normalize,
  oneMinus,
  screenDPR,
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
// The prototype tuned the floor at DPR 1.75; lower ratios keep that physical size so the faint
// unaware points do not vanish on 1x screens.
const MIN_PHYSICAL_PX = MIN_PX * 1.75
// A fresh flare's point quad is this many cores wide to carry its halo; settled points draw
// only the core there and glow through the second quad below.
const FLARE_SPAN = 2.5
// The wall-clearing nudge never exceeds this, so far flares cannot jump in front of buildings.
const MAX_NUDGE_METRES = 60
// Bloom stand-in until P-045 (drop the glow quad and these then): every point also draws a
// Gaussian glow of fixed pixel width on a second quad just past the near plane, so no wall cuts
// it and hidden points still add haze. Its energy is GLOW_ENERGY times the settled core's at
// the prototype's 48k residents (the design renders) and falls with the drawn count, so a 100k
// city cannot blob.
const GLOW_SIGMA_PX = 2
const GLOW_QUAD_SIGMAS = 5
const GLOW_ENERGY = { unaware: 0.8, aware: 0.6, ordered: 0.35 }
const GLOW_REFERENCE_RESIDENTS = 48_000
// Core energy is about 0.8 r² for the disc shape below; a Gaussian's is 2π σ² times its peak.
const GLOW_PEAK_PER_ENERGY = 0.8 / (2 * Math.PI)

/**
 * Residents as instanced quads per resident: WebGPU draws point primitives at 1 px, so
 * a camera-facing sprite stands in. State is evaluated in the shader from `time` (D-004).
 */
export class ResidentsLayer {
  readonly object: Sprite
  readonly count: number
  readonly time = uniform(0)
  readonly #focal = uniform(1)
  readonly #geometry: BufferGeometry
  readonly #material: PointsNodeMaterial

  /** `glow` follows the tier's bloom: the low tier has none, so it skips the glow quads' fill. */
  constructor(residents: Residents, events: Events, cap: number, fog: Fog, glow = true) {
    const picks = residentSubset(residents.count, cap)
    this.count = picks.length
    this.#geometry = pointAndGlowQuads(glow)
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
    this.#material = this.#createMaterial(fog, glowDensity(this.count))
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

  #createMaterial(fog: Fog, density: number): PointsNodeMaterial {
    const material = new PointsNodeMaterial({
      blending: AdditiveBlending,
      depthWrite: false,
      transparent: true,
    })
    material.sizeAttenuation = false
    material.alphaToCoverage = false

    const home = attribute('home', 'vec3')
    const isGlow = attribute('glow', 'float')
    const seed = hash(instanceIndex)
    const sinceAware = this.time.sub(attribute('awareAt', 'float'))
    const sinceOrder = this.time.sub(attribute('orderAt', 'float'))
    const aware = clamp(sinceAware.div(AWARE_RAMP_MINUTES), 0, 1)
    const ordered = clamp(sinceOrder.div(ORDER_RAMP_MINUTES), 0, 1)
    const flare = ordered.mul(oneMinus(smoothstep(0, FLARE_MINUTES, sinceOrder)))

    const unawareTint = color(palette.tube).mul(cityTokens.residentUnawareIntensity)
    const heardTint = color(palette.chatter).mul(cityTokens.residentAwareIntensity)
    const orderedTint = color(palette.lamp).mul(cityTokens.residentOrderedIntensity)
    const tint = mix(mix(unawareTint, heardTint, aware), orderedTint, ordered)
      .mul(seed.mul(0.3).add(0.85))
      .mul(flare.mul(1.2).add(1))
    const settledGrowth = mix(mix(1, 1.2, aware), 2.2, ordered)
    const growth = settledGrowth.mul(flare.mul(1.2).add(1))

    // Perspective size in logical pixels; below the floor the point dims instead of shrinking,
    // so far districts fade rather than turn to noise.
    const depth = max(modelViewMatrix.mul(vec4(home, 1)).z.negate(), 1)
    // Floored so a zero-size viewport cannot divide by zero below.
    const pixelsPerMetre = max(
      this.#focal.mul(viewportSize.y).div(screenDPR).mul(0.5).div(depth),
      1e-4,
    )
    const spread = seed.mul(0.36).add(0.82).mul(POINT_METRES).mul(pixelsPerMetre)
    const px = growth.mul(spread)
    const minPx = max(MIN_PX, float(MIN_PHYSICAL_PX).div(max(screenDPR, 0.5)))
    const dim = clamp(px.div(minPx), 0.25, 1)
    const core = clamp(px, minPx, growth.mul(MAX_PX))
    const flareSpan = flare.mul(FLARE_SPAN - 1).add(1)
    const pointSize = core.mul(flareSpan)
    material.sizeNode = mix(pointSize, GLOW_SIGMA_PX * GLOW_QUAD_SIGMAS, isGlow)

    // The point is depth-tested at its own radius in front of the resident, like the near side
    // of a glowing sphere, so the wall it sits on cannot cut it while buildings in front still hide it.
    const radius = min(pointSize.div(pixelsPerMetre).mul(0.5), MAX_NUDGE_METRES)
    const towardCamera = normalize(cameraPosition.sub(home).add(vec3(0, 1e-3, 0)))
    const pointPosition = home.add(towardCamera.mul(radius))
    // The glow sits on the same view ray just past the near plane: same screen spot, in front of
    // every wall. Near clips on view depth, not ray length, so the step grows by 1 / cos off axis.
    const offAxisCos = max(depth.div(max(length(cameraPosition.sub(home)), 1)), 0.1)
    const glowPosition = cameraPosition.sub(towardCamera.mul(cameraNear.mul(1.5).div(offAxisCos)))
    material.positionNode = mix(pointPosition, glowPosition, isGlow)

    const shade = vertexStage(tint.mul(dim).mul(oneMinus(fog.factor(depth))))
    // Across the quad, d runs in core radii for the point and in sigmas for the glow.
    const scale = vertexStage(mix(flareSpan.mul(2), GLOW_QUAD_SIGMAS, isGlow))
    const span = vertexStage(flareSpan)
    const layer = vertexStage(isGlow)
    const flareGain = vertexStage(flare.mul(0.45 * density))
    const settledRadius = clamp(settledGrowth.mul(spread), minPx, settledGrowth.mul(MAX_PX)).mul(
      0.5,
    )
    const glowEnergy = mix(
      mix(GLOW_ENERGY.unaware, GLOW_ENERGY.aware, aware),
      GLOW_ENERGY.ordered,
      ordered,
    )
    const glowPeak = vertexStage(
      settledRadius
        .mul(settledRadius)
        .mul(glowEnergy)
        .mul((GLOW_PEAK_PER_ENERGY * density) / (GLOW_SIGMA_PX * GLOW_SIGMA_PX)),
    )
    material.colorNode = Fn(() => {
      const d = length(uv().sub(0.5)).mul(scale)
      const disc = oneMinus(smoothstep(0, 1, d))
      const window = oneMinus(smoothstep(span.mul(0.7), span, d))
      const flareHalo = exp(d.mul(d).mul(-0.5)).mul(flareGain).mul(window)
      const point = disc.mul(disc).mul(0.85).add(disc.mul(0.35)).add(flareHalo)
      const glow = exp(d.mul(d).mul(-0.5))
        .mul(oneMinus(smoothstep(GLOW_QUAD_SIGMAS / 2 - 0.8, GLOW_QUAD_SIGMAS / 2, d)))
        .mul(glowPeak)
      return shade.mul(mix(point, glow, layer))
    })()
    return material
  }
}

/** Glow gain relative to the prototype's 48k renders, so denser cities add haze, not blobs. */
export function glowDensity(count: number): number {
  return Math.min(1, GLOW_REFERENCE_RESIDENTS / Math.max(1, count))
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

// Per instance: the depth-tested point (glow 0) and, unless disabled, its always-visible glow (glow 1).
export function pointAndGlowQuads(withGlow: boolean): BufferGeometry {
  const quads = withGlow ? 2 : 1
  const repeat = (values: number[]) => Float32Array.from(Array(quads).fill(values).flat())
  const geometry = new BufferGeometry()
  geometry.setAttribute(
    'position',
    new BufferAttribute(repeat([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]), 3),
  )
  geometry.setAttribute('uv', new BufferAttribute(repeat([0, 0, 1, 0, 1, 1, 0, 1]), 2))
  geometry.setAttribute(
    'glow',
    new BufferAttribute(
      Float32Array.from({ length: quads * 4 }, (_, v) => (v < 4 ? 0 : 1)),
      1,
    ),
  )
  geometry.setIndex([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7].slice(0, quads * 6))
  return geometry
}
