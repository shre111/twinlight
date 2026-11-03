import { oneMinus, vertexStage } from 'three/tsl'
import { BufferAttribute, BufferGeometry, LineBasicNodeMaterial, LineSegments } from 'three/webgpu'
import { ROAD_KIND, type StageCity } from '../inputs'
import { palette } from '../palette'
import { type Fog, viewDepth } from './fog'
import type { Layer } from './layer'

// Lift keeps roads off the ground, arterials above streets and bridges above both.
const ROAD_STYLE = [
  { kind: ROAD_KIND.street, colour: palette.street, opacity: 0.32, lift: 0.6 },
  { kind: ROAD_KIND.arterial, colour: palette.arterial, opacity: 0.42, lift: 0.9 },
  { kind: ROAD_KIND.bridge, colour: palette.bridge, opacity: 0.6, lift: 2 },
] as const

export function createRoads(roads: StageCity['roads'], fog: Fog): Layer[] {
  return ROAD_STYLE.flatMap((style) => {
    const positions = segmentPositions(roads, style.kind, style.lift)
    if (positions.length === 0) return []
    const geometry = new BufferGeometry().setAttribute(
      'position',
      new BufferAttribute(positions, 3),
    )
    const material = new LineBasicNodeMaterial({
      color: style.colour,
      transparent: true,
      depthWrite: false,
    })
    material.opacityNode = oneMinus(vertexStage(fog.factor(viewDepth))).mul(style.opacity)
    const lines = new LineSegments(geometry, material)
    lines.renderOrder = -3
    lines.frustumCulled = false
    return [
      {
        object: lines,
        dispose: () => {
          geometry.dispose()
          material.dispose()
        },
      },
    ]
  })
}

function segmentPositions(roads: StageCity['roads'], kind: number, lift: number): Float32Array {
  let count = 0
  for (const k of roads.kind) if (k === kind) count++
  const positions = new Float32Array(count * 6)
  let at = 0
  roads.kind.forEach((k, segment) => {
    if (k !== kind) return
    const s = segment * 4
    positions[at] = roads.segments[s] ?? 0
    positions[at + 1] = lift
    positions[at + 2] = -(roads.segments[s + 1] ?? 0)
    positions[at + 3] = roads.segments[s + 2] ?? 0
    positions[at + 4] = lift
    positions[at + 5] = -(roads.segments[s + 3] ?? 0)
    at += 6
  })
  return positions
}
