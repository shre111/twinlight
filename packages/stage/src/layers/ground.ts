import { color, exp, Fn, length, positionWorld, vec2 } from 'three/tsl'
import { Mesh, MeshBasicNodeMaterial, type Node, PlaneGeometry } from 'three/webgpu'
import { cityGlow, palette } from '../palette'
import { type Fog, viewDepth } from './fog'
import type { Layer } from './layer'

const EXTENT = 60_000
// The glow sits under the city's centre of mass, which is west of the pack origin.
const GLOW_CENTRE = vec2(-250, 0)
const GLOW_RADIUS = 3600

export function createGround(fog: Fog, warm: Node<'float'>): Layer {
  const material = new MeshBasicNodeMaterial({ depthWrite: false })
  material.colorNode = Fn(() => {
    const offset = positionWorld.xz.sub(GLOW_CENTRE).mul(vec2(1, 1.08))
    const r = length(offset).div(GLOW_RADIUS)
    const glow = exp(r.mul(r).negate())
    const base = color(palette.ground).mul(glow.mul(0.3).add(0.78))
    const lit = base.add(cityGlow(warm).mul(glow).mul(0.06))
    return fog.mix(lit, fog.factor(viewDepth))
  })()
  const mesh = new Mesh(new PlaneGeometry(EXTENT, EXTENT).rotateX(-Math.PI / 2), material)
  mesh.renderOrder = -5
  mesh.frustumCulled = false
  return {
    object: mesh,
    dispose: () => {
      mesh.geometry.dispose()
      material.dispose()
    },
  }
}
