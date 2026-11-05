import { color, exp, Fn, length, positionWorld, vec2 } from 'three/tsl'
import { Mesh, MeshBasicNodeMaterial, type Node, PlaneGeometry } from 'three/webgpu'
import { type CityExtent, fitPoint } from '../extent'
import { cityGlow, palette } from '../palette'
import { type Fog, viewDepth } from './fog'
import type { Layer } from './layer'

// Toy-city values, scaled onto each city's extent.
const TOY_PLANE_SIZE = 60_000
// The glow sits under the toy city's centre of mass, which is west of its pack origin.
const TOY_GLOW_CENTRE = [-250, 0] as const
const TOY_GLOW_RADIUS = 3600

export function createGround(fog: Fog, warm: Node<'float'>, extent: CityExtent): Layer {
  const [glowX, glowZ] = fitPoint(TOY_GLOW_CENTRE[0], TOY_GLOW_CENTRE[1], extent)
  const glowCentre = vec2(glowX, glowZ)
  const glowRadius = TOY_GLOW_RADIUS * extent.scale
  const size = TOY_PLANE_SIZE * Math.max(1, extent.scale)
  const material = new MeshBasicNodeMaterial({ depthWrite: false })
  material.colorNode = Fn(() => {
    const offset = positionWorld.xz.sub(glowCentre).mul(vec2(1, 1.08))
    const r = length(offset).div(glowRadius)
    const glow = exp(r.mul(r).negate())
    const base = color(palette.ground).mul(glow.mul(0.3).add(0.78))
    const lit = base.add(cityGlow(warm).mul(glow).mul(0.06))
    return fog.mix(lit, fog.factor(viewDepth))
  })()
  const mesh = new Mesh(new PlaneGeometry(size, size).rotateX(-Math.PI / 2), material)
  mesh.position.set(extent.centreX, 0, extent.centreZ)
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
