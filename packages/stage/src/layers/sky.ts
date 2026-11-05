import { color, exp, Fn, mix, normalize, positionLocal, smoothstep, varying } from 'three/tsl'
import { BackSide, Mesh, MeshBasicNodeMaterial, SphereGeometry } from 'three/webgpu'
import { palette } from '../palette'
import type { Fog } from './fog'
import type { Layer } from './layer'

// Inside the far plane at any zoom (both scale with the city); the sphere follows the camera,
// so its radius never shows.
const TOY_RADIUS = 40_000

export function createSky(fog: Fog, scale: number): Layer {
  const material = new MeshBasicNodeMaterial({
    side: BackSide,
    depthWrite: false,
    depthTest: false,
    fog: false,
  })
  material.colorNode = Fn(() => {
    const elevation = normalize(varying(positionLocal)).y
    const overhead = mix(
      color(palette.skyMid),
      color(palette.skyTop),
      smoothstep(0.06, 0.45, elevation),
    )
    const sky = mix(fog.haze, overhead, smoothstep(0, 0.12, elevation))
    // A thin violet band just above the horizon: a Gaussian 0.014 wide, never negative.
    const band = elevation.sub(0.006).div(0.014)
    const banded = sky.add(
      color(palette.horizonBand)
        .mul(exp(band.mul(band).negate()))
        .mul(0.3),
    )
    return mix(fog.haze, banded, smoothstep(-0.015, 0, elevation))
  })()
  const mesh = new Mesh(new SphereGeometry(TOY_RADIUS * scale, 48, 24), material)
  mesh.renderOrder = -10
  mesh.frustumCulled = false
  return {
    object: mesh,
    dispose: () => {
      mesh.geometry.dispose()
      material.dispose()
    },
  }
}
