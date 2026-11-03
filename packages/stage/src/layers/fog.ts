import { exp, Fn, max, mix, oneMinus, positionView, smoothstep, uniform } from 'three/tsl'
import type { Node } from 'three/webgpu'
import { palette } from '../palette'

export const viewDepth = positionView.z.negate()

export class Fog {
  readonly color = uniform(palette.fog)
  readonly haze = uniform(palette.horizon)
  readonly start = uniform(4000)
  readonly density = uniform(0.0001)

  // Clear up to `start`, then a Gaussian ramp. Both scale with the camera distance so the
  // city keeps the same depth at any zoom.
  readonly factor = Fn(([depth]: [Node<'float'>]) => {
    const f = max(0, depth.sub(this.start)).mul(this.density)
    return oneMinus(exp(f.mul(f).negate()))
  })

  // Far fog drifts from dusk to the horizon haze so the skyline melts into the sky.
  readonly mix = Fn(([colour, factor]: [Node<'vec3'>, Node<'float'>]) =>
    mix(colour, mix(this.color, this.haze, smoothstep(0.3, 1, factor)), factor),
  )

  update(cameraDistance: number): void {
    this.start.value = cameraDistance * 0.62
    this.density.value = 1.15 / cameraDistance
  }
}
