import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { type PerspectiveCamera, Vector3 } from 'three/webgpu'
import { type CityExtent, fitPoint, TOY_EXTENT } from './extent'

export interface Pose {
  target: readonly [number, number, number]
  distance: number
  polar: number
  azimuth: number
}

/** Tuned on the toy city; `fit` scales them onto the loaded city's extent. */
export const POSES = {
  home: { target: [-560, 0, -260], distance: 6200, polar: 1.3, azimuth: -0.5 },
  hero: { target: [-600, 0, -300], distance: 5900, polar: 1.33, azimuth: -0.48 },
  phone: { target: [-800, 0, -300], distance: 6400, polar: 1.02, azimuth: -0.42 },
} as const satisfies Record<string, Pose>

const PHONE_MAX_WIDTH = 700
const FOV = { desktop: 40, phone: 46 }
const TOY_TARGET_BOUNDS = { minX: -4200, maxX: 3200, minZ: -3800, maxZ: 3800 }
const TOY_DISTANCE = { min: 700, max: 16_000 }
const TOY_CLIP = { near: 20, far: 90_000 }

export class CameraRig {
  readonly #camera: PerspectiveCamera
  readonly #controls: OrbitControls
  readonly #shift = new Vector3()
  #extent: CityExtent = TOY_EXTENT
  #targetBounds = { ...TOY_TARGET_BOUNDS }

  constructor(camera: PerspectiveCamera, canvas: HTMLCanvasElement, reducedMotion: boolean) {
    this.#camera = camera
    const controls = new OrbitControls(camera, canvas)
    controls.enableDamping = !reducedMotion
    controls.dampingFactor = 0.07
    controls.minPolarAngle = 0.12
    controls.maxPolarAngle = 1.38
    controls.screenSpacePanning = false
    controls.zoomSpeed = 0.8
    this.#controls = controls
    this.fit(TOY_EXTENT)
  }

  /** Scales the toy-tuned poses, zoom range, target clamp and clip planes onto a city, then frames it. */
  fit(extent: CityExtent): void {
    this.#extent = extent
    const { scale } = extent
    const [minX, minZ] = fitPoint(TOY_TARGET_BOUNDS.minX, TOY_TARGET_BOUNDS.minZ, extent)
    const [maxX, maxZ] = fitPoint(TOY_TARGET_BOUNDS.maxX, TOY_TARGET_BOUNDS.maxZ, extent)
    this.#targetBounds = { minX, maxX, minZ, maxZ }
    this.#controls.minDistance = TOY_DISTANCE.min * scale
    this.#controls.maxDistance = TOY_DISTANCE.max * scale
    this.#camera.near = TOY_CLIP.near * scale
    this.#camera.far = TOY_CLIP.far * scale
    this.#camera.updateProjectionMatrix()
    // The window, not the canvas, which may not be laid out yet.
    this.pose(window.innerWidth < PHONE_MAX_WIDTH ? POSES.phone : POSES.home)
  }

  resize(width: number, height: number): void {
    this.#camera.fov = width < PHONE_MAX_WIDTH ? FOV.phone : FOV.desktop
    this.#camera.aspect = width / height
    this.#camera.updateProjectionMatrix()
  }

  /** Places the camera at a toy-tuned pose, scaled onto the current city. */
  pose({ target, distance, polar, azimuth }: Pose): void {
    const [x, z] = fitPoint(target[0], target[2], this.#extent)
    const d = distance * this.#extent.scale
    this.#controls.target.set(x, target[1], z)
    this.#camera.position.set(
      x + d * Math.sin(polar) * Math.sin(azimuth),
      target[1] + d * Math.cos(polar),
      z + d * Math.sin(polar) * Math.cos(azimuth),
    )
    this.#camera.lookAt(this.#controls.target)
    this.#controls.update()
  }

  /** Advances damping and keeps the target over the city. Returns the camera's distance to it. */
  update(): number {
    this.#controls.update()
    this.#clampTarget()
    return this.#camera.position.distanceTo(this.#controls.target)
  }

  dispose(): void {
    this.#controls.dispose()
  }

  #clampTarget(): void {
    const target = this.#controls.target
    const bounds = this.#targetBounds
    const x = Math.min(bounds.maxX, Math.max(bounds.minX, target.x))
    const z = Math.min(bounds.maxZ, Math.max(bounds.minZ, target.z))
    if (x === target.x && z === target.z && target.y === 0) return
    this.#shift.set(x - target.x, -target.y, z - target.z)
    target.add(this.#shift)
    this.#camera.position.add(this.#shift)
  }
}
