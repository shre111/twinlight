import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { type PerspectiveCamera, Vector3 } from 'three/webgpu'

export interface Pose {
  target: readonly [number, number, number]
  distance: number
  polar: number
  azimuth: number
}

export const POSES = {
  home: { target: [-560, 0, -260], distance: 6200, polar: 1.3, azimuth: -0.5 },
  hero: { target: [-600, 0, -300], distance: 5900, polar: 1.33, azimuth: -0.48 },
  phone: { target: [-800, 0, -300], distance: 6400, polar: 1.02, azimuth: -0.42 },
} as const satisfies Record<string, Pose>

const PHONE_MAX_WIDTH = 700
const FOV = { desktop: 40, phone: 46 }
const TARGET_BOUNDS = { minX: -4200, maxX: 3200, minZ: -3800, maxZ: 3800 }

export class CameraRig {
  readonly #camera: PerspectiveCamera
  readonly #controls: OrbitControls
  readonly #shift = new Vector3()

  constructor(camera: PerspectiveCamera, canvas: HTMLCanvasElement, reducedMotion: boolean) {
    this.#camera = camera
    const controls = new OrbitControls(camera, canvas)
    controls.enableDamping = !reducedMotion
    controls.dampingFactor = 0.07
    controls.minDistance = 700
    controls.maxDistance = 16_000
    controls.minPolarAngle = 0.12
    controls.maxPolarAngle = 1.38
    controls.screenSpacePanning = false
    controls.zoomSpeed = 0.8
    this.#controls = controls
    // The window, not the canvas, which may not be laid out yet.
    this.pose(window.innerWidth < PHONE_MAX_WIDTH ? POSES.phone : POSES.home)
  }

  resize(width: number, height: number): void {
    this.#camera.fov = width < PHONE_MAX_WIDTH ? FOV.phone : FOV.desktop
    this.#camera.aspect = width / height
    this.#camera.updateProjectionMatrix()
  }

  pose({ target, distance, polar, azimuth }: Pose): void {
    const [x, y, z] = target
    this.#controls.target.set(x, y, z)
    this.#camera.position.set(
      x + distance * Math.sin(polar) * Math.sin(azimuth),
      y + distance * Math.cos(polar),
      z + distance * Math.sin(polar) * Math.cos(azimuth),
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
    const x = Math.min(TARGET_BOUNDS.maxX, Math.max(TARGET_BOUNDS.minX, target.x))
    const z = Math.min(TARGET_BOUNDS.maxZ, Math.max(TARGET_BOUNDS.minZ, target.z))
    if (x === target.x && z === target.z && target.y === 0) return
    this.#shift.set(x - target.x, -target.y, z - target.z)
    target.add(this.#shift)
    this.#camera.position.add(this.#shift)
  }
}
