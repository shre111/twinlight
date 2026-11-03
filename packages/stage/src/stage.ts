import { PerspectiveCamera, Scene, type WebGPURenderer } from 'three/webgpu'
import { type Backend, type BackendPreference, createRenderer } from './renderer'
import { initialTier, TIERS, type TierName, type TierOption } from './tiers'
import { type ViewportSize, watchViewport } from './viewport'

export interface StageOptions {
  tier?: TierOption
  backend?: BackendPreference
}

export class Stage {
  readonly backend: Backend
  #renderer: WebGPURenderer
  #scene = new Scene()
  #camera = new PerspectiveCamera(40, 1, 20, 90_000)
  #tier: TierName
  #pendingSize: ViewportSize | null = null
  #viewport: ReturnType<typeof watchViewport>

  static async create(canvas: HTMLCanvasElement, options: StageOptions = {}): Promise<Stage> {
    const { renderer, backend } = await createRenderer(canvas, options.backend ?? 'auto')
    const tier =
      options.tier === undefined || options.tier === 'auto'
        ? initialTier(window.matchMedia('(pointer: coarse)').matches)
        : options.tier
    return new Stage(canvas, renderer, backend, tier)
  }

  private constructor(
    canvas: HTMLCanvasElement,
    renderer: WebGPURenderer,
    backend: Backend,
    tier: TierName,
  ) {
    this.#renderer = renderer
    this.backend = backend
    this.#tier = tier
    this.#viewport = watchViewport(
      canvas,
      () => TIERS[this.#tier].maxPixelRatio,
      (size) => {
        this.#pendingSize = size
      },
    )
    void renderer.setAnimationLoop((now) => this.#frame(now))
  }

  get tier(): TierName {
    return this.#tier
  }

  setTier(tier: TierName): void {
    this.#tier = tier
    this.#viewport.remeasure()
  }

  dispose(): void {
    void this.#renderer.setAnimationLoop(null)
    this.#viewport.stop()
    void this.#renderer.dispose()
  }

  #frame(_now: number): void {
    // Resizing clears the canvas, so it waits for the start of a frame instead of the observer callback.
    if (this.#pendingSize) {
      this.#applySize(this.#pendingSize)
      this.#pendingSize = null
    }
    this.#renderer.render(this.#scene, this.#camera)
  }

  #applySize({ width, height, pixelRatio }: ViewportSize): void {
    this.#renderer.setPixelRatio(pixelRatio)
    this.#renderer.setSize(width, height, false)
    this.#camera.aspect = width / height
    this.#camera.updateProjectionMatrix()
  }
}
