import { PerspectiveCamera, Scene, type WebGPURenderer } from 'three/webgpu'
import { FrameStats, type FrameSummary } from './frame-stats'
import { PerfHud } from './perf-hud'
import { type Backend, type BackendPreference, createRenderer } from './renderer'
import { initialTier, TIERS, type TierName, type TierOption } from './tiers'
import { type Viewport, type ViewportSize, watchViewport } from './viewport'

export interface StageOptions {
  tier?: TierOption
  backend?: BackendPreference
  hud?: boolean
}

export interface FrameReport extends FrameSummary {
  backend: Backend
  tier: TierName
  pixelRatio: number
  drawCalls: number
}

const HUD_INTERVAL_MS = 250

export class Stage {
  readonly backend: Backend
  #renderer: WebGPURenderer
  #scene = new Scene()
  #camera = new PerspectiveCamera(40, 1, 20, 90_000)
  #tier: TierName
  #pendingSize: ViewportSize | null = null
  #viewport: Viewport
  #disposed = false
  #stats = new FrameStats()
  #hud: PerfHud | null
  #lastFrameAt = 0
  #lastHudAt = 0
  #drawCalls = 0

  static async create(canvas: HTMLCanvasElement, options: StageOptions = {}): Promise<Stage> {
    const { renderer, backend } = await createRenderer(canvas, options.backend ?? 'auto')
    const tier =
      options.tier === undefined || options.tier === 'auto'
        ? initialTier(window.matchMedia('(pointer: coarse)').matches)
        : options.tier
    return new Stage(canvas, renderer, backend, tier, options.hud ?? false)
  }

  private constructor(
    canvas: HTMLCanvasElement,
    renderer: WebGPURenderer,
    backend: Backend,
    tier: TierName,
    hud: boolean,
  ) {
    this.#renderer = renderer
    this.backend = backend
    this.#tier = tier
    this.#hud = hud && canvas.parentElement ? new PerfHud(canvas.parentElement) : null
    this.#viewport = watchViewport(
      canvas,
      () => TIERS[this.#tier].maxPixelRatio,
      (size) => {
        this.#pendingSize = size
      },
    )
    document.addEventListener('visibilitychange', this.#restartClock)
    void renderer.setAnimationLoop((now) => this.#frame(now))
  }

  get tier(): TierName {
    return this.#tier
  }

  setTier(tier: TierName): void {
    this.#tier = tier
    this.#viewport.remeasure()
  }

  perf(): FrameReport {
    return {
      ...this.#stats.summary(),
      backend: this.backend,
      tier: this.#tier,
      pixelRatio: this.#renderer.getPixelRatio(),
      drawCalls: this.#drawCalls,
    }
  }

  dispose(): void {
    if (this.#disposed) return
    this.#disposed = true
    void this.#renderer.setAnimationLoop(null)
    document.removeEventListener('visibilitychange', this.#restartClock)
    this.#viewport.stop()
    this.#hud?.dispose()
    void this.#renderer.dispose()
  }

  #frame(now: number): void {
    // Resizing clears the canvas, so it waits for the start of a frame instead of the observer callback.
    if (this.#pendingSize) {
      this.#applySize(this.#pendingSize)
      this.#pendingSize = null
    }
    if (this.#lastFrameAt > 0) this.#stats.push(now - this.#lastFrameAt)
    this.#lastFrameAt = now

    this.#renderer.render(this.#scene, this.#camera)
    this.#drawCalls = this.#renderer.info.render.drawCalls

    if (this.#hud && now - this.#lastHudAt >= HUD_INTERVAL_MS) {
      this.#lastHudAt = now
      this.#hud.update(this.perf())
    }
  }

  #applySize({ width, height, pixelRatio }: ViewportSize): void {
    this.#renderer.setPixelRatio(pixelRatio)
    this.#renderer.setSize(width, height, false)
    this.#camera.aspect = width / height
    this.#camera.updateProjectionMatrix()
  }

  // A hidden tab pauses the loop; the gap is not a frame.
  #restartClock = () => {
    this.#lastFrameAt = 0
  }
}
