import { PerspectiveCamera, Scene, type WebGPURenderer } from 'three/webgpu'
import { CameraRig } from './camera'
import { FrameStats, type FrameSummary } from './frame-stats'
import type { StageCity } from './inputs'
import { CityLayer } from './layers/city'
import { palette } from './palette'
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
// Longer gaps come from a paused or throttled loop, not from rendering.
const MAX_FRAME_GAP_MS = 1000

export class Stage {
  readonly backend: Backend
  #renderer: WebGPURenderer
  #scene = new Scene()
  #camera = new PerspectiveCamera(40, 1, 20, 90_000)
  #rig: CameraRig
  #city: CityLayer | null = null
  #tier: TierName
  #pendingSize: ViewportSize | null = null
  #viewport: Viewport
  #disposed = false
  #loads = 0
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
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    return new Stage(canvas, renderer, backend, tier, options.hud ?? false, reducedMotion)
  }

  private constructor(
    canvas: HTMLCanvasElement,
    renderer: WebGPURenderer,
    backend: Backend,
    tier: TierName,
    hud: boolean,
    reducedMotion: boolean,
  ) {
    this.#renderer = renderer
    this.backend = backend
    this.#tier = tier
    this.#rig = new CameraRig(this.#camera, canvas, reducedMotion)
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
    this.#stats.reset()
    this.#viewport.remeasure()
  }

  async loadCity(city: StageCity): Promise<void> {
    if (this.#disposed) return
    const load = ++this.#loads
    this.#city?.dispose()
    const layer = new CityLayer(city)
    this.#city = layer
    this.#scene.add(layer.group)
    this.#renderer.setClearColor(palette.skyTop, 1)
    try {
      await this.#renderer.compileAsync(this.#scene, this.#camera)
    } catch (error) {
      if (!this.#disposed) throw error
    }
    // A newer load or a dispose may have landed while the materials compiled.
    if (this.#disposed || load !== this.#loads) return
    this.#stats.reset()
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
    this.#city?.dispose()
    this.#rig.dispose()
    void this.#renderer.dispose()
  }

  #frame(now: number): void {
    // Resizing clears the canvas, so it waits for the start of a frame instead of the observer callback.
    if (this.#pendingSize) {
      this.#applySize(this.#pendingSize)
      this.#pendingSize = null
    }
    const gap = now - this.#lastFrameAt
    if (this.#lastFrameAt > 0 && gap <= MAX_FRAME_GAP_MS) this.#stats.push(gap)
    this.#lastFrameAt = now

    const cameraDistance = this.#rig.update()
    this.#city?.update(this.#camera, cameraDistance)
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
    this.#rig.resize(width, height)
    this.#stats.reset()
  }

  // A hidden tab pauses the loop; the gap is not a frame.
  #restartClock = () => {
    this.#lastFrameAt = 0
  }
}
