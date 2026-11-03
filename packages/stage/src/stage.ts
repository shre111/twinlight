import { PerspectiveCamera, Scene, type WebGPURenderer } from 'three/webgpu'
import { CameraRig } from './camera'
import { FrameStats, type FrameSummary } from './frame-stats'
import type { StageCity, StageTape } from './inputs'
import { CityLayer } from './layers/city'
import { ResidentsLayer } from './layers/residents'
import { StoresLayer } from './layers/stores'
import { palette } from './palette'
import { PerfHud } from './perf-hud'
import { type Backend, type BackendPreference, createRenderer } from './renderer'
import { initialTier, TIERS, type TierName, type TierOption } from './tiers'
import { clampTime, warmFraction } from './time'
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
  #tape: StageTape | null = null
  #residents: ResidentsLayer | null = null
  #stores: StoresLayer | null = null
  #time = 0
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

  /** Sim minutes the city is shown at. */
  get time(): number {
    return this.#time
  }

  setTier(tier: TierName): void {
    this.#tier = tier
    this.#stats.reset()
    this.#viewport.remeasure()
    this.#mountResidents()
    // Compile the remounted residents now rather than on the next frame.
    void this.#compile().catch((error: unknown) => console.error(error))
  }

  /** Replaces the city. Its tape is cleared, so call `setTape` again for the new one. */
  async loadCity(city: StageCity): Promise<void> {
    if (this.#disposed) return
    const load = ++this.#loads
    this.#tape = null
    this.#residents?.dispose()
    this.#residents = null
    this.#stores?.dispose()
    this.#stores = null
    this.#city?.dispose()
    const layer = new CityLayer(city)
    this.#city = layer
    this.#scene.add(layer.group)
    if (city.stores.count > 0) {
      this.#stores = new StoresLayer(city.stores, layer.fog)
      this.#scene.add(this.#stores.object)
    }
    this.#renderer.setClearColor(palette.skyTop, 1)
    await this.#compile()
    // A newer load or a dispose may have landed while the materials compiled.
    if (this.#disposed || load !== this.#loads) return
    this.#stats.reset()
  }

  /** Uploads the residents' event times once; the shaders evaluate them at `setTime` (D-004). */
  async setTape(tape: StageTape): Promise<void> {
    if (this.#disposed) return
    const city = this.#city
    if (!city) throw new Error('Load a city before setting its tape')
    const expected = city.residents.count
    const { awareAt, firstOrderAt } = tape.residents
    if (awareAt.length !== expected || firstOrderAt.length !== expected) {
      throw new Error(`The tape has ${awareAt.length} residents but the city has ${expected}`)
    }
    // CSR offsets carry one more entry than there are stores, so even a storeless tape has [0].
    const offsets = tape.stores.stockoutOffsets
    if (offsets.length === 0)
      throw new Error('The tape has no store offsets; expected at least [0]')
    const stores = offsets.length - 1
    const cityStores = this.#stores?.count ?? 0
    if (stores !== cityStores) {
      throw new Error(`The tape has ${stores} stores but the city has ${cityStores}`)
    }
    const load = this.#loads
    this.#tape = tape
    this.#mountResidents()
    await this.#compile()
    if (!this.#disposed && load !== this.#loads) {
      throw new Error('A new city loaded while its tape was being set; set the tape again')
    }
    // Frames rendered while the residents compiled are not the scene's steady frame time.
    this.#stats.reset()
  }

  /** Clamps to the tape, updates uniforms and each store's stock state; no work grows with residents. */
  setTime(simMinutes: number): void {
    const tape = this.#tape
    this.#time = tape
      ? clampTime(simMinutes, tape)
      : Number.isFinite(simMinutes)
        ? Math.max(0, simMinutes)
        : 0
    this.#applyTime()
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
    this.#residents?.dispose()
    this.#stores?.dispose()
    this.#city?.dispose()
    this.#rig.dispose()
    void this.#renderer.dispose()
  }

  #mountResidents(): void {
    this.#residents?.dispose()
    this.#residents = null
    const city = this.#city
    const tape = this.#tape
    if (!city || !tape) return
    const layer = new ResidentsLayer(
      city.residents,
      tape.residents,
      TIERS[this.#tier].residents,
      city.fog,
      TIERS[this.#tier].bloomScale > 0,
    )
    this.#residents = layer
    this.#scene.add(layer.object)
    this.setTime(this.#time)
  }

  #applyTime(): void {
    const city = this.#city
    const tape = this.#tape
    if (!city || !tape) return
    if (this.#residents) this.#residents.time.value = this.#time
    city.warm.value = warmFraction(this.#time, tape, city.residents.count)
    // At the run's end the last minute holds, so a stockout still open then stays shown.
    this.#stores?.showStockouts(tape.stores, Math.min(this.#time, tape.simMinutes - 1))
  }

  async #compile(): Promise<void> {
    try {
      await this.#renderer.compileAsync(this.#scene, this.#camera)
    } catch (error) {
      if (!this.#disposed) throw error
    }
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
    this.#residents?.update(this.#camera)
    this.#stores?.update(this.#camera)
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
