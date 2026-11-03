import type { FrameReport } from './stage'

const FIELDS = [
  ['backend', 'Backend'],
  ['tier', 'Tier'],
  ['p50', 'Median frame'],
  ['p95', '95th percentile'],
  ['fps', 'Frames per second'],
  ['drawCalls', 'Draw calls'],
  ['pixelRatio', 'Pixel ratio'],
] as const

type Field = (typeof FIELDS)[number][0]

const BACKEND_LABEL = { webgpu: 'WebGPU', webgl2: 'WebGL 2' } as const
const TIER_LABEL = { high: 'High', medium: 'Medium', low: 'Low' } as const

export class PerfHud {
  #root = document.createElement('dl')
  #values = new Map<Field, HTMLElement>()

  constructor(parent: HTMLElement) {
    Object.assign(this.#root.style, {
      position: 'absolute',
      top: '12px',
      left: '50%',
      transform: 'translateX(-50%)',
      display: 'flex',
      flexWrap: 'wrap',
      justifyContent: 'center',
      gap: '6px 16px',
      width: 'max-content',
      maxWidth: 'calc(100% - 24px)',
      margin: '0',
      padding: '8px 14px',
      borderRadius: 'var(--radius-card)',
      background: 'color-mix(in srgb, var(--color-dusk-2) 86%, transparent)',
      backdropFilter: 'blur(10px)',
      font: '400 var(--text-14) / 1.3 var(--font-body)',
      fontVariantNumeric: 'tabular-nums',
      color: 'var(--color-paper)',
      pointerEvents: 'none',
    })
    this.#root.setAttribute('aria-hidden', 'true')
    for (const [field, label] of FIELDS) {
      const cell = document.createElement('div')
      cell.style.whiteSpace = 'nowrap'
      const term = document.createElement('dt')
      const value = document.createElement('dd')
      term.textContent = label
      term.style.cssText = 'font-size: var(--text-12); color: var(--color-mist)'
      value.style.margin = '0'
      cell.append(term, value)
      this.#root.append(cell)
      this.#values.set(field, value)
    }
    parent.append(this.#root)
  }

  update(report: FrameReport): void {
    this.#set('backend', BACKEND_LABEL[report.backend])
    this.#set('tier', TIER_LABEL[report.tier])
    this.#set('p50', `${report.p50.toFixed(1)} ms`)
    this.#set('p95', `${report.p95.toFixed(1)} ms`)
    this.#set('fps', report.fps.toFixed(0))
    this.#set('drawCalls', String(report.drawCalls))
    this.#set('pixelRatio', report.pixelRatio.toFixed(2))
  }

  dispose(): void {
    this.#root.remove()
  }

  #set(field: Field, text: string): void {
    const element = this.#values.get(field)
    if (element && element.textContent !== text) element.textContent = text
  }
}
