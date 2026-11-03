export interface FrameSummary {
  frames: number
  fps: number
  p50: number
  p95: number
}

export class FrameStats {
  #samples: Float32Array
  #sorted: Float32Array
  #count = 0
  #next = 0

  constructor(capacity = 240) {
    this.#samples = new Float32Array(capacity)
    this.#sorted = new Float32Array(capacity)
  }

  reset(): void {
    this.#count = 0
    this.#next = 0
  }

  push(frameMs: number): void {
    this.#samples[this.#next] = frameMs
    this.#next = (this.#next + 1) % this.#samples.length
    this.#count = Math.min(this.#count + 1, this.#samples.length)
  }

  summary(): FrameSummary {
    const n = this.#count
    if (n === 0) return { frames: 0, fps: 0, p50: 0, p95: 0 }
    const sorted = this.#sorted.subarray(0, n)
    sorted.set(this.#samples.subarray(0, n))
    sorted.sort()
    let total = 0
    for (const ms of sorted) total += ms
    return {
      frames: n,
      fps: total > 0 ? (1000 * n) / total : 0,
      p50: nearestRank(sorted, 0.5),
      p95: nearestRank(sorted, 0.95),
    }
  }
}

function nearestRank(sorted: Float32Array, p: number): number {
  return sorted[Math.max(0, Math.ceil(p * sorted.length) - 1)] ?? 0
}
