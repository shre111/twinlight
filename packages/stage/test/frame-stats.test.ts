import { describe, expect, it } from 'vitest'
import { FrameStats } from '../src/frame-stats'

describe('FrameStats', () => {
  it('reports zeros before any frame', () => {
    expect(new FrameStats().summary()).toEqual({ frames: 0, fps: 0, p50: 0, p95: 0 })
  })

  it('uses nearest-rank percentiles', () => {
    const stats = new FrameStats(100)
    for (let ms = 1; ms <= 100; ms++) stats.push(ms)
    const { frames, p50, p95 } = stats.summary()
    expect(frames).toBe(100)
    expect(p50).toBe(50)
    expect(p95).toBe(95)
  })

  it('keeps only the most recent frames once full', () => {
    const stats = new FrameStats(4)
    for (const ms of [100, 100, 100, 100, 10, 10, 10, 20]) stats.push(ms)
    const { frames, p50, p95, fps } = stats.summary()
    expect(frames).toBe(4)
    expect(p50).toBe(10)
    expect(p95).toBe(20)
    expect(fps).toBeCloseTo(80)
  })
})
