import { describe, expect, it, vi } from 'vitest'
import { advanceMinutes, ClockChannel, type ClockState, SPEEDS } from '../src/playback'

const RUN = 30 * 24 * 60

describe('advanceMinutes', () => {
  it('moves one sim hour per real second at 1x and scales with speed', () => {
    expect(advanceMinutes(0, 1000, 1, RUN)).toBe(60)
    expect(advanceMinutes(100, 500, 4, RUN)).toBe(220)
    expect(advanceMinutes(0, 1000, 16, RUN)).toBe(960)
  })

  it('stops at the end of the run', () => {
    expect(advanceMinutes(RUN - 1, 1000, 16, RUN)).toBe(RUN)
    expect(advanceMinutes(RUN, 16, 1, RUN)).toBe(RUN)
  })

  it('offers the three speeds the design shows', () => {
    expect(SPEEDS).toEqual([1, 4, 16])
  })
})

describe('ClockChannel', () => {
  const state = (minutes: number): ClockState => ({ minutes, playing: true, speed: 1 })

  it('hands listeners a new snapshot and keeps it stable between publishes', () => {
    const channel = new ClockChannel()
    const listener = vi.fn()
    channel.subscribe(listener)
    const before = channel.state
    expect(channel.state).toBe(before)
    channel.publish(state(10))
    expect(listener).toHaveBeenCalledOnce()
    expect(channel.state).not.toBe(before)
    expect(channel.state.minutes).toBe(10)
  })

  it('stops calling a listener once unsubscribed', () => {
    const channel = new ClockChannel()
    const listener = vi.fn()
    const unsubscribe = channel.subscribe(listener)
    unsubscribe()
    channel.publish(state(10))
    expect(listener).not.toHaveBeenCalled()
  })

  it('lets playback publish at most every 100 ms', () => {
    const channel = new ClockChannel()
    expect(channel.playbackDue(1000)).toBe(true)
    expect(channel.playbackDue(1016)).toBe(false)
    expect(channel.playbackDue(1099)).toBe(false)
    expect(channel.playbackDue(1100)).toBe(true)
    expect(channel.playbackDue(1116)).toBe(false)
  })
})
