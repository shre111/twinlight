import { describe, expect, it, vi } from 'vitest'
import { startFailureMessage, withOneRetry } from './stage-start'

describe('withOneRetry', () => {
  it('does not retry an attempt that works', async () => {
    const attempt = vi.fn(async () => 'stage')
    await expect(withOneRetry(attempt)).resolves.toBe('stage')
    expect(attempt).toHaveBeenCalledTimes(1)
  })

  it('tries once more after a failure', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    const attempt = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new Error('context lost'))
      .mockResolvedValueOnce('stage')
    await expect(withOneRetry(attempt)).resolves.toBe('stage')
    expect(attempt).toHaveBeenCalledTimes(2)
  })

  it('gives up after the second failure, with its error', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    const attempt = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new Error('first'))
      .mockRejectedValueOnce(new Error('second'))
    await expect(withOneRetry(attempt)).rejects.toThrow('second')
    expect(attempt).toHaveBeenCalledTimes(2)
  })
})

describe('startFailureMessage', () => {
  it('suggests a reload when the browser can do graphics', () => {
    const message = startFailureMessage(true)
    expect(message).toMatch(/graphics context was lost/)
    expect(message).toMatch(/Reload/)
    expect(message).not.toMatch(/Chrome/)
  })

  it('suggests another browser only when it cannot', () => {
    expect(startFailureMessage(false)).toMatch(/needs WebGPU or WebGL 2/)
  })
})
