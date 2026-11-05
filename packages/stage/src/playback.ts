/** At 1x one sim hour passes per real second, so a 30-day run takes 12 minutes, or 45 s at 16x. */
export const SIM_MINUTES_PER_SECOND = 60
export const SPEEDS = [1, 4, 16] as const
export type Speed = (typeof SPEEDS)[number]

/** What the UI shows of the clock; replaced, never mutated, so React can compare snapshots. */
export interface ClockState {
  minutes: number
  playing: boolean
  speed: Speed
}

export function advanceMinutes(
  minutes: number,
  gapMs: number,
  speed: Speed,
  simMinutes: number,
): number {
  return Math.min(minutes + (gapMs / 1000) * SIM_MINUTES_PER_SECOND * speed, simMinutes)
}

// A longer gap may be a paused or throttled loop, so a frame never counts for more than this.
const MAX_FRAME_STEP_MS = 1000

/** How far a frame moves the clocks, in milliseconds; 0 for the first, which has no gap. */
export function frameStep(now: number, lastFrameAt: number): number {
  return lastFrameAt > 0 ? Math.min(now - lastFrameAt, MAX_FRAME_STEP_MS) : 0
}

const PLAYBACK_PUBLISH_MS = 100

/** Hands clock snapshots to the UI, at most ten a second during playback so React stays off the frame path. */
export class ClockChannel {
  #state: ClockState = { minutes: 0, playing: false, speed: 1 }
  #listeners = new Set<() => void>()
  #lastPublishAt = Number.NEGATIVE_INFINITY

  get state(): ClockState {
    return this.#state
  }

  subscribe(listener: () => void): () => void {
    this.#listeners.add(listener)
    return () => {
      this.#listeners.delete(listener)
    }
  }

  publish(state: ClockState, now = performance.now()): void {
    this.#state = state
    this.#lastPublishAt = now
    for (const listener of this.#listeners) {
      // A failing listener must not take the render loop down with it.
      try {
        listener()
      } catch (error) {
        console.error(error)
      }
    }
  }

  /** Whether a playback frame at `now` should publish; asked first so other frames allocate nothing. */
  playbackDue(now: number): boolean {
    return now - this.#lastPublishAt >= PLAYBACK_PUBLISH_MS
  }

  clear(): void {
    this.#listeners.clear()
  }
}
