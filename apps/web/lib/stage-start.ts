const NO_GRAPHICS =
  'This view needs WebGPU or WebGL 2. Try a recent Chrome, Edge, Safari or Firefox.'
const CONTEXT_LOST =
  'The graphics context was lost while the city was starting. Reload the page to try again.'

/** Runs `attempt`, and once more if it fails: a GPU context can be lost while it starts up. */
export async function withOneRetry<T>(attempt: () => Promise<T>): Promise<T> {
  try {
    return await attempt()
  } catch (error) {
    console.warn('The stage failed to start, trying once more', error)
    return attempt()
  }
}

/** Only a browser with neither API is told to change browser; a reload cannot help it. */
export function startFailureMessage(hasGraphics: boolean): string {
  return hasGraphics ? CONTEXT_LOST : NO_GRAPHICS
}

export async function browserHasGraphics(): Promise<boolean> {
  try {
    if (document.createElement('canvas').getContext('webgl2')) return true
  } catch {
    return true
  }
  // WebGPU can be exposed with no adapter behind it, which a reload would not change.
  return 'gpu' in navigator && (await navigator.gpu.requestAdapter().catch(() => null)) !== null
}
