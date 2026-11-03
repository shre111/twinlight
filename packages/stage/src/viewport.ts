export interface ViewportSize {
  width: number
  height: number
  pixelRatio: number
}

export function watchViewport(
  canvas: HTMLCanvasElement,
  maxPixelRatio: () => number,
  onChange: (size: ViewportSize) => void,
): { remeasure: () => void; stop: () => void } {
  let resolutionQuery: MediaQueryList | undefined

  const remeasure = () => {
    onChange({
      width: Math.max(1, canvas.clientWidth),
      height: Math.max(1, canvas.clientHeight),
      pixelRatio: Math.min(window.devicePixelRatio || 1, maxPixelRatio()),
    })
  }

  // A resolution query only fires once, when the window moves to a screen with a different DPR.
  const onResolutionChange = () => {
    watchResolution()
    remeasure()
  }
  const watchResolution = () => {
    resolutionQuery?.removeEventListener('change', onResolutionChange)
    resolutionQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`)
    resolutionQuery.addEventListener('change', onResolutionChange)
  }

  const observer = new ResizeObserver(remeasure)
  observer.observe(canvas)
  watchResolution()
  remeasure()

  return {
    remeasure,
    stop: () => {
      observer.disconnect()
      resolutionQuery?.removeEventListener('change', onResolutionChange)
    },
  }
}
