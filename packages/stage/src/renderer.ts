import { WebGPURenderer } from 'three/webgpu'

export type Backend = 'webgpu' | 'webgl2'
export type BackendPreference = 'auto' | 'webgl2'

export async function createRenderer(
  canvas: HTMLCanvasElement,
  preference: BackendPreference,
): Promise<{ renderer: WebGPURenderer; backend: Backend }> {
  const renderer = new WebGPURenderer({
    canvas,
    antialias: true,
    powerPreference: 'high-performance',
    forceWebGL: preference === 'webgl2',
  })
  await renderer.init()
  const backend = 'isWebGPUBackend' in renderer.backend ? 'webgpu' : 'webgl2'
  return { renderer, backend }
}
