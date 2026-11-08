import type { StageCity } from './inputs'
import { fetchPack } from './pack'
import type { PackReply } from './pack-worker'

/** Fetches and decodes a city pack in a Web Worker where there is one (D-032), so the page stays responsive. */
export function loadPack(baseUrl: string): Promise<StageCity> {
  if (typeof Worker === 'undefined') return fetchPack(baseUrl)
  const worker = new Worker(new URL('./pack-worker.ts', import.meta.url), { type: 'module' })
  const reply = new Promise<StageCity>((resolve, reject) => {
    worker.onmessage = ({ data }: MessageEvent<PackReply>) =>
      'city' in data ? resolve(data.city) : reject(new Error(data.error))
    worker.onerror = (event) => reject(new Error(event.message || 'The pack worker failed'))
  })
  worker.postMessage(new URL(baseUrl, location.href).href)
  return reply.finally(() => worker.terminate())
}
