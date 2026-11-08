import type { StageCity } from './inputs'
import { fetchPack } from './pack'
import { cityTransfers } from './pack-transfer'

export type PackReply = { city: StageCity } | { error: string }

self.onmessage = async ({ data: url }: MessageEvent<string>) => {
  try {
    const city = await fetchPack(url)
    self.postMessage({ city } satisfies PackReply, { transfer: cityTransfers(city) })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    self.postMessage({ error: message } satisfies PackReply)
  }
}
