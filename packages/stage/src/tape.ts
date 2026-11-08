import { decodeTape, type Tape } from '@twinlight/tape'
import type { StageTape } from './inputs'

/** The parts of a decoded tape the stage draws; its columns are shared, not copied. */
export function stageTape({ header, residents, stores, aggregates }: Tape): StageTape {
  const { simMinutes, tickMinutes } = header
  return { simMinutes, tickMinutes, residents, stores, aggregates }
}

export async function fetchTape(url: string, load: typeof fetch = fetch): Promise<StageTape> {
  const response = await load(url)
  if (!response.ok) throw new Error(`Could not load the tape from ${url} (${response.status})`)
  return stageTape(await decodeTape(new Uint8Array(await response.arrayBuffer())))
}
