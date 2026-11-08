import { readFileSync } from 'node:fs'
import { decodeTape } from '@twinlight/tape'
import { describe, expect, it } from 'vitest'
import { fetchTape } from '../src/tape'

const FIXTURE = new URL('../../../fixtures/tape/v1/engine_supply_small_seed1.twlt', import.meta.url)
const bytes = new Uint8Array(readFileSync(FIXTURE))
const served = (async () => new Response(bytes)) as unknown as typeof fetch

describe('fetchTape', () => {
  it('gives the stage the decoded columns and the clock from the header', async () => {
    const decoded = await decodeTape(bytes)
    const tape = await fetchTape('/dev/tapes/small', served)
    expect(tape.simMinutes).toBe(decoded.header.simMinutes)
    expect(tape.tickMinutes).toBe(decoded.header.tickMinutes)
    expect(tape.residents.firstOrderAt).toEqual(decoded.residents.firstOrderAt)
    expect(tape.stores.stockoutIntervals).toEqual(decoded.stores.stockoutIntervals)
    expect(tape.aggregates.orders).toEqual(decoded.aggregates.orders)
    expect(tape.aggregates.orders).toHaveLength(decoded.header.ticks)
  })

  it('names the tape it could not load', async () => {
    const missing = (async () =>
      new Response('Not found', { status: 404 })) as unknown as typeof fetch
    await expect(fetchTape('/dev/tapes/gone', missing)).rejects.toThrow(/dev\/tapes\/gone.*404/)
  })

  it('refuses a tape that is not one', async () => {
    const junk = (async () => new Response(new Uint8Array(64))) as unknown as typeof fetch
    await expect(fetchTape('/dev/tapes/junk', junk)).rejects.toThrow()
  })
})
