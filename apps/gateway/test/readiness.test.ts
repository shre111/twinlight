import type { S3Client } from '@aws-sdk/client-s3'
import { describe, expect, it, vi } from 'vitest'
import { checkReadiness } from '../src/health/readiness.js'
import type { Database } from '../src/infra/database.js'
import type { Storage } from '../src/infra/storage.js'

function fakeDatabase(query: () => Promise<unknown>): Database {
  return query as unknown as Database
}

function fakeStorage(send: () => Promise<unknown>): Storage {
  return { client: { send } as unknown as S3Client, buckets: { packs: 'packs', tapes: 'tapes' } }
}

const reachable = () => Promise.resolve({})
const refused = () => Promise.reject(new Error('connect ECONNREFUSED'))
const hung = () => new Promise<never>(() => {})

describe('checkReadiness', () => {
  it('is ok when postgres and both buckets answer', async () => {
    const onFailure = vi.fn()
    const readiness = await checkReadiness(
      fakeDatabase(reachable),
      fakeStorage(reachable),
      onFailure,
    )
    expect(readiness).toEqual({ status: 'ok', checks: { postgres: 'ok', storage: 'ok' } })
    expect(onFailure).not.toHaveBeenCalled()
  })

  it('reports only the failing dependency', async () => {
    const onFailure = vi.fn()
    const readiness = await checkReadiness(fakeDatabase(reachable), fakeStorage(refused), onFailure)
    expect(readiness).toEqual({ status: 'down', checks: { postgres: 'ok', storage: 'down' } })
    expect(onFailure).toHaveBeenCalledWith('storage', expect.any(Error))
  })

  it('times out a postgres query that never answers', async () => {
    const onFailure = vi.fn()
    const readiness = await checkReadiness(
      fakeDatabase(hung),
      fakeStorage(reachable),
      onFailure,
      20,
    )
    expect(readiness.checks.postgres).toBe('down')
    expect(onFailure).toHaveBeenCalledWith('postgres', expect.any(Error))
  })
})
