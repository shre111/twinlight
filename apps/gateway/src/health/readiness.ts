import { HeadBucketCommand } from '@aws-sdk/client-s3'
import type { Database } from '../infra/database.js'
import type { Storage } from '../infra/storage.js'

const PROBE_TIMEOUT_MS = 2000

export type CheckStatus = 'ok' | 'down'

export interface Readiness {
  status: CheckStatus
  checks: { postgres: CheckStatus; storage: CheckStatus }
}

export type ProbeFailure = (check: string, error: unknown) => void

export async function checkReadiness(
  database: Database,
  storage: Storage,
  onFailure: ProbeFailure,
): Promise<Readiness> {
  const [postgres, storageStatus] = await Promise.all([
    probe('postgres', onFailure, () => database`select 1`),
    probe('storage', onFailure, () =>
      Promise.all(
        Object.values(storage.buckets).map((bucket) =>
          storage.client.send(new HeadBucketCommand({ Bucket: bucket }), {
            abortSignal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
          }),
        ),
      ),
    ),
  ])
  const allOk = postgres === 'ok' && storageStatus === 'ok'
  return { status: allOk ? 'ok' : 'down', checks: { postgres, storage: storageStatus } }
}

async function probe(
  check: string,
  onFailure: ProbeFailure,
  run: () => PromiseLike<unknown>,
): Promise<CheckStatus> {
  try {
    await run()
    return 'ok'
  } catch (error) {
    onFailure(check, error)
    return 'down'
  }
}
