import { S3Client } from '@aws-sdk/client-s3'
import type { Config } from '../config.js'

export const STORAGE = Symbol('STORAGE')

export interface Storage {
  client: S3Client
  buckets: { packs: string; tapes: string }
}

export function createStorage(config: Config): Storage {
  return {
    client: new S3Client({
      endpoint: config.S3_ENDPOINT,
      region: config.S3_REGION,
      forcePathStyle: true,
      credentials: {
        accessKeyId: config.S3_ACCESS_KEY_ID,
        secretAccessKey: config.S3_SECRET_ACCESS_KEY,
      },
    }),
    buckets: { packs: config.S3_BUCKET_PACKS, tapes: config.S3_BUCKET_TAPES },
  }
}
