import type { INestApplication } from '@nestjs/common'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { createApp } from '../src/app.js'
import { loadConfig } from '../src/config.js'

const unreachableDependencies = {
  DATABASE_URL: 'postgres://twinlight:twinlight@127.0.0.1:1/twinlight',
  S3_ENDPOINT: 'http://127.0.0.1:1',
  S3_ACCESS_KEY_ID: 'test',
  S3_SECRET_ACCESS_KEY: 'test',
}

describe('gateway', () => {
  let app: INestApplication
  let baseUrl: string

  beforeAll(async () => {
    app = await createApp(
      loadConfig({
        NODE_ENV: 'test',
        LOG_LEVEL: 'fatal',
        GIT_SHA: 'abc1234def56',
        ...unreachableDependencies,
      }),
    )
    await app.listen(0, '127.0.0.1')
    baseUrl = await app.getUrl()
  })

  afterAll(async () => {
    await app.close()
  })

  it('reports version and git sha on /v1/health', async () => {
    const response = await fetch(`${baseUrl}/v1/health`)
    expect(response.status).toBe(200)
    expect(await response.json()).toEqual({
      status: 'ok',
      version: '0.0.0',
      gitSha: 'abc1234def56',
    })
  })

  it('reports 503 on /v1/health/ready when postgres and storage are unreachable', async () => {
    const response = await fetch(`${baseUrl}/v1/health/ready`)
    expect(response.status).toBe(503)
    expect(await response.json()).toEqual({
      status: 'down',
      checks: { postgres: 'down', storage: 'down' },
    })
  }, 10_000)

  it('serves the OpenAPI document with the health routes', async () => {
    const response = await fetch(`${baseUrl}/docs-json`)
    const document = (await response.json()) as { paths: Record<string, unknown> }
    expect(Object.keys(document.paths)).toEqual(
      expect.arrayContaining(['/v1/health', '/v1/health/ready']),
    )
  })
})
