import type { INestApplication } from '@nestjs/common'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { createApp } from '../src/app.js'
import { loadConfig } from '../src/config.js'

describe('gateway', () => {
  let app: INestApplication
  let baseUrl: string

  beforeAll(async () => {
    app = await createApp(
      loadConfig({ NODE_ENV: 'test', LOG_LEVEL: 'fatal', GIT_SHA: 'abc1234def56' }),
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

  it('serves the OpenAPI document with the health route', async () => {
    const response = await fetch(`${baseUrl}/docs-json`)
    const document = (await response.json()) as { paths: Record<string, unknown> }
    expect(Object.keys(document.paths)).toContain('/v1/health')
  })
})
