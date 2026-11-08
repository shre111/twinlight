import { readFile } from 'node:fs/promises'
import path from 'node:path'

export const missing = () => new Response('Not found', { status: 404 })

/** A generated file for the dev routes, read from `dir`; never served in production. */
export async function devFile(dir: string | undefined, file: string): Promise<Response> {
  if (process.env.NODE_ENV === 'production' || !dir) return missing()
  try {
    const body = await readFile(path.join(dir, file))
    const type = file.endsWith('.json') ? 'application/json' : 'application/octet-stream'
    return new Response(body, { headers: { 'content-type': type, 'cache-control': 'no-store' } })
  } catch {
    return missing()
  }
}
