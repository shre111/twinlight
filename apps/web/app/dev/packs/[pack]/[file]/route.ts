import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { PACK_FILES } from '@twinlight/stage/pack'

// Serves generated city packs from TWL_PACK_DIR to the dev routes; MinIO can replace this later.
const FILES = new Set<string>(['manifest.json', ...PACK_FILES])
const PACK_NAME = /^[a-z0-9-]+$/

export async function GET(_request: Request, context: RouteContext<'/dev/packs/[pack]/[file]'>) {
  const dir = process.env.TWL_PACK_DIR
  const { pack, file } = await context.params
  if (process.env.NODE_ENV === 'production' || !dir || !PACK_NAME.test(pack) || !FILES.has(file)) {
    return new Response('Not found', { status: 404 })
  }
  try {
    const body = await readFile(path.join(dir, pack, file))
    const type = file.endsWith('.json') ? 'application/json' : 'application/octet-stream'
    return new Response(body, { headers: { 'content-type': type, 'cache-control': 'no-store' } })
  } catch {
    return new Response('Not found', { status: 404 })
  }
}
