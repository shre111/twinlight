import { PACK_FILES } from '@twinlight/stage/pack'
import { devFile, missing } from '../../../../../lib/dev-file'

// Serves generated city packs from TWL_PACK_DIR to the dev routes; MinIO can replace this later.
const FILES = new Set<string>(['manifest.json', ...PACK_FILES])
const PACK_NAME = /^[a-z0-9-]+$/

export async function GET(_request: Request, context: RouteContext<'/dev/packs/[pack]/[file]'>) {
  const { pack, file } = await context.params
  if (!PACK_NAME.test(pack) || !FILES.has(file)) return missing()
  return devFile(process.env.TWL_PACK_DIR, `${pack}/${file}`)
}
