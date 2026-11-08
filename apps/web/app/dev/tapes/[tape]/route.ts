import { devFile, missing } from '../../../../lib/dev-file'

// Serves engine tapes from TWL_TAPE_DIR to the dev routes; MinIO can replace this later.
const TAPE_NAME = /^[a-z0-9-]+$/

export async function GET(_request: Request, context: RouteContext<'/dev/tapes/[tape]'>) {
  const { tape } = await context.params
  if (!TAPE_NAME.test(tape)) return missing()
  return devFile(process.env.TWL_TAPE_DIR, `${tape}.twlt`)
}
