import { z } from 'zod'

const configSchema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']).default('production'),
  GATEWAY_PORT: z.coerce.number().int().positive().default(8080),
  LOG_LEVEL: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace']).default('info'),
  GIT_SHA: z.string().min(7).optional(),
  DATABASE_URL: z.url({ protocol: /^postgres(ql)?$/ }),
  S3_ENDPOINT: z.url(),
  S3_REGION: z.string().min(1).default('auto'),
  S3_ACCESS_KEY_ID: z.string().min(1),
  S3_SECRET_ACCESS_KEY: z.string().min(1),
  S3_BUCKET_PACKS: z.string().min(1).default('packs'),
  S3_BUCKET_TAPES: z.string().min(1).default('tapes'),
})

export type Config = z.infer<typeof configSchema>

export function loadConfig(env: Record<string, string | undefined> = process.env): Config {
  const result = configSchema.safeParse(env)
  if (!result.success) throw new Error(`Invalid environment:\n${z.prettifyError(result.error)}`)
  return result.data
}
