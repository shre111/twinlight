import { mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { tokenOutputs } from './tokens.ts'

for (const { url, content } of Object.values(tokenOutputs())) {
  mkdirSync(new URL('.', url), { recursive: true })
  writeFileSync(url, content)
  console.log(`wrote ${fileURLToPath(url)}`)
}
