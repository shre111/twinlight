#!/usr/bin/env node
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { extname, join, relative } from 'node:path'

function run(cmd, args, cwd) {
  try {
    execFileSync(cmd, args, { cwd, stdio: 'ignore', timeout: 20000 })
  } catch {}
}

let input
try {
  input = JSON.parse(readFileSync(0, 'utf8'))
} catch {
  process.exit(0)
}

const file = input?.tool_input?.file_path
const root = process.env.CLAUDE_PROJECT_DIR || input?.cwd || process.cwd()
if (!file || !existsSync(file)) process.exit(0)
const rel = relative(root, file)
if (rel.startsWith('..') || rel.startsWith('.claude') || rel.startsWith('docs') || rel.includes('node_modules')) process.exit(0)

const ext = extname(file)
const biome = join(root, 'node_modules', '.bin', 'biome')

if (['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.json', '.jsonc', '.css'].includes(ext) && existsSync(biome)) {
  run(biome, ['format', '--write', file], root)
} else if (ext === '.py') {
  run('uv', ['run', '--quiet', 'ruff', 'format', file], root)
}
process.exit(0)
