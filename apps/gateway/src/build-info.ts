import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'

export const BUILD_INFO = Symbol('BUILD_INFO')

export interface BuildInfo {
  version: string
  gitSha: string
}

export function readBuildInfo(gitShaFromEnv?: string): BuildInfo {
  return { version: readVersion(), gitSha: gitShaFromEnv ?? readGitSha() }
}

function readVersion(): string {
  const packageJson = readFileSync(new URL('../package.json', import.meta.url), 'utf8')
  return (JSON.parse(packageJson) as { version: string }).version
}

function readGitSha(): string {
  try {
    return execFileSync('git', ['rev-parse', '--short=12', 'HEAD'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
  } catch {
    return 'unknown'
  }
}
