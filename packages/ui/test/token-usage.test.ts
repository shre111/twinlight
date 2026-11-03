import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { tokenOutputs } from '../scripts/tokens.ts'

const repoRoot = fileURLToPath(new URL('../../../', import.meta.url))
const scannedDirs = [
  ['apps', 'app'],
  ['apps', 'components'],
  ['packages', 'src'],
] as const
const excludedDirs = ['packages/ui/src/generated/', 'node_modules/']
const sourceExtensions = /\.(?:tsx?|css)$/

const hexColour = /(?<![\w&.])#(?:[\da-f]{8}|[\da-f]{6}|[\da-f]{3,4})(?![\w-])/gi
const colourFunction = /\b(?:rgba?|hsla?|hwb|oklab|oklch|lab|lch)\(/gi
const varReference = /var\(\s*(--[\w-]+)/g
const cssDeclaration = /(--[\w-]+)\s*:/g
const quotedName = /['"](--[\w-]+)['"]/g

function sourceFiles(): string[] {
  return scannedDirs.flatMap(([workspace, dir]) => {
    const workspaceRoot = join(repoRoot, workspace)
    return readdirSync(workspaceRoot).flatMap((name) => {
      const root = join(workspaceRoot, name, dir)
      if (!existsSync(root)) return []
      return readdirSync(root, { recursive: true, encoding: 'utf8' })
        .map((file) => relative(repoRoot, join(root, file)))
        .filter((file) => sourceExtensions.test(file))
        .filter((file) => !excludedDirs.some((excluded) => file.includes(excluded)))
    })
  })
}

function findings(pattern: RegExp): string[] {
  return sourceFiles().flatMap((file) =>
    readFileSync(join(repoRoot, file), 'utf8')
      .split('\n')
      .flatMap((line, index) =>
        [...line.matchAll(pattern)].map((match) => `${file}:${index + 1} ${match[0]}`),
      ),
  )
}

function names(text: string, pattern: RegExp): string[] {
  return [...text.matchAll(pattern)].map((match) => match[1] ?? '')
}

describe('token usage in app and package source', () => {
  it('scans the web app', () => {
    expect(sourceFiles()).toContain('apps/web/components/wordmark.tsx')
  })

  it('has no hex colour literals; use @twinlight/ui tokens', () => {
    expect(findings(hexColour)).toEqual([])
  })

  it('has no rgb, hsl or other colour function literals; use @twinlight/ui tokens', () => {
    expect(findings(colourFunction)).toEqual([])
  })

  it('references only css variables that tokens or the source itself define', () => {
    const tokensCss = tokenOutputs().css.content
    const texts = sourceFiles().map((file) => readFileSync(join(repoRoot, file), 'utf8'))
    const defined = new Set([
      ...names(tokensCss, cssDeclaration),
      ...texts.flatMap((text) => [...names(text, cssDeclaration), ...names(text, quotedName)]),
    ])
    const undefinedReferences = sourceFiles().flatMap((file, i) =>
      names(texts[i] ?? '', varReference)
        .filter((name) => !defined.has(name))
        .map((name) => `${file} ${name}`),
    )
    expect(undefinedReferences).toEqual([])
  })
})
