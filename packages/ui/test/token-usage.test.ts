import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { tokenOutputs } from '../scripts/tokens.ts'

const repoRoot = fileURLToPath(new URL('../../../', import.meta.url))
const generatedDir = join(repoRoot, 'packages/ui/src/generated')
const skippedDirNames = new Set(['node_modules', '.next', '.turbo', 'dist', 'public'])
const sourceExtensions = /\.(?:tsx?|css)$/

const hexColour = /(?<![\w&.])#(?:[\da-f]{8}|[\da-f]{6}|[\da-f]{3,4})(?![\w-])/gi
const colourFunction = /\b(?:rgba?|hsla?|hwb|oklab|oklch|lab|lch)\(/gi
const varReference = /var\(\s*(--[\w-]+)/g
const cssDeclaration = /(--[\w-]+)\s*:/g
const quotedName = /['"](--[\w-]+)['"]/g

function workspaceDirs(workspace: string): string[] {
  return readdirSync(join(repoRoot, workspace)).map((name) => join(repoRoot, workspace, name))
}

function walk(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) {
      return skippedDirNames.has(entry.name) || path === generatedDir ? [] : walk(path)
    }
    return entry.isFile() && sourceExtensions.test(entry.name) ? [relative(repoRoot, path)] : []
  })
}

function sourceFiles(): string[] {
  const roots = [
    ...workspaceDirs('apps'),
    ...workspaceDirs('packages').map((dir) => join(dir, 'src')),
  ]
  return roots.filter((root) => existsSync(root)).flatMap(walk)
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
  it('scans every app and package source, skipping build output and generated tokens', () => {
    const files = sourceFiles()
    expect(files).toContain('apps/web/components/wordmark.tsx')
    expect(files).toContain('apps/web/next.config.ts')
    expect(files.some((file) => /node_modules|\.next\/|generated\//.test(file))).toBe(false)
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
