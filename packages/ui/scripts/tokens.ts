import { readFileSync } from 'node:fs'

type Token = { $type: string; $value: unknown }
type TokenGroup = Record<string, Token>
type TokenFile = Record<string, TokenGroup | string>

type Shadow = { offsetX: string; offsetY: string; blur: string; spread: string; color: string }

const tokensUrl = new URL('../../../design/tokens.json', import.meta.url)
const generatedDir = new URL('../src/generated/', import.meta.url)

const header =
  'Generated from design/tokens.json by packages/ui/scripts/generate-tokens.ts. Do not edit.'

const cssPrefixByGroup: Record<string, string> = {
  color: 'color',
  light: 'color-light',
  font: 'font',
  size: 'text',
  radius: 'radius',
  shadow: 'shadow',
  motion: 'motion',
}
const tsOnlyGroups = new Set(['city'])

const genericFontFamilies = new Set([
  'serif',
  'sans-serif',
  'monospace',
  'cursive',
  'fantasy',
  'system-ui',
  'ui-serif',
  'ui-sans-serif',
  'ui-monospace',
  'ui-rounded',
  'math',
  'emoji',
  'fangsong',
])

function readTokens(): Record<string, TokenGroup> {
  const file = JSON.parse(readFileSync(tokensUrl, 'utf8')) as TokenFile
  return Object.fromEntries(
    Object.entries(file).filter(
      (entry): entry is [string, TokenGroup] => !entry[0].startsWith('$'),
    ),
  )
}

function quoteFamily(family: string): string {
  return genericFontFamilies.has(family) ? family : `"${family}"`
}

function cssFontStack(key: string, families: string[]): string {
  const [face = '', ...fallbacks] = families
  const hook = `var(--font-${key}-face, ${quoteFamily(face)})`
  return [hook, ...fallbacks.map(quoteFamily)].join(', ')
}

function cssShadow({ offsetX, offsetY, blur, spread, color }: Shadow): string {
  return `${offsetX} ${offsetY} ${blur} ${spread} ${color}`
}

function cssValue(key: string, token: Token): string {
  switch (token.$type) {
    case 'color':
      return String(token.$value).toLowerCase()
    case 'fontFamily':
      return cssFontStack(key, token.$value as string[])
    case 'shadow':
      return cssShadow(token.$value as Shadow)
    case 'cubicBezier':
      return `cubic-bezier(${(token.$value as number[]).join(', ')})`
    default:
      return String(token.$value)
  }
}

function renderCss(tokens: Record<string, TokenGroup>): string {
  const lines = Object.entries(tokens).flatMap(([group, entries]) => {
    if (tsOnlyGroups.has(group)) return []
    const prefix = cssPrefixByGroup[group]
    if (!prefix) throw new Error(`add a css prefix for token group "${group}" in cssPrefixByGroup`)
    return Object.entries(entries).map(
      ([key, token]) => `  --${prefix}-${key}: ${cssValue(key, token)};`,
    )
  })
  return `/* ${header} */\n:root {\n${lines.join('\n')}\n}\n`
}

function camelCase(key: string): string {
  return key.replace(/-(\w)/g, (_, letter: string) => letter.toUpperCase())
}

function tsKey(key: string): string {
  return /^\d+$/.test(key) ? key : camelCase(key)
}

function tsLiteral(value: unknown): string {
  if (typeof value === 'string') return `'${value}'`
  if (Array.isArray(value)) return `[${value.map(tsLiteral).join(', ')}]`
  return String(value)
}

function tsValue(token: Token): string {
  switch (token.$type) {
    case 'color':
      return tsLiteral(String(token.$value).toLowerCase())
    case 'dimension':
    case 'duration':
      return String(Number.parseFloat(String(token.$value)))
    case 'shadow':
      return tsLiteral(cssShadow(token.$value as Shadow))
    default:
      return tsLiteral(token.$value)
  }
}

const tsUnitNote: Record<string, string> = {
  size: 'Pixels.',
  radius: 'Pixels.',
  motion: 'Durations in milliseconds; ease is cubic-bezier control points.',
}

function renderTs(tokens: Record<string, TokenGroup>): string {
  const groups = Object.entries(tokens).map(([group, entries]) => {
    const note = tsUnitNote[group] ? `/** ${tsUnitNote[group]} */\n` : ''
    const props = Object.entries(entries).map(
      ([key, token]) => `  ${tsKey(key)}: ${tsValue(token)},`,
    )
    return `${note}export const ${camelCase(group)} = {\n${props.join('\n')}\n} as const\n`
  })
  return `/* ${header} */\n\n${groups.join('\n')}`
}

type Output = { url: URL; content: string }

export function tokenOutputs(): { css: Output; ts: Output } {
  const tokens = readTokens()
  return {
    css: { url: new URL('tokens.css', generatedDir), content: renderCss(tokens) },
    ts: { url: new URL('tokens.ts', generatedDir), content: renderTs(tokens) },
  }
}
