import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { tokenOutputs } from '../scripts/tokens.ts'

describe('generated tokens', () => {
  it.each(Object.entries(tokenOutputs()))('%s matches tokens.json', (_kind, { url, content }) => {
    const message = `${fileURLToPath(url)} is stale; run pnpm -F @twinlight/ui tokens`
    expect(readFileSync(url, 'utf8'), message).toBe(content)
  })

  it('names css variables so tailwind theme namespaces can consume them', () => {
    const css = tokenOutputs().css.content
    for (const name of ['--color-dusk', '--color-light-lamp-ink', '--text-16', '--radius-card']) {
      expect(css).toContain(`${name}:`)
    }
    expect(css).toContain(
      '--font-display: var(--font-display-face, "Big Shoulders"), "Arial Narrow"',
    )
  })
})
