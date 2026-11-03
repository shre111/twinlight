import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, type Page, test } from '@playwright/test'
import type { FrameReport, TierName } from '@twinlight/stage'

const TIERS: TierName[] = ['high', 'medium', 'low']
const SAMPLE_MS = 3000
const HIDE_DEV_INDICATOR = '[data-nextjs-dev-overlay] { display: none !important; }'

async function openStage(page: Page, query: string) {
  await page.goto(`/dev/stage?${query}`)
  await page.addStyleTag({ content: HIDE_DEV_INDICATOR })
  await page.waitForFunction(() => (window.__twinlightStage?.perf().frames ?? 0) > 2)
}

test('empty stage', async ({ page }, testInfo) => {
  await openStage(page, 'hud=0')
  const still = path.join(testInfo.project.outputDir, `${testInfo.project.name}-empty.png`)
  await page.screenshot({ path: still })
  await expect(page).toHaveScreenshot('empty.png')
})

test('frame report', async ({ page }, testInfo) => {
  const report: Partial<Record<TierName, FrameReport>> = {}
  for (const tier of TIERS) {
    await openStage(page, `hud=0&tier=${tier}`)
    await page.waitForTimeout(SAMPLE_MS)
    report[tier] = await page.evaluate(() => window.__twinlightStage?.perf())
    expect(report[tier]?.frames, `${tier} tier rendered frames`).toBeGreaterThan(30)
  }
  const file = path.join(testInfo.project.outputDir, 'frame-report.json')
  const earlier = await readFile(file, 'utf8')
    .then((text) => JSON.parse(text) as object)
    .catch(() => ({}))
  await writeFile(file, JSON.stringify({ ...earlier, [testInfo.project.name]: report }, null, 2))
})
