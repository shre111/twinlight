import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, type Page, test } from '@playwright/test'
import type { FrameReport, TierName } from '@twinlight/stage'

const TIERS: TierName[] = ['high', 'medium', 'low']
// A frame count rather than a time window, so software GL samples as many frames as a GPU.
const SAMPLE_FRAMES = 60
const SAMPLE_TIMEOUT_MS = 30_000
const HIDE_DEV_INDICATOR = '[data-nextjs-dev-overlay] { display: none !important; }'

async function openStage(page: Page, query: string) {
  await page.goto(`/dev/stage?${query}`)
  await page.addStyleTag({ content: HIDE_DEV_INDICATOR })
  await page.waitForFunction(() => (window.__twinlightStage?.perf().frames ?? 0) > 2)
}

for (const day of [1, 6, 13]) {
  test(`city day ${day}`, async ({ page }, testInfo) => {
    await openStage(page, `hud=0&day=${day}`)
    const name = `city-day${day}`
    const still = path.join(testInfo.project.outputDir, `${testInfo.project.name}-${name}.png`)
    await page.screenshot({ path: still })
    await expect(page).toHaveScreenshot(`${name}.png`)
  })
}

test('frame report', async ({ page }, testInfo) => {
  const report: Partial<Record<TierName, FrameReport>> = {}
  for (const tier of TIERS) {
    await openStage(page, `hud=0&tier=${tier}`)
    await page.waitForFunction(
      (frames) => (window.__twinlightStage?.perf().frames ?? 0) >= frames,
      SAMPLE_FRAMES,
      { timeout: SAMPLE_TIMEOUT_MS },
    )
    report[tier] = await page.evaluate(() => window.__twinlightStage?.perf())
    expect(report[tier]?.frames, `${tier} tier rendered frames`).toBeGreaterThanOrEqual(
      SAMPLE_FRAMES,
    )
  }
  const file = path.join(testInfo.project.outputDir, 'frame-report.json')
  const earlier = await readFile(file, 'utf8')
    .then((text) => JSON.parse(text) as object)
    .catch(() => ({}))
  await writeFile(file, JSON.stringify({ ...earlier, [testInfo.project.name]: report }, null, 2))
})
