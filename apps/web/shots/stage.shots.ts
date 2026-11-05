import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, type Page, type TestInfo, test } from '@playwright/test'
import type { FrameReport, TierName } from '@twinlight/stage'

const TIERS: TierName[] = ['high', 'medium', 'low']
// A frame count rather than a time window, so software GL samples as many frames as a GPU.
const SAMPLE_FRAMES = 60
const SAMPLE_TIMEOUT_MS = 60_000
const LOAD_TIMEOUT_MS = 30_000
const HIDE_DEV_INDICATOR = '[data-nextjs-dev-overlay] { display: none !important; }'

async function openStage(page: Page, query: string) {
  await openRoute(page, `/dev/stage?${query}`)
}

async function openRoute(page: Page, url: string) {
  await page.goto(url)
  await page.addStyleTag({ content: HIDE_DEV_INDICATOR })
  await page.waitForFunction(
    () => window.__twinlightStage !== undefined || window.__twinlightStageError !== undefined,
    undefined,
    { timeout: LOAD_TIMEOUT_MS },
  )
  expect(
    await page.evaluate(() => window.__twinlightStageError),
    'stage load error (just fixture-pack builds a missing pack)',
  ).toBeUndefined()
  // The stage is exposed once its tape and time are set; three frames later a still shows them.
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        let left = 3
        const tick = () => (--left === 0 ? resolve() : requestAnimationFrame(tick))
        requestAnimationFrame(tick)
      }),
  )
}

async function captureStill(page: Page, testInfo: TestInfo, name: string) {
  const still = path.join(testInfo.project.outputDir, `${testInfo.project.name}-${name}.png`)
  await page.screenshot({ path: still })
  await expect(page).toHaveScreenshot(`${name}.png`)
}

// Day 10 has stores out of stock; `still=1` holds their flicker so two screenshots match.
for (const day of [1, 6, 10, 13]) {
  test(`city day ${day}`, async ({ page }, testInfo) => {
    await openStage(page, `hud=0&still=1&day=${day}`)
    await captureStill(page, testInfo, `city-day${day}`)
  })
}

test('city day 10 with reduced motion', async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await openStage(page, 'hud=0&day=10')
  const reduced = await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)
  expect(reduced, 'reduced motion emulated').toBe(true)
  await captureStill(page, testInfo, 'city-day10-reduced-motion')
})

test('fixture route starts at the launch, plays and fills its figures', async ({ page }) => {
  await openRoute(page, '/dev/fixture')
  const ordered = page.getByRole('region', { name: 'Live figures' }).locator('dd').first()
  await expect(ordered).toHaveText('0')

  await page.getByRole('button', { name: 'Play' }).click()
  await page.waitForFunction(() => (window.__twinlightStage?.time ?? 0) > 0)
  await page.getByRole('button', { name: 'Pause' }).click()

  await page.evaluate((day13) => window.__twinlightStage?.setTime(day13), 13 * 24 * 60)
  await expect(ordered).not.toHaveText('0')
})

test('frame report', async ({ page }, testInfo) => {
  // Three tiers of 60 software GL frames with 20k residents run past the default timeout.
  test.slow()
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
