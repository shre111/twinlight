import path from 'node:path'
import { expect, test } from '@playwright/test'

test('stage starts and the hud shows backend and frame time', async ({ page, browserName }) => {
  await page.goto('/dev/stage')
  await page.waitForFunction(() => window.__twinlightStage !== undefined)
  await page.waitForTimeout(1500)
  const report = await page.evaluate(() => window.__twinlightStage?.perf())
  expect(report?.frames).toBeGreaterThan(30)
  await expect(page.getByText('Backend')).toBeVisible()
  await expect(page.getByText('Median frame')).toBeVisible()
  test.info().annotations.push({
    type: 'perf',
    description: `${browserName} ${report?.backend} p50 ${report?.p50.toFixed(1)} ms`,
  })
})

test('fixture route loads, plays and takes the keyboard', async ({
  page,
  browserName,
}, testInfo) => {
  const failures: string[] = []
  page.on('pageerror', (error) => failures.push(error.message))
  await page.goto('/dev/fixture')
  await page.addStyleTag({ content: '[data-nextjs-dev-overlay] { display: none !important; }' })
  await page.waitForFunction(
    () => window.__twinlightStage !== undefined || window.__twinlightStageError !== undefined,
    undefined,
    { timeout: 45_000 },
  )
  expect(await page.evaluate(() => window.__twinlightStageError)).toBeUndefined()
  const ordered = page.getByRole('region', { name: 'Live figures' }).locator('dd').first()
  await expect(ordered).toHaveText('0')

  await page.getByRole('button', { name: 'Play' }).click()
  await page.waitForFunction(() => (window.__twinlightStage?.time ?? 0) > 0)
  await page.getByRole('button', { name: 'Pause' }).click()

  await page.getByRole('slider', { name: 'Day of the launch' }).focus()
  await page.keyboard.press('End')
  await expect(ordered).not.toHaveText('0')
  await page.keyboard.press('Home')
  await expect(ordered).toHaveText('0')

  await page.evaluate((minutes) => window.__twinlightStage?.setTime(minutes), 13 * 24 * 60 + 1200)
  await expect(ordered).not.toHaveText('0')
  await page.waitForTimeout(500)
  await page.screenshot({
    path: path.join(testInfo.project.outputDir, `${browserName}-fixture-day14.png`),
  })
  const report = await page.evaluate(() => window.__twinlightStage?.perf())
  testInfo.annotations.push({
    type: 'perf',
    description: `${browserName} ${report?.backend} fixture p50 ${report?.p50.toFixed(1)} ms`,
  })
  expect(failures).toEqual([])
})
