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
