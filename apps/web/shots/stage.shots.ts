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
    'stage load error (just fixture builds a missing pack or tape)',
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

// Measures the toy run's labels; real tapes bring other names and times.
test('timeline event labels stay apart at every width', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop', 'this test sets its own widths')
  await openStage(page, 'hud=0&still=1&day=13')
  const markers = page.locator('footer div[aria-hidden="true"]').first()
  for (const width of [701, 768, 900, 1024, 1180, 1280, 1366, 1400, 1401, 1440, 1920]) {
    await page.setViewportSize({ width, height: 900 })
    const gaps = await markers.evaluate((row) => {
      const boxes = [...row.querySelectorAll('span')]
        .filter((label) => label.children.length === 0 && label.getClientRects().length > 0)
        .map((label) => label.getBoundingClientRect())
        .sort((a, b) => a.left - b.left)
      return boxes.slice(1).map((box, i) => box.left - (boxes[i]?.right ?? 0))
    })
    expect(gaps.length, `labels at ${width} px`).toBeGreaterThan(1)
    expect(Math.min(...gaps), `narrowest gap at ${width} px`).toBeGreaterThanOrEqual(4)
    await expect(page.locator('footer').getByText('Day 15'), `at ${width} px`).toBeVisible()
  }
})

test.describe('when graphics fail at startup', () => {
  // Next's route announcer is also an alert, so name the stage's own.
  const alert = (page: Page) => page.locator('p[role="alert"]')

  test('one lost context is retried without the viewer noticing', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'desktop', 'runs once')
    await page.addInitScript(() => {
      const realGetContext = HTMLCanvasElement.prototype.getContext
      let failed = false
      // @ts-expect-error a stand-in for every getContext overload
      HTMLCanvasElement.prototype.getContext = function (...args: [string, unknown?]) {
        if (!failed) {
          failed = true
          throw new Error('context lost')
        }
        return realGetContext.apply(this, args)
      }
    })
    await openStage(page, 'hud=0&still=1&day=6')
    await expect(alert(page)).toHaveCount(0)
    await expect(page.getByRole('img', { name: 'Live city view' })).toHaveCount(1)
  })

  test('a context lost twice is named and suggests a reload', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'desktop', 'runs once')
    await page.addInitScript(() => {
      HTMLCanvasElement.prototype.getContext = () => {
        throw new Error('context lost')
      }
    })
    await page.goto('/dev/stage?hud=0')
    await expect(alert(page)).toContainText('graphics context was lost')
    await expect(alert(page)).toContainText('Reload')
  })

  test('a browser without graphics is told what it needs', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'desktop', 'runs once')
    await page.addInitScript(() => {
      Reflect.deleteProperty(Navigator.prototype, 'gpu')
      HTMLCanvasElement.prototype.getContext = () => null
    })
    await page.goto('/dev/stage?hud=0')
    await expect(alert(page)).toContainText('needs WebGPU or WebGL 2')
  })

  test('WebGPU with no adapter and no WebGL 2 counts as no graphics', async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== 'desktop', 'runs once')
    await page.addInitScript(() => {
      GPU.prototype.requestAdapter = async () => null
      HTMLCanvasElement.prototype.getContext = () => null
    })
    await page.goto('/dev/stage?hud=0')
    await expect(alert(page)).toContainText('needs WebGPU or WebGL 2')
  })
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
