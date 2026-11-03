import { execSync } from 'node:child_process'
import { defineConfig, devices } from '@playwright/test'

const port = Number(process.env.SHOTS_PORT ?? 3101)
const branch = (
  process.env.SHOTS_BRANCH ?? execSync('git branch --show-current', { encoding: 'utf8' })
)
  .trim()
  .replaceAll('/', '-')
const shotsDir = '../../artifacts/shots'

// Software GL keeps stills comparable across machines; SHOTS_GPU=hardware measures the real GPU.
const softwareGl = [
  '--use-gl=angle',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
  '--ignore-gpu-blocklist',
]
const chromeArgs = process.env.SHOTS_GPU === 'hardware' ? [] : softwareGl

const channel = process.env.CI ? undefined : 'chrome'
const engine = (name: string, device: string, engineChannel?: string) => ({
  name,
  testMatch: 'engines.shots.ts',
  use: { ...devices[device], channel: engineChannel, launchOptions: {} },
})
const engines = process.env.SHOTS_ENGINES
  ? [
      engine('chrome', 'Desktop Chrome', channel),
      engine('firefox', 'Desktop Firefox'),
      engine('webkit', 'Desktop Safari'),
    ]
  : []

export default defineConfig({
  testDir: 'shots',
  testMatch: '*.shots.ts',
  outputDir: `${shotsDir}/${branch}`,
  snapshotPathTemplate: `${shotsDir}/baseline/{arg}-{projectName}{ext}`,
  updateSnapshots: 'missing',
  workers: 1,
  reporter: 'list',
  timeout: 60_000,
  expect: { toHaveScreenshot: { maxDiffPixelRatio: 0.01 } },
  use: {
    baseURL: `http://localhost:${port}`,
    channel,
    launchOptions: { args: chromeArgs },
  },
  projects: [
    {
      name: 'desktop',
      testIgnore: 'engines.shots.ts',
      use: { viewport: { width: 1440, height: 900 } },
    },
    {
      name: 'phone',
      testIgnore: 'engines.shots.ts',
      use: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true },
    },
    ...engines,
  ],
  webServer: {
    command: `pnpm exec next dev --port ${port}`,
    url: `http://localhost:${port}/dev/stage`,
    reuseExistingServer: true,
    timeout: 120_000,
  },
})
