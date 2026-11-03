import { execSync } from 'node:child_process'
import { defineConfig, devices } from '@playwright/test'

const ci = Boolean(process.env.CI)
const port = Number(process.env.SHOTS_PORT ?? 3101)
const shotsDir = '../../artifacts/shots'

function currentBranch(): string {
  try {
    return execSync('git branch --show-current', { encoding: 'utf8' }).trim()
  } catch {
    return ''
  }
}

// Playwright wipes the run directory before every run, so it must never fall back to the
// shots root that holds the baselines (an empty branch name on a detached HEAD would).
const runName = (
  process.env.SHOTS_BRANCH ||
  process.env.GITHUB_HEAD_REF ||
  currentBranch() ||
  'detached'
).replaceAll('/', '-')

// Software GL keeps stills comparable across machines; SHOTS_GPU=hardware measures the real GPU.
const softwareGl = [
  '--use-gl=angle',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
  '--ignore-gpu-blocklist',
]
const chromeArgs = process.env.SHOTS_GPU === 'hardware' ? [] : softwareGl
const installedChrome = ci ? {} : { channel: 'chrome' }

const engine = (name: string, use: object) => ({ name, testMatch: 'engines.shots.ts', use })
const engines = process.env.SHOTS_ENGINES
  ? [
      engine('chrome', { ...devices['Desktop Chrome'], ...installedChrome }),
      engine('firefox', devices['Desktop Firefox']),
      engine('webkit', devices['Desktop Safari']),
    ]
  : []

export default defineConfig({
  testDir: 'shots',
  testMatch: '*.shots.ts',
  outputDir: `${shotsDir}/${runName}`,
  snapshotPathTemplate: `${shotsDir}/baseline/{arg}-{projectName}{ext}`,
  updateSnapshots: ci ? 'none' : 'missing',
  workers: 1,
  reporter: 'list',
  timeout: 60_000,
  // A software GL screenshot takes close to 2 s and a stable still needs two in a row.
  expect: { timeout: 15_000, toHaveScreenshot: { maxDiffPixelRatio: 0.01 } },
  use: { baseURL: `http://localhost:${port}` },
  projects: [
    {
      name: 'desktop',
      testIgnore: 'engines.shots.ts',
      use: {
        ...installedChrome,
        viewport: { width: 1440, height: 900 },
        launchOptions: { args: chromeArgs },
      },
    },
    {
      name: 'phone',
      testIgnore: 'engines.shots.ts',
      use: {
        ...installedChrome,
        viewport: { width: 390, height: 844 },
        deviceScaleFactor: 3,
        isMobile: true,
        launchOptions: { args: chromeArgs },
      },
    },
    ...engines,
  ],
  webServer: {
    command: `pnpm exec next dev --port ${port}`,
    url: `http://localhost:${port}/dev/stage`,
    reuseExistingServer: !ci,
    timeout: 120_000,
  },
})
