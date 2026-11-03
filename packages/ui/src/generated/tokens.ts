/* Generated from design/tokens.json by packages/ui/scripts/generate-tokens.ts. Do not edit. */

export const color = {
  dusk: '#0f1330',
  dusk2: '#171c42',
  haze: '#2a3166',
  mist: '#9aa3cf',
  paper: '#eef0fb',
  lamp: '#ffb547',
  tube: '#a6f0da',
  chatter: '#c8b6ff',
  sindoor: '#ff5468',
  monsoon: '#6f8cc9',
} as const

export const sky = {
  zenith: '#04061a',
  mid: '#0b0f2e',
  horizon: '#231f4e',
  band: '#4a3c7e',
} as const

export const light = {
  ground: '#f3f4fb',
  text: '#0f1330',
  muted: '#4a5288',
  line: '#d5d9ee',
  lampInk: '#9a5b00',
} as const

export const font = {
  display: ['Big Shoulders', 'Arial Narrow', 'Roboto Condensed', 'sans-serif'],
  body: ['Atkinson Hyperlegible Next', 'Segoe UI', 'system-ui', 'sans-serif'],
} as const

/** Pixels. */
export const size = {
  12: 12,
  14: 14,
  16: 16,
  18: 18,
  21: 21,
  24: 24,
  32: 32,
  44: 44,
  60: 60,
  80: 80,
} as const

/** Pixels. */
export const radius = {
  card: 14,
  input: 10,
  pill: 999,
  docked: 0,
} as const

export const shadow = {
  card: '0px 12px 40px 0px rgba(5,7,20,0.45)',
} as const

/** Durations in milliseconds; ease is cubic-bezier control points. */
export const motion = {
  card: 220,
  sheet: 260,
  intro: 2500,
  ease: [0.2, 0.8, 0.2, 1],
} as const

export const city = {
  residentUnawareIntensity: 0.2,
  residentAwareIntensity: 0.5,
  residentOrderedIntensity: 0.9,
  flareSimHours: 6,
  arcTravelSimHours: 3,
} as const
