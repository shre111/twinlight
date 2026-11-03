import { color, sky } from '@twinlight/ui/tokens'
import { Color, NoColorSpace } from 'three/webgpu'

type TokenHex = (typeof color)[keyof typeof color] | (typeof sky)[keyof typeof sky]

// Shades are mixed on the sRGB values designers read in tokens.json, then converted for the GPU.
function srgb(hex: TokenHex): Color {
  return new Color().setStyle(hex, NoColorSpace)
}

export function shade(hex: TokenHex, factor: number): Color {
  return srgb(hex).multiplyScalar(factor).convertSRGBToLinear()
}

export function blend(from: TokenHex, to: TokenHex, amount: number): Color {
  return srgb(from).lerp(srgb(to), amount).convertSRGBToLinear()
}

export const palette = {
  skyTop: shade(sky.zenith, 1),
  skyMid: shade(sky.mid, 1),
  horizon: shade(sky.horizon, 1),
  horizonBand: shade(sky.band, 1),
  fog: shade(color.dusk, 1),
  ground: shade(color.dusk, 1),
  cityGlow: shade(color.tube, 0.5),
  wall: shade(color.dusk, 1.33),
  roof: shade(color.dusk, 1.75),
  edgeLip: shade(color.haze, 1.13),
  skyLight: blend(sky.horizon, sky.band, 0.25),
  street: shade(color.haze, 1.25),
  arterial: blend(color.haze, color.mist, 0.35),
  bridge: blend(color.haze, color.mist, 0.55),
} as const
