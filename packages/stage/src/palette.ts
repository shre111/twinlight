import { color, sky } from '@twinlight/ui/tokens'
import { color as colorNode, mix } from 'three/tsl'
import { Color, NoColorSpace, type Node } from 'three/webgpu'

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
  cityGlowWarm: shade(color.lamp, 1),
  tube: shade(color.tube, 1),
  chatter: shade(color.chatter, 1),
  lamp: shade(color.lamp, 1),
  sindoor: shade(color.sindoor, 1),
  wall: shade(color.dusk, 1.33),
  roof: shade(color.dusk, 1.75),
  edgeLip: shade(color.haze, 1.13),
  skyLight: blend(sky.horizon, sky.band, 0.25),
  street: shade(color.haze, 1.25),
  arterial: blend(color.haze, color.mist, 0.35),
  bridge: blend(color.haze, color.mist, 0.55),
  riverDeep: shade(color.dusk, 0.6),
  riverSheen: blend(color.haze, color.monsoon, 0.46),
  riverBand: shade(sky.band, 0.8),
} as const

/** The city's ambient glow, cool tubelight until orders warm it to sodium amber. */
export function cityGlow(warm: Node<'float'>) {
  return mix(colorNode(palette.cityGlow), colorNode(palette.cityGlowWarm), warm)
}
