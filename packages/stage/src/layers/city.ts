import { uniform } from 'three/tsl'
import { type Camera, Group, type Node } from 'three/webgpu'
import { type CityExtent, cityBounds, extentOf } from '../extent'
import type { StageCity } from '../inputs'
import { createBuildings } from './buildings'
import { Fog } from './fog'
import { createGround } from './ground'
import type { Layer } from './layer'
import { createRiver } from './river'
import { createRoads } from './roads'
import { createSky } from './sky'

export class CityLayer {
  readonly group = new Group()
  readonly fog = new Fog()
  /** 0 is the cool launch-day city, 1 fully warmed by orders. */
  readonly warm = uniform(0)
  readonly residents: StageCity['residents']
  readonly extent: CityExtent
  readonly #sky: Layer
  readonly #layers: Layer[]

  /** `clock` is wall-clock seconds for the river's sheen, which reduced motion holds still. */
  constructor(city: StageCity, clock: Node<'float'>, reducedMotion: boolean) {
    this.residents = city.residents
    this.extent = extentOf(cityBounds(city))
    this.#sky = createSky(this.fog, this.extent.scale)
    this.#layers = [
      this.#sky,
      createGround(this.fog, this.warm, this.extent),
      ...createRiver(city.river, this.fog, this.warm, clock, reducedMotion, this.extent.scale),
      ...createRoads(city.roads, this.fog, this.extent.scale),
      createBuildings(city.buildings, this.fog, this.warm),
    ]
    this.group.add(...this.#layers.map((layer) => layer.object))
  }

  update(camera: Camera, cameraDistance: number): void {
    this.fog.update(cameraDistance)
    this.#sky.object.position.copy(camera.position)
  }

  dispose(): void {
    for (const layer of this.#layers) layer.dispose()
    this.group.removeFromParent()
  }
}
