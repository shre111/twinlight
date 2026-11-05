import { uniform } from 'three/tsl'
import { type Camera, Group } from 'three/webgpu'
import { type CityExtent, cityBounds, extentOf } from '../extent'
import type { StageCity } from '../inputs'
import { createBuildings } from './buildings'
import { Fog } from './fog'
import { createGround } from './ground'
import type { Layer } from './layer'
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

  constructor(city: StageCity) {
    this.residents = city.residents
    this.extent = extentOf(cityBounds(city))
    this.#sky = createSky(this.fog, this.extent.scale)
    this.#layers = [
      this.#sky,
      createGround(this.fog, this.warm, this.extent),
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
