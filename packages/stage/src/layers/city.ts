import { type Camera, Group } from 'three/webgpu'
import type { StageCity } from '../inputs'
import { createBuildings } from './buildings'
import { Fog } from './fog'
import { createGround } from './ground'
import type { Layer } from './layer'
import { createRoads } from './roads'
import { createSky } from './sky'

export class CityLayer {
  readonly group = new Group()
  readonly #fog = new Fog()
  readonly #sky: Layer
  readonly #layers: Layer[]

  constructor(city: StageCity) {
    this.#sky = createSky(this.#fog)
    this.#layers = [
      this.#sky,
      createGround(this.#fog),
      ...createRoads(city.roads, this.#fog),
      createBuildings(city.buildings, this.#fog),
    ]
    this.group.add(...this.#layers.map((layer) => layer.object))
  }

  update(camera: Camera, cameraDistance: number): void {
    this.#fog.update(cameraDistance)
    this.#sky.object.position.copy(camera.position)
  }

  dispose(): void {
    for (const layer of this.#layers) layer.dispose()
    this.group.removeFromParent()
  }
}
