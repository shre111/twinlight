import {
  attribute,
  clamp,
  color,
  dot,
  exp,
  Fn,
  hash,
  instanceIndex,
  max,
  mix,
  normalWorld,
  oneMinus,
  positionWorld,
  smoothstep,
  step,
  vec3,
  vertexStage,
} from 'three/tsl'
import {
  BoxGeometry,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix4,
  MeshBasicNodeMaterial,
  type Node,
  Quaternion,
  Vector3,
} from 'three/webgpu'
import type { StageCity } from '../inputs'
import { cityGlow, palette } from '../palette'
import { type Fog, viewDepth } from './fog'
import type { Layer } from './layer'

type Buildings = StageCity['buildings']

const UP = new Vector3(0, 1, 0)
const SKY_LIGHT_DIRECTION = new Vector3(-0.35, 0, -0.94).normalize()
const position = new Vector3()
const scale = new Vector3()
const quaternion = new Quaternion()

export function createBuildings(buildings: Buildings, fog: Fog, warm: Node<'float'>): Layer {
  const geometry = new BoxGeometry(1, 1, 1).translate(0, 0.5, 0)
  geometry.deleteAttribute('uv')
  geometry.setAttribute('height', new InstancedBufferAttribute(buildings.height, 1))
  const material = new MeshBasicNodeMaterial()
  material.colorNode = buildingColour(fog, warm)
  const mesh = new InstancedMesh(geometry, material, buildings.count)
  const matrix = new Matrix4()
  for (let i = 0; i < buildings.count; i++)
    mesh.setMatrixAt(i, buildingMatrix(buildings, i, matrix))
  mesh.frustumCulled = false
  return {
    object: mesh,
    dispose: () => {
      geometry.dispose()
      material.dispose()
      mesh.dispose()
    },
  }
}

// Pack axes are x east, y north, z up and the stage shows them as (x, z, -y), so a
// counter-clockwise pack rotation is a positive turn about three.js +Y.
export function buildingMatrix(buildings: Buildings, i: number, out: Matrix4): Matrix4 {
  position.set(buildings.x[i] ?? 0, 0, -(buildings.y[i] ?? 0))
  scale.set(buildings.width[i] ?? 0, buildings.height[i] ?? 0, buildings.depth[i] ?? 0)
  quaternion.setFromAxisAngle(UP, buildings.rotation[i] ?? 0)
  return out.compose(position, quaternion, scale)
}

function buildingColour(fog: Fog, warm: Node<'float'>) {
  return Fn(() => {
    const normal = normalWorld
    const y = positionWorld.y
    const height = vertexStage(attribute('height', 'float'))
    const seed = vertexStage(hash(instanceIndex))
    const fogFactor = vertexStage(fog.factor(viewDepth))

    const roof = step(0.5, normal.y)
    const skyLit = max(dot(normal, vec3(SKY_LIGHT_DIRECTION)), 0)
    const shaded = color(palette.wall)
      .mul(seed.mul(0.3).add(0.78))
      .mul(normal.x.mul(0.2).add(0.8))
      .add(color(palette.skyLight).mul(skyLit).mul(0.08))
      .mul(mix(0.55, 1, smoothstep(0, 30, y)))
    const lip = smoothstep(height.sub(1.4), height.sub(0.1), y).mul(oneMinus(roof))
    const glow = exp(max(y, 0).div(8).negate())
    const wall = shaded
      .add(color(palette.edgeLip).mul(lip).mul(0.45))
      .add(cityGlow(warm).mul(glow).mul(0.05))
    const roofColour = color(palette.roof).mul(seed.mul(0.24).add(0.86))
    return clamp(fog.mix(mix(wall, roofColour, roof), fogFactor), 0, 1)
  })()
}
