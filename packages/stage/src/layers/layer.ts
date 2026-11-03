import type { Object3D } from 'three/webgpu'

export interface Layer {
  object: Object3D
  dispose(): void
}
