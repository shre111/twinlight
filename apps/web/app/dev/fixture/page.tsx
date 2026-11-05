import { notFound } from 'next/navigation'
import { DevStage } from '../stage/dev-stage'

export default function DevFixturePage() {
  if (process.env.NODE_ENV === 'production') notFound()
  return <DevStage hud={false} still={false} ui source={{ fixture: 'toy' }} minutes={0} />
}
