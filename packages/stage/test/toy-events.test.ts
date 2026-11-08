import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { TOY_EVENTS } from '../src/toy-events'

const SCENARIO = new URL('../../../fixtures/scenarios/prototype_launch.toml', import.meta.url)

// The fixture tape does not carry its campaigns, so the timeline's markers are checked against
// the scenario that made it.
function campaignStart(name: string): number | undefined {
  const toml = readFileSync(SCENARIO, 'utf8')
  const block = toml
    .split('[[scenario.campaigns]]')
    .find((part) => part.includes(`name = "${name}"`))
  const start = block?.match(/^start_minute = (\d+)$/m)?.[1]
  return start === undefined ? undefined : Number(start)
}

describe('the toy launch events', () => {
  it('mark the fixture scenario launch and creator campaigns', () => {
    expect(TOY_EVENTS.map((event) => event.at)).toEqual([
      campaignStart('launch'),
      campaignStart('creator'),
    ])
  })
})
