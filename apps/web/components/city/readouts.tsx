'use client'

import type { Stage, StageTape } from '@twinlight/stage'
import { useSyncExternalStore } from 'react'
import { figuresAt } from '../../lib/run-figures'
import styles from './readouts.module.css'

const count = new Intl.NumberFormat('en-IN')
const share = new Intl.NumberFormat('en-IN', { style: 'percent', maximumFractionDigits: 0 })

/** The live figures card, read from the tape at the stage's clock. */
export function Readouts({ stage, tape }: { stage: Stage; tape: StageTape }) {
  const { minutes } = useSyncExternalStore(
    stage.subscribeClock,
    () => stage.clock,
    () => stage.clock,
  )
  const figures = figuresAt(tape, minutes)
  return (
    <section aria-label="Live figures" className={styles.card}>
      <dl className={styles.figures}>
        <div className={styles.figure}>
          <dt>
            <span className={styles.full}>People who have ordered</span>
            <span className={styles.short}>Have ordered</span>
          </dt>
          <dd className={styles.lamp}>{count.format(figures.ordered)}</dd>
        </div>
        <div className={styles.figure}>
          <dt>Heard about it</dt>
          <dd className={styles.chatter}>{share.format(figures.heardShare)}</dd>
        </div>
        <div className={`${styles.figure} ${styles.wide}`}>
          <dt>Orders today</dt>
          <dd>{count.format(figures.ordersToday)}</dd>
        </div>
        <div className={`${styles.figure} ${styles.wide}`}>
          <dt>Stores out of stock</dt>
          <dd className={figures.storesOut > 0 ? styles.sindoor : undefined}>
            {figures.storesOut} of {figures.stores}
          </dd>
        </div>
      </dl>
    </section>
  )
}
