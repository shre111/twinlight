import styles from './wordmark.module.css'

export function Wordmark() {
  return (
    <span className={styles.wordmark}>
      <svg className={styles.mark} aria-hidden="true" viewBox="0 0 28 20">
        <circle cx="10" cy="10" r="8" fill="var(--color-tube)" />
        <circle cx="18" cy="10" r="8" fill="var(--color-lamp)" />
      </svg>
      Twinlight
    </span>
  )
}
