import { Wordmark } from '../components/wordmark'
import styles from './page.module.css'

export default function Home() {
  return (
    <main className={styles.main}>
      <h1 className={styles.title}>
        <Wordmark />
      </h1>
    </main>
  )
}
