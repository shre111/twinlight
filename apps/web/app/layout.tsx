import type { Metadata } from 'next'
import type { ReactNode } from 'react'
import { body, display } from './fonts'
import '@twinlight/ui/tokens.css'
import './globals.css'

export const metadata: Metadata = {
  title: 'Twinlight',
  description: 'Rehearse your launch in a living city.',
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body>{children}</body>
    </html>
  )
}
