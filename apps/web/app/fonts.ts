import localFont from 'next/font/local'

export const display = localFont({
  src: '../node_modules/@fontsource-variable/big-shoulders/files/big-shoulders-latin-standard-normal.woff2',
  weight: '100 900',
  variable: '--font-display-face',
  display: 'swap',
})

export const body = localFont({
  src: '../node_modules/@fontsource-variable/atkinson-hyperlegible-next/files/atkinson-hyperlegible-next-latin-wght-normal.woff2',
  weight: '200 800',
  variable: '--font-body-face',
  display: 'swap',
})
