import localFont from 'next/font/local'

export const display = localFont({
  src: '../node_modules/@fontsource-variable/big-shoulders/files/big-shoulders-latin-standard-normal.woff2',
  weight: '100 900',
  variable: '--font-display',
  display: 'swap',
})
