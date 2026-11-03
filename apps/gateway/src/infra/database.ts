import postgres, { type Sql } from 'postgres'

export const DATABASE = Symbol('DATABASE')

export type Database = Sql

export function createDatabase(url: string): Database {
  return postgres(url, { max: 10, connect_timeout: 3, idle_timeout: 30 })
}
