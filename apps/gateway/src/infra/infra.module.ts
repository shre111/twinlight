import { type DynamicModule, Inject, Module, type OnApplicationShutdown } from '@nestjs/common'
import type { Config } from '../config.js'
import { createDatabase, DATABASE, type Database } from './database.js'
import { createStorage, STORAGE, type Storage } from './storage.js'

@Module({})
export class InfraModule implements OnApplicationShutdown {
  constructor(
    @Inject(DATABASE) private readonly database: Database,
    @Inject(STORAGE) private readonly storage: Storage,
  ) {}

  static forRoot(config: Config): DynamicModule {
    return {
      module: InfraModule,
      global: true,
      providers: [
        { provide: DATABASE, useFactory: () => createDatabase(config.DATABASE_URL) },
        { provide: STORAGE, useFactory: () => createStorage(config) },
      ],
      exports: [DATABASE, STORAGE],
    }
  }

  async onApplicationShutdown(): Promise<void> {
    this.storage.client.destroy()
    await this.database.end({ timeout: 5 })
  }
}
