import { type DynamicModule, Module } from '@nestjs/common'
import { LoggerModule } from 'nestjs-pino'
import { BUILD_INFO, readBuildInfo } from './build-info.js'
import type { Config } from './config.js'
import { HealthController } from './health/health.controller.js'
import { InfraModule } from './infra/infra.module.js'

@Module({})
export class AppModule {
  static forRoot(config: Config): DynamicModule {
    return {
      module: AppModule,
      imports: [
        LoggerModule.forRoot({
          pinoHttp: {
            level: config.LOG_LEVEL,
            redact: [
              'req.headers.authorization',
              'req.headers.cookie',
              'res.headers["set-cookie"]',
            ],
            transport: config.NODE_ENV === 'development' ? { target: 'pino-pretty' } : undefined,
          },
        }),
        InfraModule.forRoot(config),
      ],
      controllers: [HealthController],
      providers: [{ provide: BUILD_INFO, useValue: readBuildInfo(config.GIT_SHA) }],
    }
  }
}
