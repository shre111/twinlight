import { Controller, Get, Inject, Logger, ServiceUnavailableException } from '@nestjs/common'
import { ApiOkResponse, ApiServiceUnavailableResponse, ApiTags } from '@nestjs/swagger'
import { BUILD_INFO, type BuildInfo } from '../build-info.js'
import { DATABASE, type Database } from '../infra/database.js'
import { STORAGE, type Storage } from '../infra/storage.js'
import { checkReadiness, type Readiness } from './readiness.js'

interface HealthResponse extends BuildInfo {
  status: 'ok'
}

@ApiTags('health')
@Controller('health')
export class HealthController {
  private readonly logger = new Logger(HealthController.name)

  constructor(
    @Inject(BUILD_INFO) private readonly buildInfo: BuildInfo,
    @Inject(DATABASE) private readonly database: Database,
    @Inject(STORAGE) private readonly storage: Storage,
  ) {}

  @Get()
  @ApiOkResponse({ description: 'Service status with the running version and git sha' })
  health(): HealthResponse {
    return { status: 'ok', ...this.buildInfo }
  }

  @Get('ready')
  @ApiOkResponse({ description: 'Postgres and object storage are reachable' })
  @ApiServiceUnavailableResponse({ description: 'A dependency is down; see checks' })
  async ready(): Promise<Readiness> {
    const readiness = await checkReadiness(this.database, this.storage, (check, error) =>
      this.logger.warn(`${check} readiness check failed: ${String(error)}`),
    )
    if (readiness.status !== 'ok') throw new ServiceUnavailableException(readiness)
    return readiness
  }
}
