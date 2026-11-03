import { Controller, Get, Inject } from '@nestjs/common'
import { ApiOkResponse, ApiTags } from '@nestjs/swagger'
import { BUILD_INFO, type BuildInfo } from '../build-info.js'

interface HealthResponse extends BuildInfo {
  status: 'ok'
}

@ApiTags('health')
@Controller('health')
export class HealthController {
  constructor(@Inject(BUILD_INFO) private readonly buildInfo: BuildInfo) {}

  @Get()
  @ApiOkResponse({ description: 'Service status with the running version and git sha' })
  health(): HealthResponse {
    return { status: 'ok', ...this.buildInfo }
  }
}
