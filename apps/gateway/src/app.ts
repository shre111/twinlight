import type { INestApplication } from '@nestjs/common'
import { NestFactory } from '@nestjs/core'
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger'
import { Logger } from 'nestjs-pino'
import { AppModule } from './app.module.js'
import { BUILD_INFO, type BuildInfo } from './build-info.js'
import type { Config } from './config.js'

export async function createApp(config: Config): Promise<INestApplication> {
  const app = await NestFactory.create(AppModule.forRoot(config), { bufferLogs: true })
  app.useLogger(app.get(Logger))
  app.setGlobalPrefix('v1')
  app.enableShutdownHooks()

  const { version } = app.get<BuildInfo>(BUILD_INFO)
  const openApi = new DocumentBuilder().setTitle('Twinlight gateway').setVersion(version).build()
  SwaggerModule.setup('docs', app, () => SwaggerModule.createDocument(app, openApi))
  return app
}
