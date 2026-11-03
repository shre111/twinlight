default:
    @just --list

setup:
    pnpm install

dev:
    pnpm turbo run dev

lint:
    pnpm biome check .
    pnpm turbo run typecheck

test:
    pnpm turbo run test
