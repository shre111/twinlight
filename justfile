compose := "docker compose -f infra/compose/dev.yml"

default:
    @just --list

setup:
    pnpm install
    uv sync

dev: infra
    pnpm turbo run dev

infra:
    {{ compose }} up -d --wait
    {{ compose }} run --rm minio-buckets

down:
    {{ compose }} down

lint:
    pnpm biome check .
    pnpm turbo run typecheck
    uv run ruff format --check .
    uv run ruff check .
    uv run pyright

test:
    pnpm turbo run test
    uv run pytest -q

test-engine:
    uv run pytest -q py

shots *args:
    pnpm -F @twinlight/web shots {{ replace(args, "--update", "--update-snapshots=all") }}
