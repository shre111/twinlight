default:
    @just --list

setup:
    pnpm install
    uv sync

dev:
    pnpm turbo run dev

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
