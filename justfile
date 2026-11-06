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

goldens:
    TWL_GOLDEN_STRICT=1 NPY_ENABLE_CPU_FEATURES=X86_V3 NUMBA_CPU_NAME=generic NUMBA_DISABLE_INTEL_SVML=1 OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 uv run python -m twl_engine.goldens --write
    uv run python -m twl_tape.fixtures

fixture-pack:
    uv run python -m twl_engine.fixtures --out data/cache/packs/toy

shots *args: fixture-pack
    pnpm -F @twinlight/web shots {{ replace(args, "--update", "--update-snapshots=all") }}
