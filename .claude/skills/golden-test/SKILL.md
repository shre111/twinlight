---
name: golden-test
description: Add or update determinism and cross-language golden tests for the engine, tape and city packs. Use when engine behaviour, the tape format or the city pack format changes, or when a golden hash fails.
argument-hint: "[what changed]"
---

Golden tests for: $ARGUMENTS

Goldens are how Twinlight proves every number is reproducible (D-006). Three kinds:

**1. Determinism goldens (Python engine).** In `py/engine/tests/test_golden.py`, for each fixture scenario under `fixtures/scenarios/`:
- Run seeds 1 and 2 through the Monte Carlo driver with `workers` set to 1, 4 and 8 (parametrised), so seed results never depend on how many processes ran them.
- Assert the tape blake3 hash is identical across worker counts and across two runs in the same process (catches hidden global state, such as an unseeded `np.random` call or a cached array mutated in place).
- On the pinned CI platform (`TWL_GOLDEN_STRICT=1`), assert the hash equals the checked-in value in `fixtures/golden/hashes.toml` (`[scenario] seed1 = "hex"`). Elsewhere, compare the probe values and day-level aggregates stored next to the hashes, with tolerance.
- On mismatch, print a diff summary: first differing section, first differing row, and both values, so the failure explains itself.

**2. Cross-language goldens (Python writes, TS reads).** `py/tape` writes `fixtures/tape/v<version>/<name>.twlt` from fixture scenarios (`uv run python -m twl_tape.fixtures`, with `--check` in CI to confirm the checked-in files are current). `packages/tape/test/golden.test.ts` decodes each file and checks section checksums, row counts and the probe values in `fixtures/tape/v<version>/<name>.probes.json` (for example resident 1042 `aware_at`, chatter row 37, store 3 stock at tick 900). City packs follow the same pattern under `fixtures/pack/`.

**3. Model anchors.** The Bass-limit test and the conservation and event-isolation properties from `docs/MODEL.md` section 10, written as hypothesis property tests with a fixed `derandomize=True` profile in CI.

Pitfalls that break goldens in Python:
- Dtype drift: write explicit little-endian dtypes (`astype('<u4')`, `'<f4'`) before `tobytes()`; never rely on platform defaults.
- Order: sort by `(minute, seq)` with a stable sort (`kind='stable'`) before writing events; never iterate a `set` or an unordered dict built from one.
- Numba: no `parallel=True`, `prange` or `fastmath=True` in sim code; each kernel receives its own `Generator`.
- Library upgrades: NumPy or Numba bumps can move the last bit of float maths. Bump them in their own PR and regenerate goldens there.

When a golden fails:
1. Decide whether the change in output is intended. If not, it is a bug: find it.
2. If intended, regenerate with `just goldens` (it runs in the pinned CI image and rewrites hashes and probe files), review the diff of `hashes.toml` and probes, and put the reason in the PR body ("Intended: awareness decay now per MODEL.md 3; hashes regenerated").
3. A tape or pack format change bumps the format version and adds a new fixture directory; old fixtures stay to test the reader's compatibility path until a decision retires them.

Never regenerate goldens to make an unexplained failure go away.
