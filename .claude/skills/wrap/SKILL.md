---
name: wrap
description: End-of-point or end-of-session update of docs/MEMORY.md and docs/PLAN.md so the next session starts with the right context. Use after finishing a plan point, before stopping work, or when switching lanes.
---

Update the status files. Lead lane only; in another lane, write the same content to `docs/lanes/<lane>.md` instead.

1. `docs/PLAN.md`: tick points whose acceptance criteria hold and whose PRs are merged with CI green. Do not tick partially done points; add a short note under the point instead (`Progress: residents layer merged, flare pending`).
2. `docs/MEMORY.md`:
   - **Now:** milestone, point, branch, the next concrete step.
   - **Recent changes:** one dated line per merged PR or meaningful step, newest first. Keep the last ~10; older lines are dropped (git history has them).
   - **Gotchas:** anything that cost time and would cost time again. Remove ones that no longer apply.
   - **Open questions:** add new ones; remove answered ones (and record the answer as a decision if it is one).
   - **Demo log:** add a line when a milestone demo is recorded.
   Keep the whole file under ~80 lines. Stable knowledge moves to ARCHITECTURE.md or DECISIONS.md.
3. Fold in any `docs/lanes/*.md` notes from other lanes, then clear those files.
4. Commit on a `docs/` branch (or with the current point's last PR): `docs(memory): wrap p0xx`.
