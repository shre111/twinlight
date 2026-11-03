---
name: ship
description: Commit, open, check, review and squash-merge one micro PR following the project's git rules. Use whenever a small, coherent step is done and green locally.
argument-hint: "[type(scope): subject]"
---

Ship the current branch as one micro PR. Title (if given): $ARGUMENTS

1. Confirm you are not on `main` and the branch name matches `feat|fix|chore|docs|refactor|perf|test/<slug>`.
2. Run the checks for what changed: `just lint` and the relevant test suite (`just test-engine`, `pnpm -F <pkg> test`, `uv run pytest` in the touched project). For stage work also run `/scene-check`. Fix failures before continuing.
3. Make sure every commit on the branch is a single line in Conventional Commits form with no body or trailers (`git log main..HEAD --format=%B` shows one line per commit). Commit remaining work as micro commits: `git add <paths> && git commit -m "type(scope): subject"`.
4. If the diff exceeds ~300 changed lines excluding tests, fixtures and generated files, or mixes concerns, stop and propose how to split it.
5. Ask the `reviewer` subagent to review `main...HEAD`. Fix blocking findings with new micro commits.
6. Push and open the PR:
   `git push -u origin <branch>`
   `gh pr create --title "<type(scope): subject>" --body "<what>.\n<why>.\n<how tested>."` (at most three short lines; no generated-by footer).
7. Wait for CI: `gh pr checks --watch`. If it fails, fix with a micro commit, push, wait again.
8. Merge: `gh pr merge --squash --delete-branch`.
9. `git switch main && git pull`.
10. If the plan point is now complete, tick it in `docs/PLAN.md` and run `/wrap` (lead lane only; other lanes write to `docs/lanes/<lane>.md`). Status file updates go out as their own `docs(plan): ...` micro PR or ride with the next PR of the same point.
