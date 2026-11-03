---
name: start-point
description: Start work on a plan point from docs/PLAN.md. Use when beginning any P-xxx point - reads its context, creates the branch and lays out the micro-PR steps.
argument-hint: "<P-xxx>"
---

Start plan point $ARGUMENTS.

1. Read the point in `docs/PLAN.md`: title, lane, markers (`[tests]`, `[deep]`), acceptance criteria. If it is already ticked or another point in the same milestone it depends on is open, say so and stop.
2. Read `docs/MEMORY.md` and the sections of `docs/ARCHITECTURE.md`, `docs/MODEL.md`, `docs/DESIGN.md` and `docs/DECISIONS.md` that the point touches. List any decision that constrains it.
3. If the point is `[deep]` and the current model is not the strongest available, say so in one line before continuing (KICKOFF.md has the model guide).
4. Make sure the tree is clean and `main` is current: `git switch main && git pull`.
5. Create the branch: `git switch -c feat/<point-id-lowercase-without-dash>-<two-to-four-word-slug>`, for example `feat/p016-city-residents-layers`. Use `fix/` or `chore/` when the point is that kind of work.
6. Write a short plan as a task list: the micro PRs you will open (each one concern, aim for under ~300 changed lines), and for each the micro commits you expect. If the point is `[tests]`, include a step that hands the acceptance criteria to the `test-author` subagent.
7. Name the subagents you will use (engine-smith, scene-director, persona-lab, test-author, reviewer) and which steps can run in parallel.
8. Start the first step. Do not wait for confirmation unless the point needs a decision that is not already in DECISIONS.md; in that case propose it with `/decide` and ask.
