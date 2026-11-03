# Kickoff

How to start building Twinlight with Claude Code, and the prompts to paste.

## Before the first session (you, once)
1. Create the repo and seed it with this kit (the only commit made directly on `main`):
   ```
   mkdir twinlight && cd twinlight
   # copy the kit contents here
   git init -b main
   git add .
   git commit -m "chore: seed planning docs and claude code setup"
   gh repo create <owner>/twinlight --private --source . --push
   ```
2. GitHub repo settings (Settings > General > Pull Requests):
   - Allow squash merging only; default commit message: **Pull request title**.
   - Automatically delete head branches.
3. Branch protection on `main`: require a PR, require status checks `pr-title` and the `ci` jobs once they exist, no force pushes.
4. Install locally: Node 24 LTS, pnpm 10, uv (it installs Python 3.13 for you), just, Docker, gh (logged in).
5. Copy `.env.example` to `.env` and fill what you have (the LLM key is not needed until M6).

## Models
| Work | Model |
| --- | --- |
| Default driver for every session | Opus 5.5 (`/model opus`) |
| Points marked `[deep]` (engine core, Bass anchor, tape v1, residents and arcs shaders, population synthesis, social graph, calibration, backtests, compare view, voice) | Fable 5.1 (`/model fable`) for that point, back to Opus after |
| `test-author`, `reviewer` subagents | Sonnet 5.5 (set in their files) |

## Session rhythm
Every session: read `docs/MEMORY.md`, `/start-point <next point>`, work in micro commits, `/ship` each micro PR, `/wrap` at the end of the point or session.

## Prompt 1: lead session (Lane A), paste this first
```
You are the lead engineer on Twinlight. Read CLAUDE.md, then docs/PLAN.md, docs/MEMORY.md, docs/DECISIONS.md, docs/ARCHITECTURE.md, docs/MODEL.md and docs/DESIGN.md in full before doing anything.

Then give me, in under 25 lines:
1. Your understanding of the product and the two things CLAUDE.md says to protect.
2. Anything in the docs that is contradictory, ambiguous or that you would push back on, with a concrete proposal for each (use /decide for anything that is a real decision; do not change course silently).
3. The order you will take M0 in, which points can run in parallel lanes, and which subagents you will use.

Wait for my go. After that, work through M0 point by point with /start-point and /ship: one branch per point, micro commits with one-line Conventional Commit messages and no bodies, micro PRs squash-merged after CI and the reviewer pass. Tests only where the plan or CLAUDE.md says a feature is big. Minimal comments. Update MEMORY.md and PLAN.md with /wrap after each point. Stop and ask me only for real decisions, missing credentials, or anything destructive.
```

## Prompt 2: stage lane (Lane B), start after P-001 is merged
Open a second terminal: `claude --worktree stage`
```
You are Lane B (stage) on Twinlight. Read CLAUDE.md, docs/DESIGN.md, docs/ARCHITECTURE.md sections 2, 4 and 9, and D-002 to D-004 and D-019. Open design/prototype/twinlight-city.html in a browser; it is the visual reference for the city.

You own packages/stage, packages/ui and the scene pages in apps/web. Do not edit PLAN.md, MEMORY.md or DECISIONS.md; write progress and questions to docs/lanes/stage.md. Contract or tape format changes go to Lane A.

Start with P-007, then P-015 to P-018 using a synthetic in-memory tape until P-014 lands. Use the scene-director subagent for heavy 3D work and /scene-check before every /ship. Same git, test and comment rules as the lead.
```

## Prompt 3: minds lane (Lane C), start after P-003 is merged
Open a third terminal: `claude --worktree minds`
```
You are Lane C (minds) on Twinlight. Read CLAUDE.md, docs/MODEL.md sections 7 and 8, docs/ARCHITECTURE.md sections 2, 5 and 6, and D-009 and D-011 to D-014.

You own services/minds and tools/citybake. Do not edit PLAN.md, MEMORY.md or DECISIONS.md; write progress and questions to docs/lanes/minds.md. City pack format changes go to Lane A.

Start with M2 (P-020 to P-024) for Ahmedabad. Use the persona-lab subagent and test-author for [tests] points. Same git, test and comment rules as the lead.
```

## When the M1 demo works
Record 20 seconds of `/dev/fixture` playing a launch and add it to the Demo log in MEMORY.md. That is the first thing worth showing anyone.
