# Lane B (stage)

Worktree `.claude/worktrees/stage`. Lead folds this into MEMORY.md at each wrap.

## P-018
- Done with this PR (#58, #59, #65 and readouts). Keyboard: Space, arrows, page keys, Home, End, visible focus. 390 px: two figures, single speed button, 44 px targets. Checked in Chrome and in just shots; real touch and Firefox not checked.
- Readouts use the type scale, 32 and 24 px values with 12 px labels; the design's 36, 28 and 11 px are not on it.
- Figures show the last finished tick, so they are never ahead of the city. Stores out is counted from intervals at the exact minute, so it always matches the beacons.
- Numbers are one run's, with no interval yet: ranges and weighted people arrive with P-053.
- Firefox keeps the scrubber's own focus ring; it is the last browser for P-015 too, pending the user's call on the 100 MB download.

## For P-040 (D-028)
- Add chatter with `kind` to `StageTape` and the synthetic tape, and draw nudged rows fainter.
- Do not assume time order: nudged rows can land up to 72 h late, across chunks.

## Gotchas
- `just shots` times out at load 13 and above, because stills render on the CPU. It passes on the same code once load falls, so check load before suspecting a change.
- Software-render frame times are noisy under load. Judge frame cost on the M4: 100k residents at 16x with the timeline holds p50 16.7, p95 18 ms.
- The Next dev badge covers part of the play button. It is dev only, but a click there misses.
