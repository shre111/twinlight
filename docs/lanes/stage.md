# Lane B (stage)

Worktree `.claude/worktrees/stage`. Lead folds this into MEMORY.md at each wrap.

## Now
- P-016: residents layer merged (#31). This PR wires the synthetic tape into `/dev/stage` (`?day=1|6|13` at the prototype's still hours, `?t=<minutes>`, default day 6) and replaces the single city still with day 1, 6 and 13 stills. P-016 AC on the M4 with WebGPU: 20k residents at 16.7 ms p50 and 18.2 to 18.6 ms p95 at DPR 1.75 in 8 draw calls; 100k at 16.7 and 18.5; 200 random `setTime` calls at most 0.3 ms each. The empty stage's p95 is 17.3 to 17.6 ms, so the p95 above 16.7 is mostly rAF jitter.
- Next: the toy-pack loader (P-010 waits on it), then P-017 stores and P-018 timeline and readouts.

## Gotchas
- WebGPU draws `Points` at 1 px. Residents are one `Sprite` with `count = n` over a quad carrying `InstancedBufferAttribute`s, with a `PointsNodeMaterial` (`sizeAttenuation = false`; `sizeNode` in logical px, `viewportSize` in device px).
- Event times upload as float32: NEVER rounds to 2^32, so `time - at` stays negative and every ramp and the flare stay at zero.
- A steady per-point halo looked right at 20k but rimmed buildings at 48k and blobbed districts at 100k. Halos are flare-only until P-045's bloom replaces them (remove the halo then, or flares double-bloom).
- Camera-facing sprites on building faces need a depth nudge toward the camera by their own radius, or the wall they sit on cuts them into crescents. The nudge is capped at 60 m so far flares cannot jump in front of buildings.
- SwiftShader costs about 10 µs per instance per frame and about 2 s per screenshot with 20k residents. The shots allow 15 s per still and mark the frame report slow; a full `just shots` takes about 3.5 minutes. Judge resident performance on a real GPU.
