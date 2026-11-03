# Lane B (stage)

Worktree `.claude/worktrees/stage`. Lead folds this into MEMORY.md at each wrap.

## Now
- P-015: engine check done for Chrome (WebGPU, 16.7 ms p50) and Playwright WebKit (WebGPU, 17.0 ms p50). Firefox is not checked: the owner approved the WebKit download only.
- P-016: sky tokens (#19), sky, ground, fog and camera (#24), then roads and instanced buildings (this PR). On the M4 with WebGPU the full city holds 16.7 ms p50 and 17.6 ms p95 at DPR 1.75 in 7 draw calls. Next: residents layer, then the dev scrubber with day stills.
- Then the toy-pack loader against `data/cache/toy-5k` and `toy-48k` (the lead keeps P-010 unticked until it renders a generated pack).

## For P-045 (post-processing)
- No building or ground shader has a per-fragment NaN guard. The sanitize pass before bloom (DESIGN.md shader rules) must exist before bloom lands. An `x == x` check per fragment is not reliable under WGSL's NaN rules.
- Dark gradients (ground glow, sky) show faint 8-bit banding on the real GPU. Add dither in the output pass.

## Gotchas
- TSL: the renderer applies its linear-to-sRGB output transform only when `material.fragmentNode` is null. Use `colorNode` and `opacityNode`, never `fragmentNode`, or colours come out linear.
- Mix shades on the sRGB values (`new Color().setStyle(hex, NoColorSpace)`, then `convertSRGBToLinear()`). Scaling in linear space by k looks like sRGB scaling by about k^(1/2.2). The colour guard also matches hex inside comments.
- `renderer.info.render.drawCalls` reports 1 on an empty WebGL 2 scene; subtract it when counting layers.
- `three/webgpu` and `three/tsl` load in Node, so vitest can test matrices and layer construction without a browser.
- Under SwiftShader the 7k-building city runs about 100 to 117 ms p50 at 1440x900, so the frame report samples 60 frames rather than 3 seconds. `window.__twinlightStage` appears only after `loadCity` resolves.
- `POSES.phone` looks down steeply enough to hide the horizon. The phone renders in `design/renders` are still mode at the `hero` pose.
- `git rm` stages a deletion that the next `git add <paths> && git commit` sweeps in; commit deletions in the commit that owns them.
