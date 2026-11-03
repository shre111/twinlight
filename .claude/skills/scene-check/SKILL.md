---
name: scene-check
description: Visual and performance check for anything rendered in the city or visible in the web app. Takes Playwright stills at fixed sim times on desktop and phone, compares with baselines, reads the frame-time report, and critiques the result against docs/DESIGN.md.
argument-hint: "[route or scene, default /dev/fixture]"
---

Check the scene at: $ARGUMENTS (default `/dev/fixture`).

1. Make sure the web app is running (`just dev` or `pnpm -F web dev`) and the fixture tape is available.
2. Run `just shots`. It uses Playwright with Chrome and, in headless environments, the flags `--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist`. It captures:
   - desktop 1440x900 and phone 390x844
   - `#still-day1`, `#still-day6`, `#still-day13` (fixed camera, UI visible) and `#still` (no UI)
   - a resident card open and a store card open
   and writes them to `artifacts/shots/<branch>/`, plus a diff against `artifacts/shots/baseline/` and `frame-report.json` (p50/p95 frame time per tier from the stage's perf HUD).
3. Open every new screenshot with the Read tool and look at it. Judge it against `docs/DESIGN.md` and `design/prototype/twinlight-city.html`:
   - Does the city read as a dense, beautiful night city? Is the warming from cool to amber legible across the three days?
   - Are individual lights distinguishable (bloom not blowing out districts)?
   - Is the UI quiet, aligned, legible, using tokens, not overlapping on phone? Focus states visible?
   - Any z-fighting, popping, banding, clipped labels, layout shift?
4. Check `frame-report.json` against ARCHITECTURE.md section 9. Swiftshader timings are not real GPU numbers; compare them to the baseline run in the same environment, and flag regressions over 15%.
5. Report a verdict: pass, or a short list of concrete fixes. Fix them, re-run, and only then report pass.
6. When a visual change is intended, update the baseline with `just shots --update` in the same PR and say so in the PR body.
