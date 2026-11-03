# Design

Everything here is reference for the real build. None of it is production code.

| Path | What it is | Use it for |
| --- | --- | --- |
| `tokens.json` | Colours, type, spacing, motion, radii | Source of truth for `packages/ui` (P-007) |
| `prototype/twinlight-city.html` | The City screen as one runnable file: three.js WebGL scene, intro camera move, timeline, layers, store and resident cards, rain, phone layout | The look and feel the stage must match (P-016 onward). Open it in a browser; add `#still-day6` or `#still-day13` for fixed frames |
| `prototype/toy-sim.mjs` | The prototype's procedural Ahmedabad and toy simulation, pulled out as a plain module (48,000 residents, 14 dark stores, 30 days, seed 49) | The reference the Python engine ports in P-010 to P-013, and the numbers its parity tests check |
| `renders/` | Stills from the prototype: days 1, 6, 10 and 13, and phone frames on days 6 and 13 | README hero, screen mocks, visual comparison |
| `screens/snapshots/` | One image per designed screen: Landing, Brief, City, Plan, phone City, phone Resident, phone Interview | Layout, hierarchy and copy for the app shell (M2, M4, M5) |
| `screens/source/` | HTML source of those screens from the design canvas | Exact copy, structure and inline mock data. They need the canvas runtime, so read them rather than run them |

Live links: City prototype https://claude.ai/artifact/Prcx9YEATfKYn8BV8cB3er and the clickable design canvas https://claude.ai/artifact/SmMk8QYstNGDafLSCKCubS

## What the toy sim produces

`node design/prototype/toy-sim.mjs` prints this (48,000 residents):

| Day | Heard about it | Ordered at least once | Stores out of stock |
| --- | --- | --- | --- |
| 1 | 13.4% | 0.5% | 0 of 14 |
| 6 | 23.0% | 4.5% | 0 of 14 |
| 10 | 36.5% | 8.0% | 3 of 14 |
| 13 | 41.7% | 10.8% | 0 of 14 |
| 20 | 51.9% | 16.1% | 0 of 14 |
| 30 | 63.1% | 23.0% | 0 of 14 |

The screen mocks use larger, weighted numbers (each agent stands for about 16 real shoppers in a city of 16 lakh), so the City and Plan screens show figures like 61,000 people ordered rather than these percentages.

## How the prototype and the real build differ

| | Prototype | Real build |
| --- | --- | --- |
| Simulation | JavaScript, in the browser, one seed, hourly ticks | Python engine (NumPy + Numba), 50 seeds, 15-minute ticks plus a logistics event queue, calibrated parameters |
| City | Procedural Ahmedabad | Baked from OpenStreetMap and population data by `tools/citybake` |
| Rendering | WebGL with three.js r169, one file | `packages/stage`: WebGPU with a WebGL2 fallback, TSL materials, quality tiers |
| Data to the GPU | Event times computed in the page | The same idea, read from tape files streamed from the runner |
| Numbers | One run, no intervals | Every number has an interval across seeds |
