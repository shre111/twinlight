# Twinlight design

## The idea
An Indian city at dusk. At the start of a launch every home glows a dim, cool tubelight. As people hear about the product their homes tint lilac; when they order, their window turns sodium-lamp amber with a brief flare. Over 30 days the city visibly warms, neighbourhood by neighbourhood. That warming is the product's signature image and the one place the design spends its boldness. Everything around it (panels, type, controls) stays quiet, precise and docked to the edges.

Light is data. Every colour in the city means something, and the legend says what.

## Tokens (source of truth: `design/tokens.json`)
| Token | Hex | Role |
| --- | --- | --- |
| `dusk` | #0F1330 | ground, page background, fog |
| `dusk-2` | #171C42 | panels (86% opacity, 10 px backdrop blur over the city) |
| `haze` | #2A3166 | hairlines, inactive tracks, road lines |
| `mist` | #9AA3CF | secondary text |
| `paper` | #EEF0FB | primary text |
| `lamp` | #FFB547 | residents who ordered; primary action fill (text on it uses `dusk`) |
| `tube` | #A6F0DA | dark stores, riders, the unaware city (at low intensity) |
| `chatter` | #C8B6FF | word of mouth: residents who heard, arcs |
| `sindoor` | #FF5468 | stockouts, errors |
| `monsoon` | #6F8CC9 | rain, weather |

Sky shades are environment, not data, and exist only for the stage (TS tokens `sky.*`, no CSS variables): `zenith` #04061A overhead, `mid` #0B0F2E above the horizon, `horizon` #231F4E for the distant horizon glow, `band` #4A3C7E for the thin violet band. Fog is `dusk`; only its farthest reach drifts toward `sky.horizon`. The data colours above stay reserved for meaning, and blends of them did not reach these darker violets, so the sky has tokens of its own.

The app commits to the dusk look; there is no light theme for the city. Marketing pages and the Plan view's printable export may use a light variant (`paper` ground, `dusk` text) defined in tokens.json under `light`.

Contrast: `paper` on `dusk` 15:1, `mist` on `dusk` 7.4:1, `dusk` on `lamp` 10:1. Keep all text at 4.5:1 or better against the panel it sits on, including over the blurred city.

## Type
- Display: **Big Shoulders** (300 and 500). Wordmark, big numbers, card titles, the brief sentence. Its tall, narrow forms echo a skyline; use it large and light, never for running text.
- Body: **Atkinson Hyperlegible Next** (400, 500, 700). Everything else, including data labels. `font-variant-numeric: tabular-nums` wherever digits line up.
- Scale (px): 12, 14, 16, 18, 21, 24, 32, 44, 60, 80. Body 16/24. Labels 13/18 in `mist`.
- Sentence case everywhere. No all-caps labels, no monospace labels, no "A · B" strings, no arrow glyphs appended to buttons, no emoji.

## Shape and depth
- Radii by role: floating cards 14 px, inputs 10 px, chips and pills 999 px, anything docked to an edge 0.
- Shadow only on floating cards: `0 12px 40px rgba(5, 7, 20, .45)`.
- Hairline borders `1px solid haze`. No left-border accent cards, no decorative gradients. The only gradients are physical: the sky, beam falloff, fog.

## Layout
Desktop (1280 and up): the city fills the window.
```
+--------------------------------------------------------------+
| wordmark  [scenario chips]                 [readouts card]   |
| [layer pills]                                                |
|                                                              |
|                    (city, unobstructed)          [card]      |
|                                                              |
| [legend]                                                     |
|==============================================================|
| > 4x  Day 6, 7:40 pm  |--ad----creator----[monsoon]----|  |
+--------------------------------------------------------------+
```
Phone (390): wordmark + product chip on top, two readouts, the city, a compact timeline; cards and the legend open as bottom sheets. Nothing overlaps the timeline.

## Screens
1. **Landing.** The live city fills the first screen. Headline in Big Shoulders 300 at 80 px: "Watch your launch before it happens." One line under it: "Twinlight builds a living copy of a city, lets your product loose for 30 days, and shows who hears about it, who orders, and which stores run dry." One action: "Rehearse a launch". Below the fold: three short sections answering real doubts (How is this different from a survey? Where do the people come from? How wrong can it be?), the honesty meter explained, and a public demo link.
2. **Brief.** A sentence you complete, set in Big Shoulders 44: "I'm launching [Mango lassi, 200 ml] at [₹49] in [Ahmedabad] on [Blinkit, Zepto] with [₹2,00,000] on [Instagram, creators] from [12 Nov]." Each bracket is a real input with a dropdown or editor. Under it, what will happen ("We'll build Ahmedabad's quick-commerce shoppers as 100,000 residents, each standing in for about 16 people, ask 300 of them what they think, and run your launch 50 times") and the action "Build the city".
3. **City.** The main view (see prototype). Readouts, layer pills, legend, timeline, cards.
4. **Resident card and interview.** Name, age, district, the resident's month as a short vertical story with day markers, then "Interview by voice" opening a calm sheet: a soft waveform, live transcript, replies with small linked event chips ("Day 10, out of stock").
5. **Store card.** Name and district, stock over 30 days as a sparkline with red spans where it was empty, orders served, the recommended reorder point with its range.
6. **Plan.** Narration at the top (each sentence links to a moment in the city), the adoption fan chart (P10 to P90 band, P50 line in `lamp`), the stock plan table per store, next-city ranking with ranges, and the calibration panel ("Calibrated against your Diwali 2025 launch: 80% ranges held 9 times in 10" or "Not yet calibrated against your real sales").
7. **Compare.** Two cities side by side, same camera, one timeline. The difference is visible as one city warming faster.

## Motion
- One orchestrated moment per screen. City view: on load the camera eases down from altitude while districts light up in a wave (2.5 s). Brief: when "Build the city" is pressed, buildings rise from the river outward.
- In-world motion is the content: arcs travel, riders move, flares bloom. UI motion only answers an action (a card slides in from its edge in 220 ms, a sheet rises in 260 ms, ease-out).
- Reduced motion: no flights, cuts instead; flares become a static amber; arcs appear without travelling; auto-rotate off.

## The city, layer by layer (stage)
- **Sky and fog:** deep indigo overhead, a thin muted violet band at the horizon; exponential fog in `dusk`.
- **Ground and roads:** ground slightly lighter than the sky's zenith; roads as fine `haze` lines, arterials and the ring road a touch brighter.
- **River:** a dark, slightly glossy ribbon with a slow sheen; bridges as fine lines.
- **Buildings:** dark indigo boxes or extrusions, roofs a shade lighter so the skyline reads; heights from the pack.
- **Residents:** soft round points inside buildings at floor heights; cool `tube` at 20% intensity, `chatter` at 50% when aware, `lamp` at 90% when ordered with a 6 sim-hour flare.
- **Arcs:** lifted curves between homes in `chatter`; a bright head with a fading tail; at most a few hundred visible at once.
- **Stores:** thin vertical beams of `tube` with a faint ground ring for the delivery radius; `sindoor` and a slow flicker while out of stock.
- **Riders:** small `tube` points with short trails along the road network.
- **Rain:** `monsoon` streaks, darker fog, sheen on roads and river.
- **Bloom:** subtle; individual lights must stay distinguishable. If a district turns into a white blob, bloom is too strong.

## Copy voice
Plain, specific, calm. Name things the way a brand manager would: "orders", "stores out of stock", "heard about it", not "conversions" or "inventory depletion events". Ranges are stated as ranges ("between 9,400 and 12,100 people likely order in 30 days"). The product rehearses; it never predicts. Errors say what happened and what to do. Empty states invite the next action.

## Accessibility
- Every 3D insight exists as text or a table (Plan view, cards, readouts).
- Real buttons, inputs and labels; visible focus rings (`lamp`, 2 px, offset 2 px).
- Colour is never the only signal: stockouts also flicker and are listed; layer pills carry text labels.
- Targets at least 44 px on touch.
- `prefers-reduced-motion` respected as above.

## Performance tiers (see ARCHITECTURE.md section 9)
`high` 100k residents with full bloom; `medium` 60k, half-resolution bloom; `low` 25k, no bloom, no rain particles. Auto-selected from the intro's frame timings and adjustable in settings.

## Shader rules learned from the prototype
- Never call `pow()` with a base that can go negative. GLSL leaves it undefined; real GPUs return NaN, and bloom smears NaN pixels into large flickering black blocks (software renderers in CI usually hide this). Clamp the base, or use `x * x` and `exp(-x * x)` for falloffs.
- Keep a sanitize pass before bloom that turns NaN or Inf pixels black and caps HDR values, so one bad fragment cannot take over the frame.
- Test the stage on at least one real GPU (a laptop with integrated graphics is enough) before merging shader work; headless stills alone do not catch this.

## References
- `design/prototype/twinlight-city.html`: live prototype of the City screen (single file, three.js WebGL, deterministic toy sim of 48,000 residents). Visual reference for the stage; match its look, not its structure. Hosted copy: https://claude.ai/artifact/Prcx9YEATfKYn8BV8cB3er
- `design/renders/`: renders from the prototype on days 1, 6, 10 and 13, plus phone frames on days 6 and 13.
- `design/screens/snapshots/`: one image per designed screen (Landing, Brief, City, Plan, phone City, Resident, Interview). Look at these first for layout, copy and hierarchy.
- `design/screens/source/`: the HTML source of those screens from the design canvas (markup, copy, inline data). They need the canvas runtime to render, so read them for exact copy and structure, not to run. Live, clickable canvas: https://claude.ai/artifact/SmMk8QYstNGDafLSCKCubS
- `design/prototype/toy-sim.mjs`: the prototype's toy simulation as a plain module; the Python engine ports it in P-010 to P-013.
- Prototype sim at day 30: 63% heard, 23% ordered (west 33 to 34%, east 12%); five stores ran out, at most four at once, around days 9 to 12.
