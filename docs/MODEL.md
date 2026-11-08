# Twinlight simulation model

The engine is a hybrid: a time-stepped agent-based model of people (awareness, word of mouth, purchase decisions) coupled to a discrete-event model of supply (dark-store stock, picking, riders, restocks). LLM personas never run inside it. They inform the parameter priors before a run (via calibration) and explain outcomes after it (interviews, narration).

Everything here is v1. Changing any formula is a decision (`/decide`) plus updated goldens.

## 1. Time
- Sim clock: integer minutes from launch, `0..=43_200` (30 days).
- Behaviour tick: 15 minutes (2,880 ticks).
- Logistics: a min-heap event queue keyed by `(minute, seq)` processed inside each tick window, so events within a tick are ordered deterministically by `seq`.
- Time-of-day profiles are piecewise-constant per 15-minute slot.

## 2. Entities
**Resident** `r` (from the city pack): home building, H3 cell, district, segment `s(r)`, persona reference, and a weight `m_r` (people represented, set by the city pack; a metro pack of 100k agents standing in for its quick-commerce shoppers gives about 16). Counts shown in the product are weighted sums; the city renders one light per agent. Per-run parameters drawn at run start from the calibrated posterior of its segment (one draw per seed per segment, then per-resident jitter):
| Symbol | Meaning |
| --- | --- |
| `a_r` | product affinity (logit scale) |
| `beta_r` | price sensitivity |
| `kappa_r,c` | responsiveness to channel `c` |
| `q_r` | propensity to talk about products they bought |
| `rho_r` | repeat-purchase rate after first order |
| `tau_r` | awareness half-life (days) |
| `omega_r(t)` | snack/consumption occasion intensity by time of day and day type (segment profile, working day or weekend; D-036) |

**Tie** `(u, v)` from the social graph: kind (household, neighbour, friend, work), strength `w_uv` in (0, 1].

**Store** `k`: position, delivery radius, initial stock per SKU, reorder policy `(s_k, S_k)`, review period, lead time distribution, picking rate `mu_k` (orders per hour), rider shifts.

**Campaign** `c`: channel (one of `social`, `video`, `creators`, `in_app`, `outdoor`; D-034), schedule, budget per day, CPM, targeting weights over segments and districts, frequency cap.

**Weather**: rain intensity field `rain(t, hex)` from the scenario's event list.

## 3. Awareness
Each resident holds awareness strength `A_r >= 0`. Per tick:
```
A_r <- A_r * 2^(-dt / tau_r)  +  paid_r  +  wom_r  +  seen_r
aware_at_r = first tick where A_r >= theta_aware   (theta_aware = 1.0)
```
**Paid exposure.** Campaign impressions this tick `I_c = budget_c(t) / CPM_c * 1000`. Impressions are allocated to residents with probability proportional to `target_c(s(r), district(r)) * presence_c(s(r), slot)`, capped at `freq_cap_c` per day for each campaign, sampled with the campaign's stream. Each impression adds `kappa_r,c * 0.6`. Channels, their default CPMs and how presence combines are in section 11 (D-034).

**Word of mouth.** A resident who ordered at time `t_o` talks to each tie with hazard
```
lambda_uv(t) = q_u * w_uv * talk_kind(kind_uv) * (talk_floor + (1 - talk_floor) * exp(-(t - t_o) / 72h))
```
A successful talk adds `talk_gain * w_uv` to `A_v` (`talk_gain = 1.0`, `talk_floor = 0.35`; D-029). Aware non-adopters talk at `0.15 * q_u * w_uv * talk_kind * exp(-(t - aware_at) / 72h)` (hearsay, with no floor). The tape records the talks that mattered (D-028): the talk that takes `A_v` over `theta_aware` (`heard`), and the last talk a resident received in the 72 hours before their first order when that is a different one (`nudged`). Those chatter events are what the stage draws as arcs.

**Seen nearby.** Riders delivering in a hex add a small exposure `0.05` to residents in that hex with probability `0.02` per delivery (brand bags, word on the street). Kept small; exists so dense areas get a mild density effect.

## 4. Purchase decisions
Occasions arrive per resident as a Poisson process with rate `omega_r(slot) * (1 + 0.12 * evening_rain(t))`. At an occasion, an aware resident who has not ordered decides:
```
U = alpha_s + a_r + gamma * log(1 + A_r) - beta_r * price / ref_price_s + delta * avail_r(t) + eps
P(order) = sigmoid(U)
```
- `avail_r(t)` is 1 if a store within radius has stock and quoted ETA <= promise (10 to 20 minutes), else 0.
- `eps` is logistic noise from the resident stream.
- After the first order, repeat occasions use `P = sigmoid(alpha_rep_s + rho_r + satisfaction_r - beta_r * price / ref_price_s)` where `satisfaction_r` drops by 0.8 for every stockout or late delivery they experienced.

**Out of stock.** If the chosen (nearest) store has no stock: with probability `p_spill` the order goes to the next store within radius that has stock (spillover, longer ETA); else with `p_wait` the resident retries at their next occasion after 24 h; else the sale is lost and `satisfaction_r` drops.

**Bass limit (a required test).** With a fully mixed graph, no supply limits, constant budget and homogeneous parameters, cumulative first orders must follow the Bass curve `F(t) = (1 - e^{-(p+q)t}) / (1 + (q/p) e^{-(p+q)t})`, where `p` and `q` are derived from the paid and word-of-mouth terms. This anchors the model to a known closed form. D-027 fixes the reduced configuration and the tolerance: the mean of 8 seeds within 1 percentage point at the end of every day, plus a `q = 0` run within 0.5 points of `1 - exp(-p t)`.

## 5. Supply
- **Inventory.** Per store and SKU. Continuous review every `review_k` minutes: if `I <= s_k`, order `S_k - I - in_transit` arriving after lead time `L ~ Gamma(k_L, theta_L)` minutes, rounded to 15.
- **Picking.** Orders queue at the store; service time exponential with rate `mu_k`; picking stops when stock is 0 (orders already queued complete).
- **Riders.** Pool per store per shift. A picked order waits for a free rider. Travel minutes from `travel.bin` (store -> hex, time-of-day band) times `1 + 0.35 * rain(t, hex)` times lognormal noise (sigma 0.15). Rider returns after `2 * travel`.
- **Promise.** ETA = queue wait + picking + rider wait + travel. Orders with ETA over the promise are flagged late.

## 6. Scenario events
Typed modifiers with time windows: `rain {intensity, hexes|all}`, `festival {demand_multiplier, segments}`, `match_night {slot boost}`, `price_change {sku, price}`, `campaign {…}`, `store_closed {store}`. Each one only touches the parameters listed in its type, so effects are traceable in the narration.

## 7. Monte Carlo and uncertainty
- A run has N seeds. Seed `i` draws one posterior sample of segment parameters (parameter uncertainty) and runs the stochastic sim (process noise).
- Bands are computed over seeds: P10, P50, P90 for cumulative adopters by day, orders per store per day, stockout minutes, lost sales.
- **Stock plan.** For store `k` and service level `SL` (default 95%): recommended reorder point = `quantile_SL(demand during lead time + review period)` across seeds; recommended order-up-to = reorder point + `quantile_0.5(demand during review period)`.
- **Next-city ranking** (multi-city briefs): P50 adopters per 10k residents at day 30, with P10 to P90 shown, ordered by P50 and flagged where intervals overlap heavily.

## 8. Calibration (services/minds)
1. **Deep personas.** ~300 residents sampled stratified by segment from the persona pool. Each answers a fixed instrument through the LLM: purchase intent (5-point), price ladder (Gabor-Granger at 4 to 6 price points), expected frequency, likelihood to mention it to named tie kinds, channel habits. Answers are structured JSON validated by schema.
2. **Mapping.** Stated answers are converted into behaviour parameters with a learned mapping that includes an *intent deflator* (stated intent overstates behaviour). The deflator starts from a conservative prior and is updated by backtests.
3. **Hierarchical model** (NumPyro): segment-level parameters with partial pooling across segments; likelihood from persona answers; optional likelihood from a backtest dataset. Output: 1,000 posterior draws per segment parameter, stored as Parquet.
4. **Backtests (honesty meter).** Given a brand's past launch (daily units by city), fit on days 1 to 10, forecast days 11 to 30, report 80% interval coverage and MAPE. Shown on every Plan view. Without a backtest the Plan view says the run is uncalibrated against real sales.

## 9. Outputs per seed (tape) and per run (plan)
Per seed: see ARCHITECTURE.md section 4. Per run: bands, stock plan, ranking, attribution of first orders by source (paid, word of mouth, seen nearby), lost sales, stockout minutes, delivery-time distribution, calibration diagnostics, narration with event links.

## 10. Required tests (engine)
- Determinism: same `(city_pack_hash, scenario_hash, params_hash, seed)` -> identical tape blake3 on the pinned CI platform, across 1, 4 and 8 worker processes. Other platforms: probe values exact, aggregates within 1e-9 relative.
- Bass limit (section 4).
- Conservation: deliveries + lost + pending == orders; stock never negative; stock out of a store == picks from it.
- Monotonicity (hypothesis property tests): raising price never raises P50 adopters in a no-supply-limit scenario; raising budget never lowers awareness.
- Event isolation: a rain event outside its window changes nothing (tape hash equal to the no-rain run until the window starts).

## 11. v1 parameters and pinned readings (D-027)
Every constant below is a field of the run parameters with this default, so tests and calibration can override it. Segment values are listed for the toy city's five age bands; real segments (D-026) get theirs from calibration.

**Readings the formulas left open**
- Rates are per hour. A tick is `dt = 0.25 h`; an event with rate `r` happens in a tick with probability `1 - exp(-r dt)`, and where several can happen in a tick (impressions, talks) the count is `Poisson(r dt)`.
- Tick order: decay `A`; paid impressions; word of mouth (talkers and their rates taken at the start of the tick); awareness crossings; occasions and orders.
- Paid: impressions in a tick are `Poisson(I_c(t))`, where the day's impressions (`budget / CPM * 1000`, divided by the mean resident weight so they are in agent units) are spread over the day's ticks by the channel's presence profile. Each lands on a resident with probability proportional to its targeting weight, up to the frequency cap per day.
- A resident is in the decision pool from `aware_at` onward, even after `A_r` decays; the current `A_r` enters `U`.
- `P(order) = sigmoid(U)` is the probability itself; `eps` is the logistic noise it integrates over, so the engine draws one uniform per decision.
- `t_o` is the resident's most recent order, so repeat orders renew talk.
- Crossing: within a tick, contributions apply in the order paid, then talks by (minute, talker). The first to bring `A_r` to `theta_aware` sets `aware_at`, `first_source` and, for a talk, `heard_from`.
- With `wom_mixing = "uniform"` (the Bass configuration) a talker emits talks at rate `q_u` to residents drawn uniformly, with strength and kind factor 1.
- Impressions that land on a resident already at the frequency cap are lost, not re-landed. A paid crossing gets a uniform minute inside its tick.
- An order's minute is uniform between the latest of (tick start, `aware_at`, the last talk received) and the tick's end, so `aware_at <= first_order_at` and a nudged talk never follows the order it nudged. With `order_on_awareness`, `first_order_at = aware_at`.
- The nudge window is a fixed 72 hours whatever the talk decay time, and a nudged talk can predate `aware_at` (a talk that came before a paid crossing).
- Channels (D-034): the gain per impression is `0.6 * kappa_s * channel_kappa[c][s]`, with `channel_kappa` 1 until calibrated. Presence for segment `s` in hour `h` is the campaign's own hourly weight (its dayparting, flat when empty) times the channel's presence `channel_presence[c][s][h]`, which is 1 until calibrated; for `in_app` the channel's presence is the resident's occasion profile, so an in-app ad is seen when people open the app. A day's impressions are spread over its active ticks and its residents in proportion to targeting times presence: each tick gets its share of the day's summed targeting times presence, and within the tick the impressions are split over segments by summed targeting times presence, then land by targeting within the segment. When presence is the same for every segment this reduces to the v1 rule: presence sets each tick's share and targeting alone picks the resident. A day whose presence is zero in every active hour buys nothing. An `outdoor` campaign must have district weights.
- A resident who becomes aware during a tick gets that tick's full occasion probability (up to 15 minutes more exposure than strictly due). Occasions are drawn as one binomial count over the aware pool when every segment shares the tick's rate, and otherwise as one binomial count per segment over its aware residents; both are exact while a rate is shared within a segment (rain by cell, P-034, will need more groups).
- `order_on_awareness` is for the reduced Bass configuration, which runs with occasions off; with occasions on, a resident could place a first and a repeat order in one tick.
- Until P-013: an order is counted at its occasion, `avail = 1`, `satisfaction = 0`, and the DELIVERIES section stays empty.

**Default CPMs** (D-034; rupees per 1,000 impressions before GST, used when a campaign gives no `cpm`; researched November 2026 at Rs 95.25 per USD, the RBI reference rate of 30 October 2026)
| Channel | Default | Range | Confidence | Source |
| --- | --- | --- | --- | --- |
| `social` | 130 | 25 to 290 | medium | India Meta CPM of USD 1.36 (Lebesgue, 2026 ecommerce benchmarks, lebesgue.io/facebook-ads/facebook-cpm-by-country) and USD 1.35 median over July 2025 to July 2026 (SuperAds, superads.ai/facebook-ads-costs/cpm-cost-per-mille/india); the government's lowest-bid rate of Rs 15 to 25 (CBC, February 2024) is the floor |
| `video` | 110 | 70 to 480 | low | YouTube skippable at Rs 0.11 per impression (The Media Ant listing, 2026, themediaant.com/digital/youtube-advertising); range from the CBC OTT lowest-bid rate (Rs 70, May 2026) to JioStar's IPL premium video (Rs 480, 2025) |
| `creators` | 2,500 per 1,000 Reel views | 300 to 5,000 | low | Micro-creator fee of Rs 8,000 to 80,000 per Reel (Grynow, August 2026, grynow.in/blog/influencer-marketing-cost-in-india.html) over typical views from the HypeAuditor views calculator (global medians): about 10,000 views, the mean of the midpoints of its 10k to 50k follower band (1,200 to 8,100) and 50k to 100k band (3,900 to 26,800); the geometric-mean fee of Rs 25,300 over 10,000 views gives Rs 2,530 |
| `in_app` | 125 | 40 to 250 | low | Instamart banner at Rs 0.127 per impression, rounded (The Media Ant listing, 2026); the Zepto masthead lists at Rs 0.09 to 0.10. No app publishes CPMs, and sponsored listings are sold per click |
| `outdoor` | 40 | 10 to 100 | low | Ahmedabad (Bopal) hoarding at Rs 2,200 a day for 30 days against an estimated 1.7 million impressions a month, Rs 38.8, plus printing and mounting (The Media Ant, themediaant.com/outdoor/hoarding-ahmedabad-80330) |

Impressions are not like for like across channels: platforms count ads served, creators count Reel views, outdoor counts estimated chances to see. Until calibration sets `channel_kappa`, an impression on any channel adds the same awareness, so channel comparisons are uncalibrated (D-034).

**Defaults**
| Parameter | Default |
| --- | --- |
| `theta_aware` | 1.0 |
| gain per impression | `0.6 * kappa` |
| gain per talk | `talk_gain * w_uv`, `talk_gain = 1.0` (D-029) |
| talk floor for adopters | 0.35 (D-029) |
| hearsay factor | 0.15 |
| talk decay time | 72 h |
| `tau` (awareness half-life) | 4 days |
| `kappa` by age band | 1.0, 0.9, 0.62, 0.38, 0.2 (the prototype's ad responsiveness) |
| `q` (talk rate per unit tie strength) | 0.02 per hour |
| `talk_kind` | household 1.0, neighbour 0.6, friend 1.0, work 0.5 |
| frequency cap | 3 impressions per resident per day and campaign |
| `omega` | 3 occasions per day, shaped by the prototype's hourly profile (peak 19:00 to 22:00) |
| `alpha_s` | `alpha_0 + ln(1.35, 1.2, 0.9, 0.6, 0.38)` by age band (the prototype's affinity), `alpha_0 = -3.0` |
| `a_r` | Normal(0, 0.3) around the segment value |
| `gamma` | 2.5 |
| `beta` | 1.0, with `ref_price_s` equal to the launch price unless the scenario says otherwise |
| `delta` | 1.0 |
| `alpha_rep_s`, `rho_r` | -2.3; Normal(0, 0.3) |
| evening rain boost | 0.12 (events arrive with P-034) |

The prototype-parity fixture fits three of these (a paid reach multiplier, `q` and `alpha_0`) and checks them in next to the scenario.

## 12. v1 supply parameters and pinned readings (P-013)
Supply is on when the scenario carries a supply block. Without one the engine keeps the no-supply-limit behaviour the Bass and parity tests use: every aware resident can order, an order is counted at its occasion, `avail = 1`, `satisfaction = 0`.

**Readings**
- Units: stock and orders are in agent orders (one resident placing one order); a real pack divides brand quantities by the mean resident weight.
- Serving store: a resident's own store is the nearest store whose radius contains them. A resident inside no radius cannot order while supply is on (they can still hear about the product). A store's catchment is the residents it is the own store of.
- Stock is taken when the order is placed, so queued orders always complete and stock never goes negative. A store is out of stock while its stock is below 1; the interval starts at the minute of the order that took the last unit and ends at the minute a restock arrives.
- Decision: `avail_r = 1` when the own store has stock at the start of the tick and its quoted ETA (wait for the picker, mean picking time, wait for a rider, straight-line travel without noise) is within the promise; otherwise 0. A resident can still decide to order with `avail = 0`.
- Out of stock at the own store when the order is placed: one uniform draw. Below `p_spill`, and the next nearest store within `spill_radius_factor` times its radius has stock, the order is served from there (spilled). Otherwise, below `p_spill + p_wait`, the resident waits: at their first occasion at least 24 hours later they try again without a new decision. Otherwise the sale is lost. Every such hit, whatever its outcome, lowers `satisfaction_r` by 0.8 and sets the stockout flag; a spill and a loss set their own flags.
- Within a tick, restock arrivals and order attempts are processed in minute order, arrivals first on a tie. A restock that sells out in the minute it arrives leaves the store out of stock again at once, so the two intervals merge into one and that order sits inside it.
- Restock: every `review_minutes` each store with `stock <= s_k` orders `S_k - stock - in_transit`, arriving after a fixed lead time (v0; the Gamma lead time arrives with P-032). In a minute where several things fall together the order is arrivals, then the review, then order attempts. A store never reorders when that quantity is not positive, so an order-up-to level of zero means no restocking.
- A store whose initial stock is below 1 (for example one with an empty catchment) is out of stock from minute 0 and is logged and counted as such.
- `avail` counts a restock that arrives exactly at the start of the tick, not one arriving later in it.
- The spill store is the nearest other store whose extended radius contains the resident, which can differ from the second-nearest store. A draw below `p_spill` when that store is empty or absent falls through to waiting.
- `satisfaction_r` has no floor and applies to residents who have not ordered yet too, but it only enters the repeat-order utility.
- Picking: one picker per store, first come first served; each pick takes `max(1, ceil(Exp(60 / mu_k)))` whole minutes. Riders: a fixed pool per store; a picked order leaves with the rider free earliest (lowest index on a tie); travel is `max(1, ceil(distance / speed * LogNormal(0, 0.15)))` minutes along the straight line; the rider is free again after twice the travel time. Road travel times, shifts and rain arrive with P-033 and P-034.
- An order is late when `delivered_at - ordered_at` exceeds the promise; lateness lowers `satisfaction_r` by 0.8 at the delivery minute.
- `first_order_at`, `order_count` and the per-tick order aggregates count served orders only.
- Conservation: attempts = served + deferred + lost; delivery rows = served; per store, initial stock + restocked units - stock at the end = orders it served.
- Seen-nearby exposure needs real cells and arrives with P-033.

**Defaults** (stock rates are per catchment resident)
| Parameter | Default |
| --- | --- |
| initial stock, order-up-to level `S_k` | 0.036 |
| reorder point `s_k` | 0.012 |
| review period | 60 minutes |
| lead time | 24 hours |
| picking rate `mu_k` | 120 orders per hour |
| riders per store | 6 |
| rider speed | 20 km/h |
| travel noise sigma | 0.15 |
| promise | 20 minutes |
| `p_spill`, `p_wait` | 0.42, 0.38 (the prototype's shares; the remaining 0.20 is lost) |
| wait before retrying | 24 hours |
| `spill_radius_factor` | 1.67 (the prototype's 1,500 m against its 900 m radius) |
| satisfaction drop | 0.8 |
