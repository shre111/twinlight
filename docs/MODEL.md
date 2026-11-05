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
| `omega_r(t)` | snack/consumption occasion intensity by time of day (segment profile) |

**Tie** `(u, v)` from the social graph: kind (household, neighbour, friend, work), strength `w_uv` in (0, 1].

**Store** `k`: position, delivery radius, initial stock per SKU, reorder policy `(s_k, S_k)`, review period, lead time distribution, picking rate `mu_k` (orders per hour), rider shifts.

**Campaign** `c`: channel, schedule, budget per day, CPM, targeting weights over segments and districts, frequency cap.

**Weather**: rain intensity field `rain(t, hex)` from the scenario's event list.

## 3. Awareness
Each resident holds awareness strength `A_r >= 0`. Per tick:
```
A_r <- A_r * 2^(-dt / tau_r)  +  paid_r  +  wom_r  +  seen_r
aware_at_r = first tick where A_r >= theta_aware   (theta_aware = 1.0)
```
**Paid exposure.** Campaign impressions this tick `I_c = budget_c(t) / CPM_c * 1000`. Impressions are allocated to residents with probability proportional to `target_c(s(r), district(r)) * presence_c(s(r), slot)`, capped at `freq_cap_c` per day, sampled with the campaign's stream. Each impression adds `kappa_r,c * 0.6`.

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
- `presence` is per campaign (24 hourly weights) in v1; per-segment presence arrives with P-030.
- Until P-013: an order is counted at its occasion, `avail = 1`, `satisfaction = 0`, and the DELIVERIES section stays empty.

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
