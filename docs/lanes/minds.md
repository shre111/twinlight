# Lane C (minds)

Worktree `.claude/worktrees/minds`. Lead folds this into MEMORY.md at each wrap.

## P-023 social graph (ready to tick)

`citybake graph ahmedabad` on the real city, no parameter tuned. All approved targets hold. 100,002 residents, 297,531 undirected ties (household 132,032, neighbour 79,057, friend 59,937, work 26,505).

| Measure | Result | Target |
|---|---|---|
| Mean degree | 5.95 | 5 to 8 |
| Average local clustering, all ties | 0.2175 | 0.10 to 0.50 |
| Cross-river share, non-household ties | 0.139 | 0.05 to 0.25 |
| Largest component | 0.9992 | at least 0.95 |
| Isolated residents | 0.0004 | at most 0.02 |

- Degree: p10 3, p50 6, p90 9, p99 12, max 18. By kind: household 2.64, neighbour 1.58, friend 1.20, work 0.53.
- Friend homophily: 0.521 same segment against 0.231 under random mixing.
- Strength means: household 0.849, friend 0.601, neighbour 0.390, work 0.350. Per resident mean 0.646 (p10 0.539, p90 0.759).
- Workplaces: 6,803 buildings (OSM tagged 2,656, untagged footprints 4,147). Workers: east 30,277, west 22,984.
- 1.3 s, 5.8 MB, byte-identical on rerun on this machine. `layer.json` records the digests of residents, assignment and buildings, and which buildings stage was used.

Questions and limits for the lead:

- **Clustering target measures households, not social closure.** Without household ties clustering is 0.005 (transitivity 0.0045). Friends of friends rarely know each other, so word of mouth may spread faster than in a real city once it leaves a household. The 0.10 to 0.50 target passes because of household cliques. Triadic closure would be a model change, so I have not added it. Worth checking against the first P-012 adoption curves.
- **Influence is mostly household.** Share of summed w^2: household 71.6%, friend 16.7%, neighbour 9.2%, work 2.6%; share of summed w: 59.6%, 19.1%, 16.4%, 4.9%. Both before `talk_kind`.
- **No single talk reaches the awareness threshold at talk_gain 0.9**: spouse 0.90, housemate 0.72, friend mean 0.54, neighbour mean 0.35, work mean 0.31. Relevant to the possible strength or gain normalisation.
- **The east's workplaces rest on the untagged rule.** Untagged footprints are 90.3% of the east's workplace floor area (36.4% in the west). The east's worker share depends on `workplace_min_area_m2`, which nothing anchors: 2,000 m2 gives 46.7%, 1,000 m2 gives 56.8%, 500 m2 gives 62.4%.
- **3,872 residents (3.87%) live in footprints that also count as workplaces.** P-021 treats unknown footprints as 90% housing. A small double use, recorded rather than changed.
- Work ties cross the river at 0.379 against 0.442 for random mixing, a mild preference for the home bank from the distance decay.
- Byte-identity was checked on one machine only.

`docs/MEMORY.md` on main cites D-027, D-028 and MODEL.md section 11, but DECISIONS.md there stops at D-026 and MODEL.md ends at section 10. Possibly not pushed yet.
