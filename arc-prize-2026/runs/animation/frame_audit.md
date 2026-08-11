# Animation-frame audit -- 25 official games (our engines, our recorded behaviour)

generated 2026-08-11T07:52:12 | bench `benchmark.json` | probe A = recorded history (<= 400 actions), probe B = 300 seeded actions | LM-free, offline

`MULTI` = engine returned >1 frame for one action. `INVISIBLE` = settled board identical to the previous settled board AND at least one intermediate frame differed -- the agent saw "nothing happened" while the engine had rendered something.

| game | type | actions | MULTI | MULTI% | max frames | first==last | settled-unchanged | INVISIBLE | INV% actions | INV% of no-ops | max transient cells | INV (probe A recorded) | INV (probe B seeded) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ar25 | single | 531 | 0 | 0.0% | 1 | 0 | 85 | **0** | 0.0% | 0.0% | 0 | 0/231 | 0/300 |
| bp35 | type2 | 458 | 458 | 100.0% | 47 | 163 | 0 | **0** | 0.0% | 0.0% | 2392 | 0/158 | 0/300 |
| cd82 | type1 | 522 | 96 | 18.4% | 16 | 55 | 120 | **20** | 3.8% | 16.7% | 460 | 8/222 | 12/300 |
| cn04 | single | 700 | 0 | 0.0% | 1 | 0 | 116 | **0** | 0.0% | 0.0% | 0 | 0/400 | 0/300 |
| dc22 | single | 343 | 0 | 0.0% | 1 | 0 | 76 | **0** | 0.0% | 0.0% | 0 | 0/43 | 0/300 |
| ft09 | type1 | 352 | 284 | 80.7% | 5 | 281 | 283 | **281** | 79.8% | 99.3% | 3562 | 0/52 | 281/300 |
| g50t | type2 | 346 | 182 | 52.6% | 53 | 2 | 84 | **0** | 0.0% | 0.0% | 1199 | 0/46 | 0/300 |
| ka59 | type2 | 367 | 3 | 0.8% | 7 | 0 | 51 | **0** | 0.0% | 0.0% | 3534 | 0/67 | 0/300 |
| lf52 | type2 | 325 | 325 | 100.0% | 16 | 259 | 0 | **0** | 0.0% | 0.0% | 36 | 0/25 | 0/300 |
| lp85 | type2 | 309 | 1 | 0.3% | 2 | 0 | 294 | **0** | 0.0% | 0.0% | 1491 | 0/9 | 0/300 |
| ls20 | type1 | 533 | 33 | 6.2% | 6 | 0 | 19 | **19** | 3.6% | 100.0% | 4012 | 19/233 | 0/300 |
| m0r0 | single | 354 | 0 | 0.0% | 1 | 0 | 94 | **0** | 0.0% | 0.0% | 0 | 0/54 | 0/300 |
| r11l | type2 | 330 | 246 | 74.5% | 23 | 6 | 0 | **0** | 0.0% | 0.0% | 1392 | 0/30 | 0/300 |
| re86 | single | 700 | 0 | 0.0% | 1 | 0 | 26 | **0** | 0.0% | 0.0% | 0 | 0/400 | 0/300 |
| s5i5 | single | 360 | 0 | 0.0% | 1 | 0 | 0 | **0** | 0.0% | 0.0% | 0 | 0/60 | 0/300 |
| sb26 | type2 | 482 | 173 | 35.9% | 118 | 109 | 225 | **0** | 0.0% | 0.0% | 882 | 0/182 | 0/300 |
| sc25 | type1 | 428 | 91 | 21.3% | 22 | 85 | 111 | **81** | 18.9% | 73.0% | 81 | 24/128 | 57/300 |
| sk48 | type2 | 511 | 231 | 45.2% | 3 | 0 | 198 | **0** | 0.0% | 0.0% | 226 | 0/211 | 0/300 |
| sp80 | type2 | 635 | 51 | 8.0% | 28 | 0 | 0 | **0** | 0.0% | 0.0% | 1568 | 0/335 | 0/300 |
| su15 | type2 | 376 | 192 | 51.1% | 14 | 0 | 173 | **0** | 0.0% | 0.0% | 341 | 0/76 | 0/300 |
| tn36 | type2 | 355 | 19 | 5.4% | 7 | 4 | 0 | **0** | 0.0% | 0.0% | 69 | 0/55 | 0/300 |
| tr87 | single | 395 | 0 | 0.0% | 1 | 0 | 0 | **0** | 0.0% | 0.0% | 0 | 0/95 | 0/300 |
| tu93 | type2 | 358 | 187 | 52.2% | 15 | 0 | 0 | **0** | 0.0% | 0.0% | 729 | 0/58 | 0/300 |
| vc33 | type2 | 334 | 1 | 0.3% | 2 | 0 | 0 | **0** | 0.0% | 0.0% | 2921 | 0/34 | 0/300 |
| wa30 | single | 700 | 0 | 0.0% | 1 | 0 | 156 | **0** | 0.0% | 0.0% | 0 | 0/400 | 0/300 |

**Totals:** 17/25 games return multi-frame responses (4 type-1, 13 type-2, 8 single-frame). 2573/11104 actions (23.2%) were animated. **401 actions (3.6% of all, 19.0% of apparent no-ops) carried signal the agent could not see.**
