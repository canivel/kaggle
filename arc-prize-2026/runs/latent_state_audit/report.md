# Latent-state audit — per-game hidden-state quantification

Protocol: `learnings/war_room/latent_state_audit_protocol.md`. Selftest: **PASS** (synthetic hidden mod-3 counter recovered; clean stream = CLEAN; coin-flip stream = UNRESOLVED).

Coverage: 40 versioned games, 200 streams, 33777 actions; analysis-frame drift events: 0.

Determinism = P(modal next frame | frame, action) over keys seen >= 2x, visit-weighted (pooled across streams of the same engine version). Entropy = mean outcome entropy (bits). 'within' = keys scoped to a single stream (strongest hidden-state evidence). Resolved = augmented determinism >= 0.99.

## Verdict table

| game | streams | actions | rep.visits | det | H bits | within det | resolver | det.res | verdict | EWM step_acc | step0-abort/plan | EWM carrier | resync | banking |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---|---|---|
| ar25 | 8 | 1020 | 184 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | - | - | SAFE | NOT-NEEDED | PREFIX-SAFE |
| bp35 | 8 | 2914 | 1534 | 0.996 | 0.012 | 0.997 | - | - | **CLEAN** | - | - | SAFE | NOT-NEEDED | PREFIX-SAFE |
| cd82 | 8 | 1362 | 300 | 0.753 | 0.547 | 0.608 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| cn04 | 8 | 1685 | 275 | 0.938 | 0.131 | 0.968 | mod3 | 1.000 | **ALIASED-RESOLVABLE(mod3)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| dc22 | 8 | 787 | 73 | 0.671 | 0.658 | 0.548 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| ft09 | 8 | 801 | 232 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.922 | 0.116 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| g50t | 8 | 865 | 199 | 0.693 | 0.640 | 0.511 | - | - | **ALIASED-UNRESOLVED** | - | - | NO | NO | FULL-REPLAY-ONLY |
| ka59 | 8 | 957 | 108 | 0.741 | 0.530 | 0.519 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| lf52 | 8 | 1107 | 97 | 1.000 | 0.000 | - | - | - | **CLEAN** | 0.532 | 0.479 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| lp85 | 8 | 724 | 142 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.129 | 0.889 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| ls20 | 8 | 739 | 126 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.793 | 0.384 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| m0r0 | 8 | 1674 | 821 | 0.618 | 0.858 | 0.589 | - | - | **ALIASED-UNRESOLVED** | - | - | NO | NO | FULL-REPLAY-ONLY |
| r11l | 8 | 808 | 99 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | - | - | SAFE | NOT-NEEDED | PREFIX-SAFE |
| re86 | 8 | 1193 | 72 | 0.958 | 0.083 | 0.500 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| s5i5 | 8 | 417 | 36 | 0.972 | 0.056 | 0.833 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | 0.293 | 0.707 | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| sb26 | 8 | 1791 | 332 | 0.985 | 0.041 | 0.976 | parity | 0.996 | **ALIASED-RESOLVABLE(parity)** | 0.167 | 0.833 | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| sc25 | 8 | 1341 | 221 | 0.760 | 0.519 | 0.691 | mod5 | 1.000 | **ALIASED-RESOLVABLE(mod5)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| sk48 | 8 | 1310 | 365 | 0.767 | 0.586 | 0.724 | - | - | **ALIASED-UNRESOLVED** | - | - | NO | NO | FULL-REPLAY-ONLY |
| sp80 | 8 | 1658 | 489 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.564 | 0.524 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| su15 | 8 | 1087 | 338 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.312 | 0.724 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| tn36 | 8 | 932 | 31 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.965 | 0.094 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| tr87 | 8 | 1740 | 156 | 0.910 | 0.207 | 1.000 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | 0.800 | 0.236 | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| tu93 | 8 | 1057 | 667 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.922 | 0.152 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| vc33 | 8 | 697 | 60 | 0.983 | 0.033 | 0.967 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | 0.458 | 0.591 | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| wa30 | 8 | 2390 | 425 | 0.739 | 0.704 | 0.735 | mod4 | 1.000 | **ALIASED-RESOLVABLE(mod4)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |

Rows reflect the benchmark engine version (most streams). Minority-version drift and near-misses:

- **cn04**: older engine version(s) disagree — cn04-65d47d14:ALIASED-RESOLVABLE(mod5) (phase1_ab/seed1 era); engine-version drift, NOT merged into the verdict.
- **ka59**: older engine version(s) disagree — ka59-9f096b4a:ALIASED-UNRESOLVED (phase1_ab/seed1 era); engine-version drift, NOT merged into the verdict.
- **m0r0**: candidate(s) mod3, mod4, mod5 reach >= 0.99 determinism on the repeat support that survives augmentation, but fail the support guard (SUPPORT-COLLAPSED) — plausibly resolvable with more data; treated as UNRESOLVED until then.

## Candidate breakdown (aliased games only)

### cd82 (cd82-fb555c5d) — base det 0.753, 68/126 keys aliased, 35 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.753 | 300 | y | - |
| score | observable-meta | 0.753 | 300 | y | - |
| meta | observable-meta | 0.753 | 300 | y | - |
| parity | hidden-phase | 1.000 | 190 | y | YES |
| mod3 | hidden-phase | 1.000 | 190 | y | YES |
| mod4 | hidden-phase | 1.000 | 190 | y | YES |
| mod5 | hidden-phase | 1.000 | 190 | y | YES |
| prev_bc | hidden-history | 0.838 | 216 | y | - |
| hist1 | hidden-history | 0.720 | 168 | y | - |
| hist2 | hidden-history | 0.705 | 88 | y | - |
| hist3 | hidden-history | 0.708 | 48 | n | - |
| meta_parity | compound | 1.000 | 190 | y | YES |
| meta_hist1 | compound | 0.720 | 168 | y | - |
| tcount | diagnostic | 1.000 | 190 | n | - |

### cn04 (cn04-2fe56bfb) — base det 0.938, 17/106 keys aliased, 5 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.938 | 275 | y | - |
| score | observable-meta | 0.938 | 275 | y | - |
| meta | observable-meta | 0.938 | 275 | y | - |
| parity | hidden-phase | 0.976 | 245 | y | - |
| mod3 | hidden-phase | 1.000 | 235 | y | YES |
| mod4 | hidden-phase | 1.000 | 235 | y | YES |
| mod5 | hidden-phase | 1.000 | 235 | y | YES |
| prev_bc | hidden-history | 0.954 | 261 | y | - |
| hist1 | hidden-history | 0.955 | 242 | y | - |
| hist2 | hidden-history | 0.968 | 217 | y | - |
| hist3 | hidden-history | 0.975 | 198 | y | - |
| meta_parity | compound | 0.976 | 245 | y | - |
| meta_hist1 | compound | 0.955 | 242 | y | - |
| tcount | diagnostic | 1.000 | 235 | n | - |

### dc22 (dc22-fdcac232) — base det 0.671, 24/36 keys aliased, 19 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.671 | 73 | y | - |
| score | observable-meta | 0.671 | 73 | y | - |
| meta | observable-meta | 0.671 | 73 | y | - |
| parity | hidden-phase | 1.000 | 23 | y | YES |
| mod3 | hidden-phase | 0.957 | 23 | y | - |
| mod4 | hidden-phase | 1.000 | 21 | y | YES |
| mod5 | hidden-phase | 0.920 | 25 | y | - |
| prev_bc | hidden-history | 0.848 | 33 | y | - |
| hist1 | hidden-history | 0.653 | 49 | y | - |
| hist2 | hidden-history | 0.667 | 33 | y | - |
| hist3 | hidden-history | 0.737 | 19 | y | - |
| meta_parity | compound | 1.000 | 23 | y | YES |
| meta_hist1 | compound | 0.653 | 49 | y | - |
| tcount | diagnostic | 1.000 | 21 | n | - |

### g50t (g50t-5849a774) — base det 0.693, 59/83 keys aliased, 45 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.693 | 199 | y | - |
| score | observable-meta | 0.693 | 199 | y | - |
| meta | observable-meta | 0.693 | 199 | y | - |
| parity | hidden-phase | 0.978 | 93 | y | - |
| mod3 | hidden-phase | 0.989 | 91 | y | - |
| mod4 | hidden-phase | 0.989 | 91 | y | - |
| mod5 | hidden-phase | 0.989 | 91 | y | - |
| prev_bc | hidden-history | 0.858 | 106 | y | - |
| hist1 | hidden-history | 0.717 | 145 | y | - |
| hist2 | hidden-history | 0.702 | 114 | y | - |
| hist3 | hidden-history | 0.716 | 88 | y | - |
| meta_parity | compound | 0.978 | 93 | y | - |
| meta_hist1 | compound | 0.717 | 145 | y | - |
| tcount | diagnostic | 0.989 | 91 | n | - |

### ka59 (ka59-38d34dbb) — base det 0.741, 28/45 keys aliased, 26 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.741 | 108 | y | - |
| score | observable-meta | 0.741 | 108 | y | - |
| meta | observable-meta | 0.741 | 108 | y | - |
| parity | hidden-phase | 1.000 | 41 | y | YES |
| mod3 | hidden-phase | 1.000 | 33 | y | YES |
| mod4 | hidden-phase | 1.000 | 37 | y | YES |
| mod5 | hidden-phase | 1.000 | 29 | y | YES |
| prev_bc | hidden-history | 0.947 | 57 | y | - |
| hist1 | hidden-history | 0.724 | 98 | y | - |
| hist2 | hidden-history | 0.736 | 87 | y | - |
| hist3 | hidden-history | 0.728 | 81 | y | - |
| meta_parity | compound | 1.000 | 41 | y | YES |
| meta_hist1 | compound | 0.724 | 98 | y | - |
| tcount | diagnostic | 1.000 | 23 | n | - |

### m0r0 (m0r0-492f87ba) — base det 0.618, 298/340 keys aliased, 282 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.618 | 821 | y | - |
| score | observable-meta | 0.618 | 821 | y | - |
| meta | observable-meta | 0.618 | 821 | y | - |
| parity | hidden-phase | 0.728 | 309 | y | - |
| mod3 | hidden-phase | 1.000 | 147 | n | - |
| mod4 | hidden-phase | 1.000 | 147 | n | - |
| mod5 | hidden-phase | 1.000 | 147 | n | - |
| prev_bc | hidden-history | 0.674 | 313 | y | - |
| hist1 | hidden-history | 0.613 | 741 | y | - |
| hist2 | hidden-history | 0.606 | 688 | y | - |
| hist3 | hidden-history | 0.614 | 611 | y | - |
| meta_parity | compound | 0.728 | 309 | y | - |
| meta_hist1 | compound | 0.613 | 741 | y | - |
| tcount | diagnostic | 1.000 | 147 | n | - |

### re86 (re86-8af5384d) — base det 0.958, 3/27 keys aliased, 3 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.958 | 72 | y | - |
| score | observable-meta | 0.958 | 72 | y | - |
| meta | observable-meta | 0.958 | 72 | y | - |
| parity | hidden-phase | 1.000 | 47 | y | YES |
| mod3 | hidden-phase | 1.000 | 51 | y | YES |
| mod4 | hidden-phase | 1.000 | 33 | y | YES |
| mod5 | hidden-phase | 1.000 | 19 | y | YES |
| prev_bc | hidden-history | 1.000 | 66 | y | YES |
| hist1 | hidden-history | 0.984 | 61 | y | - |
| hist2 | hidden-history | 0.982 | 55 | y | - |
| hist3 | hidden-history | 0.979 | 47 | y | - |
| meta_parity | compound | 1.000 | 47 | y | YES |
| meta_hist1 | compound | 0.984 | 61 | y | - |
| tcount | diagnostic | 1.000 | 19 | n | - |

### s5i5 (s5i5-18d95033) — base det 0.972, 1/15 keys aliased, 1 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.972 | 36 | y | - |
| score | observable-meta | 0.972 | 36 | y | - |
| meta | observable-meta | 0.972 | 36 | y | - |
| parity | hidden-phase | 1.000 | 32 | y | YES |
| mod3 | hidden-phase | 1.000 | 32 | y | YES |
| mod4 | hidden-phase | 1.000 | 32 | y | YES |
| mod5 | hidden-phase | 1.000 | 32 | y | YES |
| prev_bc | hidden-history | 1.000 | 31 | y | YES |
| hist1 | hidden-history | 0.929 | 14 | y | - |
| hist2 | hidden-history | 0.917 | 12 | y | - |
| hist3 | hidden-history | 0.917 | 12 | y | - |
| meta_parity | compound | 1.000 | 32 | y | YES |
| meta_hist1 | compound | 0.929 | 14 | y | - |
| tcount | diagnostic | 1.000 | 32 | n | - |

### sb26 (sb26-7fbdac44) — base det 0.985, 5/119 keys aliased, 3 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.985 | 332 | y | - |
| score | observable-meta | 0.985 | 332 | y | - |
| meta | observable-meta | 0.985 | 332 | y | - |
| parity | hidden-phase | 0.996 | 274 | y | YES |
| mod3 | hidden-phase | 0.989 | 266 | y | - |
| mod4 | hidden-phase | 0.996 | 255 | y | YES |
| mod5 | hidden-phase | 0.996 | 234 | y | YES |
| prev_bc | hidden-history | 0.993 | 305 | y | YES |
| hist1 | hidden-history | 0.996 | 267 | y | YES |
| hist2 | hidden-history | 1.000 | 237 | y | YES |
| hist3 | hidden-history | 1.000 | 211 | y | YES |
| meta_parity | compound | 0.996 | 274 | y | YES |
| meta_hist1 | compound | 0.996 | 267 | y | YES |
| tcount | diagnostic | 0.995 | 220 | n | - |

### sc25 (sc25-635fd71a) — base det 0.760, 44/85 keys aliased, 43 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.760 | 221 | y | - |
| score | observable-meta | 0.760 | 221 | y | - |
| meta | observable-meta | 0.760 | 221 | y | - |
| parity | hidden-phase | 0.915 | 129 | y | - |
| mod3 | hidden-phase | 0.971 | 104 | y | - |
| mod4 | hidden-phase | 0.982 | 112 | y | - |
| mod5 | hidden-phase | 1.000 | 98 | y | YES |
| prev_bc | hidden-history | 0.993 | 136 | y | YES |
| hist1 | hidden-history | 0.789 | 180 | y | - |
| hist2 | hidden-history | 0.773 | 141 | y | - |
| hist3 | hidden-history | 0.760 | 100 | y | - |
| meta_parity | compound | 0.915 | 129 | y | - |
| meta_hist1 | compound | 0.789 | 180 | y | - |
| tcount | diagnostic | 1.000 | 86 | n | - |

### sk48 (sk48-d8078629) — base det 0.767, 79/129 keys aliased, 61 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.767 | 365 | y | - |
| score | observable-meta | 0.767 | 365 | y | - |
| meta | observable-meta | 0.767 | 365 | y | - |
| parity | hidden-phase | 0.750 | 200 | y | - |
| mod3 | hidden-phase | 0.957 | 117 | y | - |
| mod4 | hidden-phase | 0.974 | 114 | y | - |
| mod5 | hidden-phase | 0.982 | 112 | y | - |
| prev_bc | hidden-history | 0.754 | 264 | y | - |
| hist1 | hidden-history | 0.768 | 340 | y | - |
| hist2 | hidden-history | 0.762 | 311 | y | - |
| hist3 | hidden-history | 0.762 | 286 | y | - |
| meta_parity | compound | 0.750 | 200 | y | - |
| meta_hist1 | compound | 0.768 | 340 | y | - |
| tcount | diagnostic | 0.982 | 112 | n | - |

### tr87 (tr87-cd924810) — base det 0.910, 11/51 keys aliased, 0 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.910 | 156 | y | - |
| score | observable-meta | 0.910 | 156 | y | - |
| meta | observable-meta | 0.910 | 156 | y | - |
| parity | hidden-phase | 1.000 | 143 | y | YES |
| mod3 | hidden-phase | 1.000 | 143 | y | YES |
| mod4 | hidden-phase | 1.000 | 143 | y | YES |
| mod5 | hidden-phase | 1.000 | 143 | y | YES |
| prev_bc | hidden-history | 0.910 | 156 | y | - |
| hist1 | hidden-history | 0.928 | 139 | y | - |
| hist2 | hidden-history | 0.933 | 120 | y | - |
| hist3 | hidden-history | 0.933 | 105 | y | - |
| meta_parity | compound | 1.000 | 143 | y | YES |
| meta_hist1 | compound | 0.928 | 139 | y | - |
| tcount | diagnostic | 1.000 | 143 | n | - |

### vc33 (vc33-5430563c) — base det 0.983, 1/28 keys aliased, 1 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.983 | 60 | y | - |
| score | observable-meta | 0.983 | 60 | y | - |
| meta | observable-meta | 0.983 | 60 | y | - |
| parity | hidden-phase | 1.000 | 54 | y | YES |
| mod3 | hidden-phase | 1.000 | 32 | y | YES |
| mod4 | hidden-phase | 1.000 | 52 | y | YES |
| mod5 | hidden-phase | 1.000 | 29 | y | YES |
| prev_bc | hidden-history | 1.000 | 58 | y | YES |
| hist1 | hidden-history | 0.979 | 47 | y | - |
| hist2 | hidden-history | 1.000 | 31 | y | YES |
| hist3 | hidden-history | 1.000 | 25 | y | YES |
| meta_parity | compound | 1.000 | 54 | y | YES |
| meta_hist1 | compound | 0.979 | 47 | y | - |
| tcount | diagnostic | 1.000 | 9 | n | - |

### wa30 (wa30-ee6fef47) — base det 0.739, 105/142 keys aliased, 98 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.739 | 425 | y | - |
| score | observable-meta | 0.739 | 425 | y | - |
| meta | observable-meta | 0.739 | 425 | y | - |
| parity | hidden-phase | 0.673 | 263 | y | - |
| mod3 | hidden-phase | 0.906 | 106 | y | - |
| mod4 | hidden-phase | 1.000 | 86 | y | YES |
| mod5 | hidden-phase | 1.000 | 86 | y | YES |
| prev_bc | hidden-history | 0.654 | 257 | y | - |
| hist1 | hidden-history | 0.741 | 394 | y | - |
| hist2 | hidden-history | 0.736 | 364 | y | - |
| hist3 | hidden-history | 0.732 | 336 | y | - |
| meta_parity | compound | 0.673 | 263 | y | - |
| meta_hist1 | compound | 0.741 | 394 | y | - |
| tcount | diagnostic | 1.000 | 86 | n | - |

## Findings (ties to the three R15 failures)

1. **Hidden phase counters are the dominant aliasing mechanism**: 11/14 aliased benchmark games are fully resolved (det -> ~1.000) by a small modular counter of actions-since-RESET (parity or mod 3/4/5) — an invisible blink/tick phase. Observable metadata (level/score) resolves NOTHING: the hidden variable is truly outside the observation.
2. **This is the predict-metric 0.465 mechanism**: in the aliased games, most aliased (frame,action) keys have a no-effect outcome on one phase and an effect on the other (see 'involve a no-effect outcome' counts). A no-effect FACT keyed on (frame,action) alone is wrong whenever the phase differs on recurrence — exactly the ~54% flip rate R14 measured.
3. **This is the N5 prune_trace mechanism**: no-op actions still advance the phase counter; dropping leading no-ops desyncs the phase and the first replayed action lands on a different frame (step-0 frame_divergence on sc25/m0r0 — sc25 is mod5-aliased here; m0r0 is the worst unresolved game, det 0.618).
4. **EWM step-0 aborts split into two causes**: on ALIASED games (s5i5, sb26, vc33, tr87) low sim step_acc co-occurs with phase aliasing — the sim is phase-blind, and resync/phase-augmentation fixes it. But lf52, lp85, sp80, su15 are frame-Markov CLEAN yet still have step_acc < 0.6 — those sims are just wrong (sim bugs / engine-version drift), and NO amount of state augmentation or resync will save them; they need sim fixes, not aliasing work.

## Consumer answers

- **EWM Stage-1 safe carriers** (frame(+meta) is Markov): ar25, bp35, ft09, lf52, lp85, ls20, r11l, sp80, su15, tn36, tu93
- **Resync-before-abort viable** (phase variable drifts, reality deterministic): cd82, cn04, dc22, ka59, re86, s5i5, sb26, sc25, tr87, vc33, wa30
- **EWM no-go** (unresolved aliasing — abort-and-fallback is correct): g50t, m0r0, sk48
- **Banking prefix-splice safe**: ar25, bp35, ft09, lf52, lp85, ls20, r11l, sp80, su15, tn36, tu93; all other audited games are FULL-REPLAY-ONLY from RESET with ZERO pruning (N5: full unpruned replay survives on all 25; the prune_trace bug dropped hidden-state-mutating no-ops).
