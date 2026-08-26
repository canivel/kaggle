# Held-out resolver validation — latent-state audit (R17 item 2)

Mandate: R16 C6 (rl-planning MAJOR#2, prog-synthesis N2); R17 checklist item 2. Protocol: `learnings/war_room/latent_state_audit_protocol.md` + this holdout extension. Selftest: **PASS (base) / PASS (holdout)** (synthetic hidden-mod3 game: resolver fit on TRAIN certifies on HELD-OUT streams with Wilson LB >= 0.95 -> KEEP; synthetic coin-flip noise -> UNRESOLVED; synthetic low-support mod3 -> DROP despite held-out det 1.0).

Method: for each in-sample ALIASED-RESOLVABLE game, streams of the **benchmark engine version only** (versions are never pooled — cn04/ka59 drift) are split sorted stream ids, alternating; even index -> TRAIN (4/4 on 8 streams); engine versions never pooled. The resolver is fit/selected on TRAIN streams alone (same minimal-candidate rule + support guard as the main audit, det >= 0.99); it certifies iff held-out pooled augmented determinism >= 0.99 AND its Wilson 95% lower bound >= 0.95. Any failure (fit fails on TRAIN, or holdout certificate fails) -> **ALIASED-UNRESOLVED**. No fallback to the in-sample resolver — that is the selection leak C6 bans.

## Per-game validation (in-sample ALIASED-RESOLVABLE games)

| game | engine version | streams (tr/ho) | in-sample verdict | train resolver | hold det | hold visits | Wilson LB | status | validated verdict | EWM carrier | resync | banking |
|---|---|---|---|---|---:|---:|---:|---|---|---|---|---|
| cd82 | cd82-fb555c5d | 4/4 | ALIASED-RESOLVABLE(parity) | parity | 1.000 | 43 | 0.918 | DROP-HOLDOUT | **ALIASED-UNRESOLVED** | NO | NO | FULL-REPLAY-ONLY |
| cn04 | cn04-2fe56bfb | 4/3 | ALIASED-RESOLVABLE(mod3) | mod3 | 1.000 | 23 | 0.857 | DROP-HOLDOUT | **ALIASED-UNRESOLVED** | NO | NO | FULL-REPLAY-ONLY |
| dc22 | dc22-fdcac232 | 4/3 | ALIASED-RESOLVABLE(parity) | - | - | - | - | DROP-FIT-FAILED | **ALIASED-UNRESOLVED** | NO | NO | FULL-REPLAY-ONLY |
| ka59 | ka59-38d34dbb | 4/3 | ALIASED-RESOLVABLE(parity) | parity | 1.000 | 6 | 0.610 | DROP-HOLDOUT | **ALIASED-UNRESOLVED** | NO | NO | FULL-REPLAY-ONLY |
| re86 | re86-8af5384d | 4/3 | ALIASED-RESOLVABLE(parity) | - | - | - | - | DROP-FIT-FAILED | **ALIASED-UNRESOLVED** | NO | NO | FULL-REPLAY-ONLY |
| s5i5 | s5i5-18d95033 | 4/3 | ALIASED-RESOLVABLE(parity) | - | - | - | - | DROP-FIT-FAILED | **ALIASED-UNRESOLVED** | NO | NO | FULL-REPLAY-ONLY |
| sb26 | sb26-7fbdac44 | 4/4 | ALIASED-RESOLVABLE(parity) | hist1 | 1.000 | 190 | 0.980 | KEEP | **ALIASED-RESOLVABLE(hist1)** | HISTORY-AUGMENT | N/A | FULL-REPLAY-ONLY |
| sc25 | sc25-635fd71a | 4/3 | ALIASED-RESOLVABLE(mod5) | mod4 | 0.931 | 29 | 0.780 | DROP-HOLDOUT | **ALIASED-UNRESOLVED** | NO | NO | FULL-REPLAY-ONLY |
| tr87 | tr87-cd924810 | 4/4 | ALIASED-RESOLVABLE(parity) | parity | 1.000 | 49 | 0.927 | DROP-HOLDOUT | **ALIASED-UNRESOLVED** | NO | NO | FULL-REPLAY-ONLY |
| vc33 | vc33-5430563c | 4/3 | ALIASED-RESOLVABLE(parity) | - | - | - | - | DROP-FIT-FAILED | **ALIASED-UNRESOLVED** | NO | NO | FULL-REPLAY-ONLY |
| wa30 | wa30-ee6fef47 | 4/4 | ALIASED-RESOLVABLE(mod4) | - | - | - | - | DROP-FIT-FAILED | **ALIASED-UNRESOLVED** | NO | NO | FULL-REPLAY-ONLY |

**Verdict changes: 10/11 in-sample RESOLVABLE games drop to UNRESOLVED** (cd82, cn04, dc22, ka59, re86, s5i5, sc25, tr87, vc33, wa30); sb26 keep RESOLVABLE with a held-out certificate.

### 3/5-split sensitivity (non-binding)

Fit on 3 streams / certify on 5 (the directive's 'fit on <=6' variant maximises holdout support). Published for transparency; the 4/4 split above is the binding certificate. Any KEEP that flips across splits is split-sensitive and should be treated as fragile by consumers.

| game | status(3/5) | resolver | hold det | hold visits | Wilson LB |
|---|---|---|---:|---:|---:|
| cd82 | DROP-HOLDOUT | parity | 1.000 | 53 | 0.932 |
| cn04 | DROP-HOLDOUT | parity | 0.965 | 57 | 0.881 |
| dc22 | DROP-FIT-FAILED | - | - | - | - |
| ka59 | DROP-HOLDOUT | parity | 1.000 | 11 | 0.741 |
| re86 | DROP-FIT-FAILED | - | - | - | - |
| s5i5 | DROP-FIT-FAILED | - | - | - | - |
| sb26 | DROP-FIT-FAILED | - | - | - | - |
| sc25 | DROP-HOLDOUT | mod4 | 0.935 | 31 | 0.793 |
| tr87 | DROP-FIT-FAILED | - | - | - | - |
| vc33 | DROP-FIT-FAILED | - | - | - | - |
| wa30 | DROP-FIT-FAILED | - | - | - | - |

## Per-stream resolver table

Per-stream determinism under the TRAIN-selected resolver key (within-stream repeat visits); CLEAN games appear with the base (unaugmented) key and no split (nothing was fit).

| game | stream | engine version | split | visits | det | Wilson LB | game verdict (validated) |
|---|---|---|---|---:|---:|---:|---|
| cd82 | phase1_v5 | cd82-fb555c5d | TRAIN | 4 | 1.000 | 0.510 | ALIASED-UNRESOLVED |
| cd82 | sched_v1 | cd82-fb555c5d | HOLDOUT | 4 | 1.000 | 0.510 | ALIASED-UNRESOLVED |
| cd82 | seed1 | cd82-fb555c5d | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| cd82 | w0_eval_s1 | cd82-fb555c5d | HOLDOUT | 2 | 1.000 | 0.342 | ALIASED-UNRESOLVED |
| cd82 | war_eval_v1 | cd82-fb555c5d | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| cd82 | war_eval_v2 | cd82-fb555c5d | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| cd82 | war_eval_v3 | cd82-fb555c5d | TRAIN | 6 | 1.000 | 0.610 | ALIASED-UNRESOLVED |
| cd82 | war_v2_eval_s1 | cd82-fb555c5d | HOLDOUT | 2 | 1.000 | 0.342 | ALIASED-UNRESOLVED |
| cn04 | phase1_v5 | cn04-2fe56bfb | TRAIN | 6 | 1.000 | 0.610 | ALIASED-UNRESOLVED |
| cn04 | sched_v1 | cn04-2fe56bfb | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| cn04 | w0_eval_s1 | cn04-2fe56bfb | TRAIN | 26 | 1.000 | 0.871 | ALIASED-UNRESOLVED |
| cn04 | war_eval_v1 | cn04-2fe56bfb | HOLDOUT | 2 | 1.000 | 0.342 | ALIASED-UNRESOLVED |
| cn04 | war_eval_v2 | cn04-2fe56bfb | TRAIN | 7 | 1.000 | 0.646 | ALIASED-UNRESOLVED |
| cn04 | war_eval_v3 | cn04-2fe56bfb | HOLDOUT | 12 | 1.000 | 0.758 | ALIASED-UNRESOLVED |
| cn04 | war_v2_eval_s1 | cn04-2fe56bfb | TRAIN | 89 | 1.000 | 0.959 | ALIASED-UNRESOLVED |
| dc22 | phase1_v5 | dc22-fdcac232 | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| dc22 | sched_v1 | dc22-fdcac232 | HOLDOUT | 12 | 0.583 | 0.320 | ALIASED-UNRESOLVED |
| dc22 | w0_eval_s1 | dc22-fdcac232 | TRAIN | 6 | 0.667 | 0.300 | ALIASED-UNRESOLVED |
| dc22 | war_eval_v1 | dc22-fdcac232 | HOLDOUT | 4 | 0.500 | 0.150 | ALIASED-UNRESOLVED |
| dc22 | war_eval_v2 | dc22-fdcac232 | TRAIN | 8 | 0.500 | 0.215 | ALIASED-UNRESOLVED |
| dc22 | war_eval_v3 | dc22-fdcac232 | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| dc22 | war_v2_eval_s1 | dc22-fdcac232 | TRAIN | 12 | 0.500 | 0.254 | ALIASED-UNRESOLVED |
| ka59 | phase1_v5 | ka59-38d34dbb | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| ka59 | sched_v1 | ka59-38d34dbb | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| ka59 | w0_eval_s1 | ka59-38d34dbb | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| ka59 | war_eval_v1 | ka59-38d34dbb | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| ka59 | war_eval_v2 | ka59-38d34dbb | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| ka59 | war_eval_v3 | ka59-38d34dbb | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| ka59 | war_v2_eval_s1 | ka59-38d34dbb | TRAIN | 2 | 1.000 | 0.342 | ALIASED-UNRESOLVED |
| re86 | phase1_v5 | re86-8af5384d | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| re86 | sched_v1 | re86-8af5384d | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| re86 | w0_eval_s1 | re86-8af5384d | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| re86 | war_eval_v1 | re86-8af5384d | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| re86 | war_eval_v2 | re86-8af5384d | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| re86 | war_eval_v3 | re86-8af5384d | HOLDOUT | 6 | 0.500 | 0.188 | ALIASED-UNRESOLVED |
| re86 | war_v2_eval_s1 | re86-8af5384d | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| s5i5 | phase1_v5 | s5i5-18d95033 | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| s5i5 | sched_v1 | s5i5-18d95033 | HOLDOUT | 2 | 1.000 | 0.342 | ALIASED-UNRESOLVED |
| s5i5 | w0_eval_s1 | s5i5-18d95033 | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| s5i5 | war_eval_v1 | s5i5-18d95033 | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| s5i5 | war_eval_v2 | s5i5-18d95033 | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| s5i5 | war_eval_v3 | s5i5-18d95033 | HOLDOUT | 4 | 0.750 | 0.301 | ALIASED-UNRESOLVED |
| s5i5 | war_v2_eval_s1 | s5i5-18d95033 | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| sb26 | phase1_v5 | sb26-7fbdac44 | TRAIN | 0 | - | - | ALIASED-RESOLVABLE(hist1) |
| sb26 | sched_v1 | sb26-7fbdac44 | HOLDOUT | 0 | - | - | ALIASED-RESOLVABLE(hist1) |
| sb26 | seed1 | sb26-7fbdac44 | TRAIN | 2 | 1.000 | 0.342 | ALIASED-RESOLVABLE(hist1) |
| sb26 | w0_eval_s1 | sb26-7fbdac44 | HOLDOUT | 0 | - | - | ALIASED-RESOLVABLE(hist1) |
| sb26 | war_eval_v1 | sb26-7fbdac44 | TRAIN | 0 | - | - | ALIASED-RESOLVABLE(hist1) |
| sb26 | war_eval_v2 | sb26-7fbdac44 | HOLDOUT | 2 | 1.000 | 0.342 | ALIASED-RESOLVABLE(hist1) |
| sb26 | war_eval_v3 | sb26-7fbdac44 | TRAIN | 0 | - | - | ALIASED-RESOLVABLE(hist1) |
| sb26 | war_v2_eval_s1 | sb26-7fbdac44 | HOLDOUT | 123 | 1.000 | 0.970 | ALIASED-RESOLVABLE(hist1) |
| sc25 | phase1_v5 | sc25-635fd71a | TRAIN | 12 | 1.000 | 0.758 | ALIASED-UNRESOLVED |
| sc25 | sched_v1 | sc25-635fd71a | HOLDOUT | 6 | 0.667 | 0.300 | ALIASED-UNRESOLVED |
| sc25 | w0_eval_s1 | sc25-635fd71a | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| sc25 | war_eval_v1 | sc25-635fd71a | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| sc25 | war_eval_v2 | sc25-635fd71a | TRAIN | 6 | 1.000 | 0.610 | ALIASED-UNRESOLVED |
| sc25 | war_eval_v3 | sc25-635fd71a | HOLDOUT | 4 | 1.000 | 0.510 | ALIASED-UNRESOLVED |
| sc25 | war_v2_eval_s1 | sc25-635fd71a | TRAIN | 12 | 1.000 | 0.758 | ALIASED-UNRESOLVED |
| tr87 | phase1_v5 | tr87-cd924810 | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| tr87 | sched_v1 | tr87-cd924810 | HOLDOUT | 10 | 1.000 | 0.722 | ALIASED-UNRESOLVED |
| tr87 | seed1 | tr87-cd924810 | TRAIN | 3 | 1.000 | 0.439 | ALIASED-UNRESOLVED |
| tr87 | w0_eval_s1 | tr87-cd924810 | HOLDOUT | 14 | 1.000 | 0.785 | ALIASED-UNRESOLVED |
| tr87 | war_eval_v1 | tr87-cd924810 | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| tr87 | war_eval_v2 | tr87-cd924810 | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| tr87 | war_eval_v3 | tr87-cd924810 | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| tr87 | war_v2_eval_s1 | tr87-cd924810 | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| vc33 | phase1_v5 | vc33-5430563c | TRAIN | 0 | - | - | ALIASED-UNRESOLVED |
| vc33 | sched_v1 | vc33-5430563c | HOLDOUT | 2 | 1.000 | 0.342 | ALIASED-UNRESOLVED |
| vc33 | w0_eval_s1 | vc33-5430563c | TRAIN | 2 | 0.500 | 0.095 | ALIASED-UNRESOLVED |
| vc33 | war_eval_v1 | vc33-5430563c | HOLDOUT | 0 | - | - | ALIASED-UNRESOLVED |
| vc33 | war_eval_v2 | vc33-5430563c | TRAIN | 2 | 1.000 | 0.342 | ALIASED-UNRESOLVED |
| vc33 | war_eval_v3 | vc33-5430563c | HOLDOUT | 22 | 1.000 | 0.851 | ALIASED-UNRESOLVED |
| vc33 | war_v2_eval_s1 | vc33-5430563c | TRAIN | 2 | 1.000 | 0.342 | ALIASED-UNRESOLVED |
| wa30 | phase1_v5 | wa30-ee6fef47 | TRAIN | 122 | 0.689 | 0.602 | ALIASED-UNRESOLVED |
| wa30 | sched_v1 | wa30-ee6fef47 | HOLDOUT | 48 | 0.750 | 0.612 | ALIASED-UNRESOLVED |
| wa30 | seed1 | wa30-ee6fef47 | TRAIN | 138 | 0.688 | 0.607 | ALIASED-UNRESOLVED |
| wa30 | w0_eval_s1 | wa30-ee6fef47 | HOLDOUT | 29 | 0.759 | 0.579 | ALIASED-UNRESOLVED |
| wa30 | war_eval_v1 | wa30-ee6fef47 | TRAIN | 8 | 1.000 | 0.676 | ALIASED-UNRESOLVED |
| wa30 | war_eval_v2 | wa30-ee6fef47 | HOLDOUT | 14 | 0.643 | 0.388 | ALIASED-UNRESOLVED |
| wa30 | war_eval_v3 | wa30-ee6fef47 | TRAIN | 11 | 1.000 | 0.741 | ALIASED-UNRESOLVED |
| wa30 | war_v2_eval_s1 | wa30-ee6fef47 | HOLDOUT | 30 | 0.967 | 0.833 | ALIASED-UNRESOLVED |
| ar25 | phase1_v5 | ar25-0c556536 | - | 51 | 1.000 | 0.930 | CLEAN |
| ar25 | sched_v1 | ar25-0c556536 | - | 5 | 1.000 | 0.566 | CLEAN |
| ar25 | w0_eval_s1 | ar25-0c556536 | - | 16 | 1.000 | 0.806 | CLEAN |
| ar25 | war_eval_v1 | ar25-0c556536 | - | 6 | 1.000 | 0.610 | CLEAN |
| ar25 | war_eval_v2 | ar25-0c556536 | - | 0 | - | - | CLEAN |
| ar25 | war_eval_v3 | ar25-0c556536 | - | 4 | 1.000 | 0.510 | CLEAN |
| ar25 | war_v2_eval_s1 | ar25-0c556536 | - | 0 | - | - | CLEAN |
| bp35 | phase1_v5 | bp35-0a0ad940 | - | 106 | 1.000 | 0.965 | CLEAN |
| bp35 | sched_v1 | bp35-0a0ad940 | - | 118 | 1.000 | 0.968 | CLEAN |
| bp35 | seed1 | bp35-0a0ad940 | - | 112 | 1.000 | 0.967 | CLEAN |
| bp35 | w0_eval_s1 | bp35-0a0ad940 | - | 238 | 1.000 | 0.984 | CLEAN |
| bp35 | war_eval_v1 | bp35-0a0ad940 | - | 23 | 1.000 | 0.857 | CLEAN |
| bp35 | war_eval_v2 | bp35-0a0ad940 | - | 230 | 0.991 | 0.969 | CLEAN |
| bp35 | war_eval_v3 | bp35-0a0ad940 | - | 54 | 1.000 | 0.934 | CLEAN |
| bp35 | war_v2_eval_s1 | bp35-0a0ad940 | - | 223 | 0.996 | 0.975 | CLEAN |
| ft09 | phase1_v5 | ft09-0d8bbf25 | - | 0 | - | - | CLEAN |
| ft09 | sched_v1 | ft09-0d8bbf25 | - | 5 | 1.000 | 0.566 | CLEAN |
| ft09 | seed1 | ft09-0d8bbf25 | - | 80 | 1.000 | 0.954 | CLEAN |
| ft09 | w0_eval_s1 | ft09-0d8bbf25 | - | 0 | - | - | CLEAN |
| ft09 | war_eval_v1 | ft09-0d8bbf25 | - | 7 | 1.000 | 0.646 | CLEAN |
| ft09 | war_eval_v2 | ft09-0d8bbf25 | - | 0 | - | - | CLEAN |
| ft09 | war_eval_v3 | ft09-0d8bbf25 | - | 0 | - | - | CLEAN |
| ft09 | war_v2_eval_s1 | ft09-0d8bbf25 | - | 2 | 1.000 | 0.342 | CLEAN |
| lf52 | phase1_v5 | lf52-271a04aa | - | 0 | - | - | CLEAN |
| lf52 | sched_v1 | lf52-271a04aa | - | 0 | - | - | CLEAN |
| lf52 | seed1 | lf52-271a04aa | - | 0 | - | - | CLEAN |
| lf52 | w0_eval_s1 | lf52-271a04aa | - | 0 | - | - | CLEAN |
| lf52 | war_eval_v1 | lf52-271a04aa | - | 0 | - | - | CLEAN |
| lf52 | war_eval_v2 | lf52-271a04aa | - | 0 | - | - | CLEAN |
| lf52 | war_eval_v3 | lf52-271a04aa | - | 0 | - | - | CLEAN |
| lf52 | war_v2_eval_s1 | lf52-271a04aa | - | 0 | - | - | CLEAN |
| lp85 | phase1_v5 | lp85-305b61c3 | - | 0 | - | - | CLEAN |
| lp85 | sched_v1 | lp85-305b61c3 | - | 57 | 1.000 | 0.937 | CLEAN |
| lp85 | seed1 | lp85-305b61c3 | - | 0 | - | - | CLEAN |
| lp85 | w0_eval_s1 | lp85-305b61c3 | - | 14 | 1.000 | 0.785 | CLEAN |
| lp85 | war_eval_v1 | lp85-305b61c3 | - | 0 | - | - | CLEAN |
| lp85 | war_eval_v2 | lp85-305b61c3 | - | 0 | - | - | CLEAN |
| lp85 | war_eval_v3 | lp85-305b61c3 | - | 2 | 1.000 | 0.342 | CLEAN |
| lp85 | war_v2_eval_s1 | lp85-305b61c3 | - | 0 | - | - | CLEAN |
| ls20 | phase1_v5 | ls20-9607627b | - | 0 | - | - | CLEAN |
| ls20 | sched_v1 | ls20-9607627b | - | 35 | 1.000 | 0.901 | CLEAN |
| ls20 | seed1 | ls20-9607627b | - | 5 | 1.000 | 0.566 | CLEAN |
| ls20 | w0_eval_s1 | ls20-9607627b | - | 0 | - | - | CLEAN |
| ls20 | war_eval_v1 | ls20-9607627b | - | 0 | - | - | CLEAN |
| ls20 | war_eval_v2 | ls20-9607627b | - | 0 | - | - | CLEAN |
| ls20 | war_eval_v3 | ls20-9607627b | - | 0 | - | - | CLEAN |
| ls20 | war_v2_eval_s1 | ls20-9607627b | - | 0 | - | - | CLEAN |
| r11l | phase1_v5 | r11l-495a7899 | - | 0 | - | - | CLEAN |
| r11l | sched_v1 | r11l-495a7899 | - | 0 | - | - | CLEAN |
| r11l | w0_eval_s1 | r11l-495a7899 | - | 0 | - | - | CLEAN |
| r11l | war_eval_v1 | r11l-495a7899 | - | 0 | - | - | CLEAN |
| r11l | war_eval_v2 | r11l-495a7899 | - | 57 | 1.000 | 0.937 | CLEAN |
| r11l | war_eval_v3 | r11l-495a7899 | - | 0 | - | - | CLEAN |
| r11l | war_v2_eval_s1 | r11l-495a7899 | - | 0 | - | - | CLEAN |
| sp80 | phase1_v5 | sp80-589a99af | - | 121 | 1.000 | 0.969 | CLEAN |
| sp80 | sched_v1 | sp80-589a99af | - | 45 | 1.000 | 0.921 | CLEAN |
| sp80 | w0_eval_s1 | sp80-589a99af | - | 6 | 1.000 | 0.610 | CLEAN |
| sp80 | war_eval_v1 | sp80-589a99af | - | 65 | 1.000 | 0.944 | CLEAN |
| sp80 | war_eval_v2 | sp80-589a99af | - | 19 | 1.000 | 0.832 | CLEAN |
| sp80 | war_eval_v3 | sp80-589a99af | - | 10 | 1.000 | 0.722 | CLEAN |
| sp80 | war_v2_eval_s1 | sp80-589a99af | - | 15 | 1.000 | 0.796 | CLEAN |
| su15 | phase1_v5 | su15-1944f8ab | - | 24 | 1.000 | 0.862 | CLEAN |
| su15 | sched_v1 | su15-1944f8ab | - | 55 | 1.000 | 0.935 | CLEAN |
| su15 | w0_eval_s1 | su15-1944f8ab | - | 61 | 1.000 | 0.941 | CLEAN |
| su15 | war_eval_v1 | su15-1944f8ab | - | 4 | 1.000 | 0.510 | CLEAN |
| su15 | war_eval_v2 | su15-1944f8ab | - | 10 | 1.000 | 0.722 | CLEAN |
| su15 | war_eval_v3 | su15-1944f8ab | - | 0 | - | - | CLEAN |
| su15 | war_v2_eval_s1 | su15-1944f8ab | - | 4 | 1.000 | 0.510 | CLEAN |
| tn36 | phase1_v5 | tn36-ef4dde99 | - | 0 | - | - | CLEAN |
| tn36 | sched_v1 | tn36-ef4dde99 | - | 0 | - | - | CLEAN |
| tn36 | w0_eval_s1 | tn36-ef4dde99 | - | 0 | - | - | CLEAN |
| tn36 | war_eval_v1 | tn36-ef4dde99 | - | 0 | - | - | CLEAN |
| tn36 | war_eval_v2 | tn36-ef4dde99 | - | 2 | 1.000 | 0.342 | CLEAN |
| tn36 | war_eval_v3 | tn36-ef4dde99 | - | 0 | - | - | CLEAN |
| tn36 | war_v2_eval_s1 | tn36-ef4dde99 | - | 0 | - | - | CLEAN |
| tu93 | phase1_v5 | tu93-0768757b | - | 8 | 1.000 | 0.676 | CLEAN |
| tu93 | sched_v1 | tu93-0768757b | - | 85 | 1.000 | 0.957 | CLEAN |
| tu93 | w0_eval_s1 | tu93-0768757b | - | 30 | 1.000 | 0.886 | CLEAN |
| tu93 | war_eval_v1 | tu93-0768757b | - | 46 | 1.000 | 0.923 | CLEAN |
| tu93 | war_eval_v2 | tu93-0768757b | - | 24 | 1.000 | 0.862 | CLEAN |
| tu93 | war_eval_v3 | tu93-0768757b | - | 88 | 1.000 | 0.958 | CLEAN |
| tu93 | war_v2_eval_s1 | tu93-0768757b | - | 110 | 1.000 | 0.966 | CLEAN |

## CLEAN-game pooled certificates (Q5: splice legality inherits the Wilson standard)

No resolver was selected for CLEAN games (no selection leak), so the pooled base determinism carries the certificate.

| game | engine version | det | rep.visits | Wilson LB | prefix-splice |
|---|---|---:|---:|---:|---|
| ar25 | ar25-0c556536 | 1.000 | 184 | 0.980 | CONFIRMED |
| bp35 | bp35-0a0ad940 | 0.996 | 1534 | 0.991 | CONFIRMED |
| ft09 | ft09-0d8bbf25 | 1.000 | 232 | 0.984 | CONFIRMED |
| lf52 | lf52-271a04aa | 1.000 | 97 | 0.962 | CONFIRMED |
| lp85 | lp85-305b61c3 | 1.000 | 142 | 0.974 | CONFIRMED |
| ls20 | ls20-9607627b | 1.000 | 126 | 0.970 | CONFIRMED |
| r11l | r11l-495a7899 | 1.000 | 99 | 0.963 | CONFIRMED |
| sp80 | sp80-589a99af | 1.000 | 489 | 0.992 | CONFIRMED |
| su15 | su15-1944f8ab | 1.000 | 338 | 0.989 | CONFIRMED |
| tn36 | tn36-ef4dde99 | 1.000 | 31 | 0.890 | LB<0.95 — flagged |
| tu93 | tu93-0768757b | 1.000 | 667 | 0.994 | CONFIRMED |

Unchanged (already UNRESOLVED or LOW-SUPPORT in-sample): g50t (ALIASED-UNRESOLVED), m0r0 (ALIASED-UNRESOLVED), sk48 (ALIASED-UNRESOLVED).

## Updated consumer answers (held-out numbers are now the binding ones)

- **EWM Stage-1 safe carriers** (unchanged): ar25, bp35, ft09, lf52, lp85, ls20, r11l, sp80, su15, tn36, tu93
- **PHASE-AUGMENT / resync-viable** (held-out certified only): NONE
- **HISTORY-AUGMENT** (held-out certified history resolver; resync-before-abort NOT implied): sb26
- **EWM no-go** (unresolved, incl. holdout drops): cd82, cn04, dc22, g50t, ka59, m0r0, re86, s5i5, sc25, sk48, tr87, vc33, wa30
- **Banking prefix-splice** (CLEAN only; see pooled certificates): ar25, bp35, ft09, lf52, lp85, ls20, r11l, sp80, su15, tn36 (flagged), tu93; everything else FULL-REPLAY-ONLY from RESET, zero pruning.
- Downstream consumers (EWM phase-augment, banking keying, resurrection prong (i)) must re-point at THESE numbers per R17 checklist item 2.
