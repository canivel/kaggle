# Panel R14 component (d) — offline PREDICT->RESULT no-effect FACT metric

Zero-cost offline answer to prog-synthesis objection 4 (learnings/panel/round14/prog-synthesis.md): does the mechanical "no-effect FACT" half of component (d) have >chance predictive accuracy and enough trigger opportunities to justify its gate window (seal Jul 21)?

## Methodology

- **Rule**: IF a (key) was previously observed board_changed=False, predict board_changed=False on its next occurrence.
- **Ground truth**: recorded board_changed flag on each action event (full `board` frames also present; used as an independent frame-digest cross-check).
- **Key modes** (reported side by side):
  - `state_action` = (digest(board acted on), action_display)
  - `action_only` = action_display only
- **Trigger** = an action whose key was previously seen no-effect; = an A10 (d) firing opportunity. Trigger count per run = number of (d) firings = A10 canary opportunities.
- **Baseline** = majority-class over all actions; here no-effect is rare so majority='always effect', acc = 1 - noeff_rate.
- CPU-only, read-only over recorded traces; no engine replay, no cloud, no kernel push.

## Coverage (which traces were usable)

All discovered `*_events.jsonl` pulls carry a `board_changed` flag on every action event, so 100% of recorded actions are usable. These are recorded Qwen / duck-harness runs over the 25 official games.

| run pull | games | actions | frame-digest disagreements |
|---|---:|---:|---:|
| runs/kernel_pulls/war_eval_v1 | 25 | 3638 | 0 |
| runs/kernel_pulls/war_eval_v2 | 25 | 4026 | 0 |
| runs/kernel_pulls/war_eval_v3 | 25 | 3985 | 0 |
| runs/kernel_pulls/war_v2_eval_s1 | 25 | 4335 | 0 |
| runs/kernel_pulls/sched_v1 | 25 | 4532 | 0 |
| runs/kernel_pulls/phase1_v5 | 25 | 4418 | 0 |
| runs/phase1_ab/seed1 | 25 | 4553 | 0 |
| **pooled** | **175** | **29487** | — |

_(Frame-digest disagreements = actions where the recorded `board_changed` flag disagreed with a direct hash of consecutive `board` frames. Low counts confirm the flag is a faithful no-effect label; a nonzero count means the flag counts changes the visible grid hash does not, e.g. hidden-state or score-only changes.)_

## Pooled results

| key mode | actions | no-effect rate | triggers | recurrence acc (P no-eff again \| was no-eff) | Wilson 95% | majority baseline | games w/ >=1 trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| state_action | 29487 | 0.097 | 1147 | 0.465 | [0.436, 0.494] | 0.903 | 68/175 |
| action_only | 29487 | 0.097 | 8278 | 0.233 | [0.224, 0.243] | 0.903 | 94/175 |

## Triggers per run (feeds A10 canary: >=1/run on >=5 games)

**state_action**: mean 6.55 triggers/run, median 0, 68/175 game-runs fire >=1 trigger.

| run pull | triggers | triggers/game | recurrence acc | majority baseline |
|---|---:|---:|---:|---:|
| war_eval_v1 | 117 | 4.68 | 0.282 | 0.910 |
| war_eval_v2 | 47 | 1.88 | 0.277 | 0.956 |
| war_eval_v3 | 89 | 3.56 | 0.348 | 0.911 |
| war_v2_eval_s1 | 102 | 4.08 | 0.284 | 0.904 |
| sched_v1 | 281 | 11.24 | 0.427 | 0.900 |
| phase1_v5 | 185 | 7.40 | 0.508 | 0.903 |
| seed1 | 326 | 13.04 | 0.653 | 0.843 |

**action_only**: mean 47.30 triggers/run, median 2, 94/175 game-runs fire >=1 trigger.

| run pull | triggers | triggers/game | recurrence acc | majority baseline |
|---|---:|---:|---:|---:|
| war_eval_v1 | 1133 | 45.32 | 0.196 | 0.910 |
| war_eval_v2 | 536 | 21.44 | 0.192 | 0.956 |
| war_eval_v3 | 1164 | 46.56 | 0.205 | 0.911 |
| war_v2_eval_s1 | 1002 | 40.08 | 0.206 | 0.904 |
| sched_v1 | 1409 | 56.36 | 0.262 | 0.900 |
| phase1_v5 | 1544 | 61.76 | 0.218 | 0.903 |
| seed1 | 1490 | 59.60 | 0.306 | 0.843 |

## Verdict

**KILL.** Component (d)'s no-effect FACT rule should be KILLED CHEAPLY NOW: its best recurrence accuracy is only 0.465 — FAR BELOW the majority-class baseline 0.903 (predicting 'always effect'). When a (state,action) or action key recurs after a no-effect, the board in fact changes ~54% of the time, so the (d) rule is actively wrong most times it fires. The recorded Qwen streams are near-deterministic engines (N5 audit: 0/25 divergent) yet the no-effect label almost never recurs for the same (state,action) context — no-effects here are one-off / context-transient, not a stable 'this never works' property a memorised FACT could exploit. Emitting a PREDICT line before repeating an action family would be graded worse than a trivial 'it will have an effect' constant, so the wiring adds latency and grading surface for negative expected value. Do not open (d)'s gate window.
