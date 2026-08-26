# EWMEVT — event-shaped canary schema for the EWM-execute line

Status: v1, filed 2026-07-18. Corrected adopt #1 from the Kimi-3 review cycle.
Producer: the (future) plan-execute-verify executor (OPINE §3.5 contract on the
12 saturated `exec_wm/` sims; `learnings/war_room/opine_world_deepread.md`
Stage 1, `state_of_the_war_2026-07-18.md` priority 1).
Consumer: `scripts/ewm_events.py` (aggregator + A10 canary + GSME activation
prong-0 verdicts). Dry-run producer: `scripts/ewm_replay_dryrun.py` (replays
recorded action streams through the saturated sims — no executor build needed).

## Why event-shaped, not totals-shaped

`scripts/predict_metric.py` (R14 artifact) proved totals-shaped counters are
enough to KILL a component ((d): pooled recurrence accuracy vs baseline). They
are NOT enough to GATE the EWM executor, whose Stage-0 mechanism prong is
"plans executed >=1/run on >=5 games, mismatch-aborts logged, 0 post-abort
deadlocks". Totals cannot express:

1. **Per-game plan-step rates** — a total step count cannot distinguish
   "one 40-step plan on ls20" from "40 one-step plans scattered anywhere".
2. **Mismatch-abort survival** — whether the agent RECOVERS after an abort
   (another plan or a live fallback follows) or DEADLOCKS (abort is terminal)
   is an *ordering* property of the stream, invisible to any counter.
3. **Plan-length / abort-step distributions** — where in a plan divergence
   lands (step 1 = sim junk or engine-version drift; step 15 = deep-horizon
   drift) decides whether the sim is shippable. A totals counter collapses
   this to a single number.

One greppable stdout line per event preserves exactly the ordering and
per-game structure the gate needs, at bounded cost (see §Volume).

## Grammar

One event per stdout line. ASCII only. No spaces inside values.

```
EWMEVT v=1 kind=<kind> game=<gid> [k=v ...]
```

- First token is the literal `EWMEVT` (grep anchor; a prefix such as a
  timestamp before it is legal — parsers must anchor on the first occurrence
  of `"EWMEVT "` in the line, not on column 0).
- Then space-separated `key=value` tokens, order fixed as specified per kind
  (parsers MUST NOT rely on order; emitters keep it fixed for greppability).
- `v=1` = schema version; bump on any breaking change.
- `game` = 4-char game id (`ls20`), NOT the versioned id (`ls20-9607627b`);
  the versioned id appears once in `plan_start.gv`.
- Every line MUST be < 200 chars (worst legal line below is 154).
- Unknown keys MUST be ignored by parsers (forward compatibility).

### Common fields

| key | meaning |
|---|---|
| `v` | schema version (int, =1) |
| `kind` | event kind (below) |
| `game` | 4-char game id |
| `plan` | plan ordinal within this game's stream, 0-based |
| `step` | 0-based step index within the plan |
| `t` | elapsed seconds since run start, 1 decimal (`t=na` in offline replay) |

### Kinds

**`plan_start`** — a sim-guided plan begins (BFS over the sim found a path).
```
EWMEVT v=1 kind=plan_start game=ls20 plan=0 len=12 sim=ls20_sim:9f3c21ab gv=ls20-9607627b lvl=1 t=412.3
```
`len` = planned action count; `sim` = sim module name + 8-hex blake2b of its
file bytes (drift audit); `gv` = versioned game id; `lvl` = level at planning.

**`plan_step`** — one live action executed under the contract, predicted vs
settled frame compared.
```
EWMEVT v=1 kind=plan_step game=ls20 plan=0 step=3 act=A6:31,22 pred=a1b2c3d4 obs=a1b2c3d4 match=1 lvl=1 t=413.0
```
`act` = `A1`..`A5`, or `A6:<y>,<x>` (y=row, x=col — engine `data{x,y}`
convention, x=col; see `exec_wm/collect_observations.py`), or `RESET`.
`pred` = hash8 of the sim-predicted settled frame; `obs` = hash8 of the
observed settled frame (see §Hashing); `match` = `1` iff byte-identical
frames (authoritative bit — computed on full frames, NOT on the hashes).

**`mismatch_abort`** — first predicted-vs-settled mismatch (or sim error /
double-run self-disagreement); the plan is dead, mismatch is control flow.
```
EWMEVT v=1 kind=mismatch_abort game=ls20 plan=0 step=3 len=12 reason=mismatch pred=a1b2c3d4 obs=e5f60718 t=413.0
```
`reason` in `{mismatch, sim_error, selfdiff}`. `step` = divergence step.
`selfdiff` = the double-run check (two fresh sim evaluations of the same
transition disagreed) — rejects hidden-state/nondeterministic sims by
construction.

**`plan_done`** — every step of the plan matched.
```
EWMEVT v=1 kind=plan_done game=ls20 plan=0 len=12 steps=12 lvl_done=1 t=419.9
```
`steps` = steps actually executed (== `len` on success); `lvl_done` = 1 iff
the plan's terminal action completed a level.

**`fallback`** — control handed back to the normal LLM duck loop.
```
EWMEVT v=1 kind=fallback game=ls20 plan=0 reason=mismatch t=413.1
```
`reason` in `{mismatch, sim_error, selfdiff, no_plan, budget}`; `no_plan` =
BFS found no path within bound (OPINE's own failure mode on deep games);
`budget` = per-game action/time budget guard stopped the plan.

**`trunc`** — volume guard engaged (see §Volume); emitted once per game.
```
EWMEVT v=1 kind=trunc game=ls20 dropped_after=2000 sample=10 t=6301.0
```
After this, `plan_step` lines for that game are sampled 1-in-`sample`;
`plan_start`/`mismatch_abort`/`plan_done`/`fallback` are ALWAYS emitted
(the gate reads those, so the mechanism prong is truncation-proof).

## Hashing

`hash8(frame) = blake2b(json.dumps(frame, separators=(",",":")).encode(), digest_size=4).hexdigest()`
— 8 hex chars over the canonical JSON of the 64x64 int grid. Same
serialization family as `scripts/predict_metric.py::board_digest` (which uses
digest_size=8; the canary only needs match forensics, and `match` is computed
on full frames, so 32 bits is enough for log-side disambiguation).

## Aggregator semantics (sealed here so the gate can't be argued post hoc)

Computed by `scripts/ewm_events.py` per game stream:

- **plans** = count of `plan_start`; **steps** = count of `plan_step`.
- **step accuracy** = mean of `match` over `plan_step` lines.
- **abort-step distribution** = multiset of `mismatch_abort.step`.
- **post-abort survival**: for each `mismatch_abort`, look at the SAME game's
  subsequent events; `survived` iff a `plan_start` or `fallback` occurs within
  the next N events (default N=25) of that game. (The contract emits the
  paired `fallback` immediately, so a healthy abort always survives; a
  crash/hang between abort and handoff shows up as a non-survival.)
- **deadlock**: a `mismatch_abort` with NO subsequent progress event
  (`plan_start`, `plan_step`, `plan_done`, `fallback`) for that game anywhere
  later in the log = the abort was terminal. Gate requires deadlocks == 0.
- **A10 canary** (per run): `fired_games` = games with >=1 `plan_start` whose
  plan executed >=1 `plan_step`. Verdict line:
  `EWM_CANARY games_fired=<k> threshold=5 verdict=<PASS|FAIL>`.
- **GSME activation prong-0** (arXiv:2607.13683 — activation gate before
  significance gate; the anti-0/1552-ledger check): ACTIVE iff plans >= 1 AND
  steps >= 1 AND (plan_done + mismatch_abort) >= 1 — i.e. the mechanism ran
  AND produced at least one verified terminal outcome. Verdict line:
  `EWM_ACTIVATION plans=<p> steps=<s> outcomes=<o> deadlocks=<d> verdict=<ACTIVE|INERT>`.

## Volume (10 MB Docker log cap — discussions_2026-07-18 infra constants)

Docker logs are silently truncated at 10 MB/container; the schema must stay
far under it so post-mortems survive (war-v3 summarizer already budgets
< 10 MB total).

- Longest legal `plan_step` line: 154 chars incl. `\n` (64-char prefix-stamped
  worst case still < 200). Budget at 200 B/line to be safe.
- Observed duck-harness volume: ~4,200 action events per 25-game 8h run
  (predict_metric coverage: 29,487 actions / 7 pulls). EWM plans only run on
  the 12 sim games, and only while the sim survives verification, so the
  realistic ceiling is ~2,000-5,000 `plan_step` lines/run plus < 10% overhead
  lines: **~0.4-1.1 MB typical**.
- Adversarial worst case (every action of a hyperactive run is a verified
  plan step): hard caps make it bounded by construction —
  **cap = 2,000 EWMEVT lines/game and 25,000/run**, then `trunc` + 1-in-10
  step sampling. Absolute worst: 25,000 x 200 B = **5.0 MB**, leaving >= 5 MB
  of the 10 MB cap for the rest of the harness's stdout. Without the cap the
  theoretical worst (~40k lines) would crowd the cap — the cap is therefore
  normative, not advisory.
