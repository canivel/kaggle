# (a) Budget sentinel — build report (2026-07-19)

Component (a) of the war-v3 conversion stack (grinder_cracking_design.md §(a),
§3, §5 "W2 (a)" row). LOCAL BUILD ONLY — nothing pushed. The push decision
belongs to the main loop after panel R15. A6 build deadline (Jul 20) met.

## What was built

| deliverable | path | status |
|---|---|---|
| runtime graft | `duck_eval/sentinel/budget_sentinel_patch.py` | done |
| dataset copy (arc-war-kit staging) | `duck_eval/warpack/_kaggle_dataset/budget_sentinel_patch.py` | byte-identical copy |
| smoke test | `duck_eval/sentinel/sentinel_smoke.py` | **29/29 PASS** |
| `--sentinel` notebook mode | `duck_eval/warpack/build_eval_notebook.py` | done |
| assembled eval notebook | `notebooks/ducksentinel-eval/arc3-duck-sentinel-eval.ipynb` (+ `kernel-metadata.json`) | built, structurally verified |
| A10 compressed-budget canary | `duck_eval/sentinel/compressed_canary.py` | **canary PASS, R15 O5 predicate PASS** |
| canary artifact | `runs/sentinel_canary.json` | written |

### Mechanism (as implemented)

Two monkeypatches on the REAL bundled harness (same class-level patch mechanism
as `ledger_patch.py`, so the pickled `bm.solver` picks them up at call time):

1. `_HarnessGameSession._execute_action` (harness-side, has both the per-game
   budget `solver.max_actions_per_game` and the actions spent `action_count`):
   after each executed action, computes the fraction of the per-LEVEL-ATTEMPT
   budget consumed. On the first crossing of each registered threshold
   (default 50% / 75% / 90%) for the current attempt it (a) emits ONE countable
   `SENTINEL` event (greppable stdout line + per-game
   `<run_stem>_sentinel_events.jsonl` sidecar), and (b) queues ONE budget-state
   FACT for that game.
2. `ToolAgent.analyze` binds the per-game store path; `ToolAgent._build_user_prompt`
   drains any pending FACT and appends it to the NEXT prompt exactly once, then
   discards it. **The FACT is appended ONLY on crossing turns** — on every other
   turn the prompt is returned unchanged (zero token tax; avoids the war-v2
   always-on-injection failure mode, 1552 digests / 0 escalations).

A fresh level attempt (level number advancing, or a GAME_OVER restart) re-arms
all thresholds and restarts the budget clock, so a multi-level game gets a fresh
budget warning per attempt.

Per-game store keyed by the runtime-state **filename stem** (`<game_id>_p<pass>`),
NOT the parent dir — this avoids ledger v1's bug where parent-dir keying shared
one store across all concurrent games (panel R12 N6). The module lock guards the
shared registry dict ONLY; every call-out (FACT build, event emit, sidecar I/O)
happens OUTSIDE the lock (the deadlock-that-scored-0.00 lesson).

Failure policy: any apply() failure → returns False → notebook cell continues as
VANILLA duck (stock harness, never 0), matching the warpack/continuation graft.

## Smoke results — 29/29 PASS

`uv run python duck_eval/sentinel/sentinel_smoke.py` → `RESULT: 29 passed, 0 failed`
(includes the subprocess kill-switch arm, 3/3). Also passes with
`--warkit duck_eval/warpack/_kaggle_dataset` (the real dataset load path).

Coverage: structural parity of the assembled notebook (S1–S6, cells 2 & 12 are
the ONLY cells differing from the raw `arc3-duck-war.ipynb` source), cell-2 and
cell-12 real-source execution (I1/I2 — banner printed, `bm.label` stamped, no
warpack/ledger imported), and the mechanism against the REAL bundled
`_HarnessGameSession`/`ToolAgent` classes (M0–M4): three threshold events fire
at 50/75/90%, events carry game/action_num/threshold, per-game sidecar written,
FACT injected on the crossing turn and NOT re-injected the next turn, a fresh
level attempt re-arms and re-fires.

## A10 compressed-budget canary — PASS

`uv run python duck_eval/sentinel/compressed_canary.py` — replays the recorded
Qwen duck-harness action streams (`runs/kernel_pulls/war_eval_v{1,2,3}`, 25
games each) through a compressed **per-level-attempt budget B=60** (≈40% of the
Qwen observed median max-actions/game ≈117–165), driving the ACTUAL
`_GameBudget` logic from the patch (not a reimplementation).

Compressed budget=60/level-attempt, thresholds=50%/75%/90%:

| run | canary | fired_games | budget_deaths | predicate_violations |
|---|---|---:|---:|---:|
| war_eval_v1 | PASS | 25/25 | 12 | 0 |
| war_eval_v2 | PASS | 23/25 | 18 | 0 |
| war_eval_v3 | PASS | 24/25 | 19 | 0 |

Per-game firing counts on the 8 target games (`fN` = firings, `dN` = budget
deaths; all `V`iolations = 0):

| game | v1 | v2 | v3 |
|---|---|---|---|
| ft09 | f3/d0 | f0/d0 | f1/d0 |
| ka59 | f4/d0 | f3/d0 | f4/d1 |
| lp85 | f3/d1 | f0/d0 | f6/d2 |
| re86 | f10/d2 | f6/d0 | f7/d2 |
| sb26 | f3/d0 | f6/d2 | f4/d1 |
| sc25 | f5/d0 | f9/d0 | f4/d0 |
| su15 | f2/d0 | f4/d0 | f0/d0 |
| tu93 | f2/d0 | f2/d0 | f12/d0 |

(The 3 non-firing target-game cells — lp85 v2, ft09 v2, su15 v3 — are games that
never spent ≥50% of a 60-action attempt in that seed, i.e. short runs; they are
not violations, they simply had no threshold to cross.)

### Verdicts

- **A10 canary (≥5 games fire on EVERY run): PASS** — 23–25 games fire per run,
  far above the ≥5 threshold.
- **Panel R15 O5 mechanism predicate — "sentinel fired before every budget
  death": PASS** — across all 3 seeds there are **49 budget-attributable
  GAME_OVERs** (a recorded GAME_OVER whose level-attempt consumed ≥ B, i.e. the
  model overran the budget) and **0 violations**: every one had a sentinel
  threshold firing on a STRICTLY earlier action in the same attempt. Worked
  example — lp85 v1: firings at actions 37/52/61, budget death at action 68 (the
  90% warning at action 61 preceded the death). The check is not vacuous
  (negative path validated: a synthetic death whose only firing is on the death
  action itself is correctly flagged as 1 violation).
- **Secondary (binomial fallback): pooled firing units = 72/75** (game,seed)
  pairs fired ≥1 — a one-sided exact binomial remains available if the paired
  primary prong is underpowered.

## Exact banner strings (for post-run build-log verification)

The dataset version that actually ran is provable only from these stdout lines
(feedback_kaggle_dataset_code_sync). Grep the kernel build log for:

- Graft applied (printed by `apply()` itself):
  `sentinel v1: budget sentinel ACTIVE (thresholds=50%/75%/90%; FACT injected on crossing only)`
- Cell-12 echo:
  `sentinel v1: (a) budget-sentinel graft applied from <dir> (applied=True); NO warpack/ledger`
- Cell-2 seed banner:
  `sentinel-eval: SEED=1 (a) budget sentinel ON, NO warpack (pairs with the prior-stack seed 1)`
- Each threshold firing (countable trigger event; grep anchor `SENTINEL `):
  `SENTINEL v=1 kind=budget_threshold game=<gid> action_num=<n> threshold=0.90 pct=90 budget=<B> remaining=<r> attempt=<a>`
- `bm.label` gains the suffix `-sentinel-v1`.

Kill switch (`SENTINEL_DISABLE=1`): none of the above print; harness left
unpatched (vanilla duck).

## Design decisions recorded (minimal-intervention choices where the doc was silent)

1. **Budget source.** The per-game budget is `solver.max_actions_per_game`. If
   it is `None` (uncapped run), the sentinel emits nothing and is a silent
   no-op — it cannot know remaining budget. **Open risk (see below).** An env
   override `SENTINEL_BUDGET=<int>` forces a budget (used by the smoke/canary).
2. **Budget unit = per LEVEL ATTEMPT, not per game.** The harness `should_stop`
   caps `action_count` per game, but a budget death wastes the *level attempt in
   progress*; firing per attempt gives the model a fresh warning after each
   level-up/restart. Chosen as the intervention that maps to the death events
   the doc cites (lp85 L2 attempt, tu93 L1 attempt).
3. **Thresholds 50/75/90%** (override `SENTINEL_THRESHOLDS`). 90% leaves ~10% of
   the budget for the model to convert an in-progress level after the last
   warning; 50/75% are early wind-down nudges.
4. **Attempt-base off-by-one.** On a level-up the change is detected ON the
   first action of the new attempt, so that action counts toward the fresh
   budget (base = action_num − 1). A GAME_OVER restart action is itself the
   terminal wasteful action, so the fresh attempt starts after it (base =
   action_num). Death attribution in the canary is measured against the
   pre-restart base so a GAME_OVER is attributed to the attempt that overran.
5. **Events channel.** Primary countable channel = greppable `SENTINEL` stdout
   line (EVENT_SCHEMA.md convention: anchor token + fixed key=value tokens,
   ASCII, one event/line), which the canary parses; plus a best-effort per-game
   `<run_stem>_sentinel_events.jsonl` sidecar next to the viewer data. (I did
   NOT reuse the viewer `_events.jsonl` sidecar — that is the harness's own
   trace; a separate `_sentinel_events.jsonl` keeps the sentinel's events
   cleanly attributable and avoids mutating harness artifacts.)
6. **preflight.py:** N/A for this notebook family. `scripts/preflight.py` targets
   the `agents/`-swarm baseline (`arc3-baseline`) and REQUIRES a live Kaggle
   kernel pull (K1); this duck-eval notebook is the taaf-harness structure and
   is unpushed. Structural drift is instead guarded by the smoke's S1–S6 checks
   (17 cells; only cells 2 & 12 differ from the raw source; metadata delta =
   exactly {id, title, code_file}) — the same guard the W0 (f) build used.

## Open risks

1. **Uncapped live budget.** If the scored Kaggle run leaves
   `max_actions_per_game=None`, the sentinel is inert (no budget to warn
   against) and the whole component silently does nothing — the run would look
   like a vanilla duck. The banner still prints ACTIVE (the patch installed),
   but no `SENTINEL` events would appear. **Mitigation / verification for the
   main loop:** before sealing, confirm the eval harness sets
   `max_actions_per_game` (grep the build log for `SENTINEL v=1` events; zero
   events on a run known to have long games = the budget is unset, and
   `SENTINEL_BUDGET` must be exported in cell 2 to a value matching the intended
   compressed cap). This is the single highest-value pre-seal check.
2. **Firing ≠ paying.** Per §2(a) the honest expectation is +0.01–0.03/draw and
   the two canonical grinders (sb26, lp85) carry expected Δclears of ZERO at
   Qwen tier — the sentinel supplies awareness, not the missing concept. The
   canary proves the mechanism FIRES and warns before every budget death; it
   does NOT prove the model acts on the warning. If the sealed 3-seed gate shows
   the score prongs fail with the mechanism prong firing, the honest label is
   "mechanism fires, doesn't pay" (A10 guarantees no regime excuse).
3. **FACT wording is a prompt, and prompts are noise** (feedback_prompt_is_noise).
   The value is the *presence/timing* of the budget signal, not its phrasing; do
   not A/B the wording.
4. **Compressed-budget canary uses recorded traces**, whose engine versions may
   differ from the Kaggle build (15/25 local engines drift). The predicate is a
   structural check on action/GAME_OVER positions, which is robust to frame
   drift, but the exact death counts are trace-specific.

---

## ADDENDUM 2026-07-22 — R16 Q2 discharge: v2 unit re-key + W1 prong context-tax

**Everything above describes sentinel v1 (per-LEVEL-ATTEMPT unit). Panel R16
(C4, 5/5 reviewers; llm-agents N11 MAJOR) ruled the unit conflated: the 63k-token
envelope that derives B=150 is per GAME, and attempt re-arming made v1 warn late
or never in multi-attempt grinder games — (a)'s target population.** The
attempt-unit analysis (`runs/sentinel_attempt_unit_b150.json`, from the certified
seeds) confirmed the defect: the carrier games ka59/re86/tu93 are multi-attempt
in EVERY certified seed (attempt counts ka59 2–3, re86 3–4, tu93 7–12); 15 of 33
envelope-crossing (game,seed) units got no v1 warning by 0.9×B cumulative; 13
cross-attempt-waste episodes (game total > 150, no attempt > 75) were
structurally invisible to v1, including tu93 twice. The single-attempt
approximation FAILS on the carriers → **re-key mandated and done.**

**v2 (current code, both copies byte-identical):** budget fraction = CUMULATIVE
game actions / per-game budget; each threshold fires at most once per GAME (hard
cap 3 events/game); level-attempt boundaries are tracked only as event metadata
(the `attempt=` field). Banner/event format preserved (`SENTINEL v=2 ...`, same
key=value tokens; banner adds `unit=game-envelope`; label suffix `-sentinel-v2`).
Smoke re-run post-re-key: **30/30 PASS** (v1's suite updated for v2 semantics:
M4 now asserts NO re-arm on a fresh attempt, M4b asserts cumulative counting
across an attempt boundary, M4c asserts events carry the attempt ordinal as
metadata; kill-switch arm 3/3 included). Canary v3 (`runs/sentinel_canary_v3_b150.json`): A10 PASS (20/25
games fire per run), R15 O5 predicate PASS (54 budget deaths, 0 violations),
and the R16 defect-sensitive counter — all 13/13 cross-attempt-waste episodes
now fire at cumulative action 75 (50% threshold), well before 0.9×B=135;
59/75 units multi-attempt (≥1 required).

### W1 prong — context-tax sentence (R16 Q2 condition 3, systems #3)

Tokens-per-fire is ≤ ~95 tokens (worst-case FACT measured at 370 chars / 62
words; ≤124 tokens even at a conservative 3 chars/token bound), and because v2
fires each threshold at most once per GAME the worst-case cumulative injection
is 3 fires × ~95 ≈ 285 tokens (≤372 conservative) ≈ **0.45% (≤0.59%) of the 63k
per-game context envelope**, even if all three FACTs persist in history for the
rest of the game; the v1 scenario systems #3 asked about — 3 fires/attempt
across a multi-attempt game, worst observed 14 attempts (sp80 v1) → up to 42
fires ≈ 3.9–5.2k tokens ≈ 6.2–8.3% of envelope — is structurally impossible in
v2 (no re-arm).
