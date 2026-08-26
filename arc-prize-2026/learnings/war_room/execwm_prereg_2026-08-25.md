# PREREG — EXEC-WM (arm `execwm`, kernel `canivel/arc3-execwm-eval`), SEALED 2026-08-25, pre-push

**Program:** executable-world-model lane (principal's direct order 2026-08-25: *"lets just build what we need to get to first now"*). Mechanism class established by three independent ~99–100% systems (NVIDIA AVO; VISTA, vista-research.github.io; Schema, schema-harness.github.io — self-reported, not ARC-Prize-verified) and by arXiv **2605.05138** (*Executable World Models for ARC-AGI-3 in the Era of Coding Agents*): **write each game's mechanics as an executable program, verify it against recorded history, plan inside it with search.** Community brief 2026-08-25 finds 2/3 (lawbook prior-kills; the "refutation against recorded history" pattern).
**Sealed by:** exec-wm build lane, 2026-08-25, after the CPU loop was proven and **before any Kaggle build**. No GPU numbers for this arm exist.
**Vehicle drift ruling inherited:** anim-20260807 is the vehicle; `bundle_20260815` anchors are NOT valid here (P2's 233-diff-line lesson).

---

## 0. THE QUESTION, AND WHAT ALREADY DECIDED THE DESIGN

**Question.** The floor's binding constraint is the decision budget (17 acting turns/game vs 132 designed; 6.4k–9.6k tokens/action), while actions are nearly free (ε = 0.17; T3 bought 3× actions for +0.04). Does moving the inner loop into a **deterministic executable world model** — scripted exploration, mechanically mined + prequentially verified per-action object rules, BFS planning with per-step settled-frame checks, and **per-level fallback to the untouched stock agent** — convert free actions into levels on the certified floor?

**Priors folded in (not re-opened):**
- lawbook (647 real-27B actions): board-keyed memoization DEAD (2.6%/47%), global effect signatures DEAD, **object-level laws 83.7–92.4% precision under run-gating** ⇒ the program is object-centric (sprite pattern + per-action translate/no-op rules + permeability), never board-hash-keyed.
- P0.1 [V-sim]: RESET returns to level start under the shipped flag ⇒ cheap experiments are legal. P0.3/P0.4: full-grid comparison provably cannot fire; **HUD strips are excluded from every comparison** (HudMask lineage).
- exp 36 (0-for-6 sweep): verbose injected obligations HARM ⇒ the model-facing surface adds **zero prompt text**; the stock agent's prompt is byte-untouched. The LLM is called only by the controller's PHASE I, off-band, ≤ 2 calls/game.
- perturn program: the agent already reads `transitions` (16.3%) but never searches over them ⇒ the search lives HARNESS-SIDE, costing zero LLM tokens per action.

**Strongest pre-data evidence AGAINST, registered now.** (i) The 24-sim exec_wm corpus is in-sample-fit evidence only (91.7% is a class share; on-trajectory transfer had median range 0.400) — v1's miner is therefore verified **prequentially at runtime**, never trusted from history. (ii) On CPU, of 5 real games driven, only **ls20** cleared a level via plan; bp35/tu93 fell back with no verified model, sp80 verified 2 moves then broke (multi-object dynamics out of v1 scope). The mover-with-blockers class may be a minority of the 25 public games. (iii) Pellegrin's field datum: local gains can buy zero LB. **P(SIGNAL) ≈ 15–25%.** The floor is protected by construction: every non-modelled level runs stock.

---

## 1. VEHICLE AND TREATMENT

- Base: certified field floor `notebooks/q38-field-eval/arc3-q38-field-eval.ipynb` (11 cells), **byte-untouched**.
- Treatment: **ONE inserted patch cell at position 6** (P1/P2 slot) ⇒ 12 cells. Staged: `notebooks/execwm-eval/arc3-execwm-eval.ipynb`, code sha `87fbe2feed3f2fd8`, embedded module sha `d5c4b85872d5bba5` (= sha256[:16] of `duck_eval/execwm/exec_wm.py`, the single source of truth; the cell asserts the embedded bytes match at boot).
- The cell: copies the mounted anim-20260807 bundle to `/kaggle/working/execwm_patched/`, writes `inference/agent/exec_wm.py` (new file, zero anchor risk), applies **one** anchored replacement in `inference/framework/solver.py` (`"    def _make_analyzer(\n"`, asserted count==1, verified =1 in the vehicle; drift ⇒ loud INFRA DEATH), shadows the path, asserts module resolution, prints `[execwm] patch applied sha=…`.
- Kill-switch: `ARC3_EXECWM=0` ⇒ byte-floor behaviour (smoke S3). `ARC3_EXECWM_LLM` defaults on.
- Kernel metadata: dataset/model/docker/machine byte-identical to the floor's; fresh slug `canivel/arc3-execwm-eval`, `is_private: true`. Pull-back verify incl. **model_sources EXACT**.

### Sealed parameters (constants in `exec_wm.py`; not tunable post-data)
E_REPEATS=4 · E_MAX_TURNS=4 · PROBE_BATCH=10 · PLAN_BATCH=24 · MASK_MIN_PAIRS=8 · MASK_RATE=0.5 · MASK_BORDER=6 · MAX_DELTA=8 · MIN_SPRITE_CELLS=2 · VERIFY_MIN_N=3 · VERIFY_PRECISION=0.90 · MIN_VERIFIED_MOVES=2 · MAX_BREAKS_PER_LEVEL=3 · MAX_GOAL_COLORS=6 · GOAL_MAX_CELLS=30 · MAX_SWEEP_PLANS=48 · MAX_PLANS_PER_LEVEL=96 · LLM_CALLS_PER_GAME=2 · LLM_TIMEOUT_S=180.

Two design rulings made ON CPU EVIDENCE, recorded as part of the seal: (a) a consecutive-tail acceptance clause was tried and **rejected** (one ~1/43 ls20 counter-teleport permanently killed a 30-for-31 rule and halved connectivity); acceptance is n ≥ 3 ∧ precision ≥ 0.90, with **unexplained-class misses excluded from the count as world events** (they instead feed the event counter). (b) blockers are re-mined from **post-event no-ops only**, and probes re-arm on each new unexplained event with previously-refused probes retried first (the ls20 key/door pair is unreachable otherwise; proven by the S4 clear).

---

## 2. THE LOOP (all deterministic except PHASE I's optional LLM half)

Per game (analyzer wrapper `ExecWMAnalyzer` around the stock ToolAgent; solver reads `.generated_tokens` through it):
- **Triage:** games offering no keyboard action (mouse-only) are disabled game-wide ⇒ stock agent, `[execwm] disabled` logged.
- **PHASE E** — scripted probes: each candidate action pressed E_REPEATS times (≤ PROBE_BATCH env actions per controller turn, zero LLM round-trips).
- **PHASE I** — deterministic miner: per-transition translation detection (full interior-diff coverage as departures+arrivals; symmetric ambiguity broken by color RARITY; sprite = most-frequent pattern across move instances); underlay + permeable + blocker sets. LLM half fills ONLY unexplained actions: one lean chat call, constrained JSON, parsed strictly, injected as CANDIDATES.
- **PHASE V** — prequential replay of every recorded same-level transition; a rule below threshold is never used. LLM-filled and mined rules verify identically.
- **PHASE P** — BFS over sprite positions in the verified program; targets in order: rare-color goals (≤ GOAL_MAX_CELLS) → coverage sweep of every unvisited reachable position → frontier probes (one-action experiments past the verified boundary, rarest unknown color first, refused-first after events). Every plan step: predict → act → compare settled interior; a clean mispredict = BREAK (≥ MAX_BREAKS_PER_LEVEL ⇒ fallback), an unexplained mispredict = EVENT (plan aborted, probes re-armed, no break charged).
- **FALLBACK** — per level, latched: explore budget exhausted / no verified model / breaks / targets exhausted / any controller exception ⇒ `inner.analyze(...)` verbatim. **Fallback = the certified floor behaviour.**

## 3. RUNTIME CERTIFICATION (any failure ⇒ INFRA DEATH, never NULL)

Sealed scorer `duck_eval/execwm/execwm_score.py` (selftest 15/15; healthy-fixture positive control per the arm-mismatch lesson; cross-arm negatives pass in `local_gate --arm execwm` A2):
- REQUIRED in log: `Qwen/Qwen3.8-27B-FP8` · `anim-20260807` · `[execwm] armed`.
- FORBIDDEN: `[notes] persistent-namespace armed`, `[p2] reset semantics OK`, `[goalkeep]/[clickmap]/[banking] armed`, `TAAF_GRAFTS FEATURES`, `PRIVATE-ARM BANNER`.
- A 0-byte log (P1 class) ⇒ INFRA DEATH, certification undefined — **but delivery instruments do not depend on the log**: the wrapper writes per-game JSON reports to `job_dir/execwm/*.json` (pulled with the artifact), which the scorer reads first.

## 4. DELIVERY — read BEFORE the effect; it has its own verdict class

From the report files (log markers only as fallback):
- **D1 armed:** reports present for ≥ 20/25 games (wrapper constructed). Fewer ⇒ DELIVERY FAILURE.
- **D2 modelled:** ≥ 3 games reach PHASE P (a verified model + ≥1 executed plan). 0–2 ⇒ **DELIVERY-LIMITED**: the lc read is FLOOR-EQUIVALENT-descriptive; no mechanism verdict may be issued, and the next step is per-game triage analysis, not a re-run.
- **D3 conversion:** `levels_cleared_by_plan ≥ 1`. If ≥ 10 levels were planned and 0 cleared ⇒ **mechanism-refuted for the v1 rule class** (that IS a verdict; the lane re-scopes to rule-class v2, no same-class re-run).
- **D4 graceful degradation (the assert):** if fallback rate = 100% (no level ever planned), lc MUST land within the comparator's noise (|lc − 29.0| ≤ 5.6). Outside it ⇒ the wrapper perturbed the floor ⇒ INFRA investigation before any other read. CPU S2 already asserts the wrapped floor still wins the scripted game.
- Descriptives always reported: per-phase probes, rules mined/verified/rejected, plans, breaks vs events, llm_calls (≤ 50 total), wm_actions, fallback reasons.

## 5. SEALED READ (seed 1)

- **Comparator (re-derive at read time, never cached):** local-rail lc series = field floor **28** + Arm A base **30** ⇒ mean **29.0**, pooled seed sd **2.80** (identical to the P1/P2/C-family comparator). Board-side context only: field-floor config n=5 mean 1.5760 sd 0.2713 (no board read from one kernel).
- **Primary:** `lc_total`. **Co-primary:** `trim1` (raw mean_score retired: 50.4% one game).
- **Bands (±2σ on n=1):** **HARM ≤ 23** · **NULL 24–34** · **SIGNAL ≥ 35.** A SIGNAL additionally requires D2 delivered AND `levels_cleared_by_plan ≥ 3` (the lift must be attributable to planning, not draw noise — `feedback_seed_vs_own_config`).
- **Decisive kill, pre-stated:** D2 delivered on ≥ 5 games with `levels_cleared_by_plan = 0` AND lc ≤ 23 ⇒ the v1 exec-WM class is dead on this rail; no parameter-tweak re-run may be authorized against this prereg.
- Certification failure ⇒ INFRA DEATH ⇒ re-run-not-advance rules apply.

## 6. CPU EVIDENCE ALREADY BANKED (why a slot is justified)

`duck_eval/execwm/ewm_smoke.py` — **22/22 PASS** on the real wheels (arc_agi 0.9.6 + arcengine 0.9.3 + taaf) with the real HarnessSolver/ToolAgent from a patched vehicle copy and a stdlib stub LLM:
- **S4: ls20 level 1 CLEARED VIA PLAN with ZERO LLM tokens** — 4/4 move rules mined and verified at precision 1.0 (deltas ±5 rows/cols), lane color learned permeable by frontier probe, key/door solved by event-re-armed refused-first probing (~108 actions vs human 21; actions ε=0.17).
- S2/S3: wrapped scripted game still WON by the stock agent (graceful degradation); kill-switch leaves stock untouched.
- S5: bp35 (non-mover) correctly falls back; stock agent observed running (363 stub round-trips).
- S8: the verifier REFUSES a deliberately wrong program (flipped deltas all rejected) — the gate that keeps a weak coder honest can actually refuse.
- S1/S1b/S1c: anchored patch applies once, double-apply and drifted-vehicle die loudly.
- `local_gate --arm execwm`: fast **PASS 43/0** (full run in progress at seal time; push conditional on full PASS).

## 7. ORDER, SLOTS, COST

- This arm takes **no submission and no queue edit** (daemon owns the window; Arm 0 banks the floor nightly).
- Slot arithmetic 2026-08-25: 0/2 kernel slots spent today; **P2 holds the standing slot-1 claim for 08-26** (lane lock). exec-WM pushes **only into a slot the coordinator confirms free** — if both 08-26 slots are owed (P2 seed 1 + its contingencies), exec-WM is push-ready and waits. Registered in `runs/lane_locks.json` at take-time.
- Cost: 1 kernel build (~8 h GPU). LLM budget inside the run: ≤ 2 calls × 25 games, 512 max_tokens each — noise against the run's ~2M generated tokens.
- Wall-clock risk priced: controller turns are LLM-free; worst case ≈ 2.3k extra env actions/game ≈ minutes against the 7,920 s clock.

## 8. WHAT THIS ARM CANNOT SETTLE

Whether the exec-WM CLASS (all three 99–100% systems used frontier models with free-form program synthesis) works on a 27B — v1 tests only the deterministic-miner + translate-rule instantiation with an LLM gap-filler. A NULL with delivered D2 kills v1's rule class, not the class; a SIGNAL licenses rule-class v2 (multi-object dynamics, ACTION6 games, LLM-proposed rule forms), each behind its own prereg.

## 9. ADVERSARIAL NOTE (my own strongest objection)

The CPU proof leaned on ls20's obliging structure (unique 2-color sprite, cardinal ±5 moves, singleton blockers) and on iterating the design against ls20 until it cleared — the design choices in §1(a)/(b) are therefore ls20-fitted to a degree, and the only true out-of-sample test is the other 24 games on the real rail. That is exactly what the slot buys; the per-level fallback bounds the downside at the floor minus noise, which D4 asserts mechanically.
