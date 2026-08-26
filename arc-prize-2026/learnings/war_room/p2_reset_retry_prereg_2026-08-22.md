# PREREG — P2: RESET-ANCHORED EPISODIC RETRY (`arc3-p2-retry-eval`), SEALED 2026-08-22, pre-build
**Program:** `perturn_program_2026-08-22.md` §3-P2 / §5.5 (arm rank 2 of 8), sealed here VERBATIM in mechanism and parameters, with the gate discharged and the build recipe added.
**Gate:** P0.1 (RESET semantics) — **PASSED on the real simulator, 2026-08-22.** See §2. The program's own words: *"P2 does not get built until this passes."* It passed; **P2 may be built.**
**Sealed by:** weekend-prep lane, 2026-08-22, **before any build.** No numbers from this arm exist.
**Slot:** after P1 seed 2 and the C1 cadence pair; P2 is rank-2 by expected value and is gated only on slot arithmetic, not on any other arm's result.

---

## 1. THE QUESTION AND THE RESOURCE INVERSION IT RESTS ON

**Question.** Does converting the resource we hold in **surplus** (environment actions) into the resource we **lack** (independent attempts at a level) turn dead turns into cleared levels?

The inversion is measured, not argued:

- **Actions are nearly free at the margin.** exp 39's elasticity above our operating point is **ε = 0.17**: T3 spent **2,184 extra actions** and bought **+0.04 mean_score**. A resource whose marginal value is that low is a resource we may spend freely. **[V]**
- **Independent attempts are what we do not have.** `ONLY_RESET_LEVELS=true` makes every level a restartable episode, and the agent uses it **9 times in 1,639 actions (0.55%)**. Meanwhile **229 of 424 acting turns (54.0%)** are spent on a level that is never cleared, and **56.7% of all actions** land on a never-cleared level — i.e. the dead phase is spent **mutating a progressively more broken state** rather than resampling from a known-good one. **[V]**
- **The score arithmetic pays for it exactly where the levels are.** Going from `k` to `k+1` cleared levels at action ratio `r` relative to baseline wins iff `r < sqrt((k+2)/k)`: unbounded at k=0, 1.73 at k=1, 1.41 at k=2. **20 of our 25 games sit at k ≤ 1** (8 at k=0, 12 at k=1) and our realised ratio on cleared levels is **0.90**, so the retry allowance at k ≤ 1 is **≈1.6–1.9×** and at k = 0 unlimited. **[V arithmetic, perturn §2.5]**
- **Budget price, priced pre-data:** turns **0** (episodes run inside one turn); tokens **+18% on retry turns only**, ≤54% of turns ⇒ **≈ +10% overall** ⇒ ×0.91 budget ⇒ a standing tax of **−0.8 lc** at ε = 0.17. **P2 must clear +0.8 lc to break even**, and that bar is written into the read.

**Why this is not the 0-for-6 class.** The six dead additions (exp 3, 4, 9, 33, 36, 37) each added an **obligation to every turn**. P2 adds an **affordance** and changes what a turn can *buy*. I concede in §10 that I cannot prove that distinction in advance.

---

## 2. THE GATE, DISCHARGED — RESET RETURNS TO LEVEL START [V-sim]

Driven end to end on the **real** competition wheels (`arc_agi 0.9.8` + `arcengine 0.9.3`) against the **real** environment file `ls20-9607627b`, taken to level 1 by replaying 21 recorded human actions. Script `scratchpad/p0_reset_semantics.py`; raw evidence `duck_eval/p0/p0_reset_results.json`; findings `duck_eval/p0/P0_FINDINGS_2026-08-22.md` §P0.1.

| case | config | result |
|---|---|---|
| **A** | `ONLY_RESET_LEVELS=true`, RESET after 2 actions on level 1 | **level_reset** — level index 1 kept, score kept, `full_reset=False`, returned frame **byte-equal to the recorded level-1 start frame** |
| **D** | flag true, RESET at `_action_count == 0` | level_reset, frame byte-equal to level-1 start |
| **B** | flag UNSET, RESET at `_action_count == 0` on level 1 | **FULL reset** to level 0, score 0 — the documented footgun, reproduced (this is P2's negative control, and it fires) |
| **C** | flag UNSET, RESET after 1 action | level_reset (the footgun bites only at `_action_count == 0`) |

Also verified: the level-1 start frame is **deterministic across two independent clears** (byte-equal), so "RESET re-anchors to a known frame" holds. `handle_reset` reads `os.getenv` **per call** (`arcengine/base_game.py:305-316`), `level_reset` clones only the current level and preserves `_score` and `_current_level_index` (L326-329), and **none of the 25 environment files overrides `handle_reset`/`full_reset`/`level_reset`** — base-class semantics apply everywhere. The flag is set process-wide by the shipped harness at `taaf/game_api.py:222` *and* by the deployed notebook at cell 7 (`os.environ["ONLY_RESET_LEVELS"] = "true"`) — belt and braces. **[V-source]**

**One caveat carried forward:** with the flag set, RESET **after WIN** still full-resets (`state != WIN` guard). Irrelevant to P2 (no retry after a win) and asserted against in §4.

These four cases are now **permanent regression checks**: `scripts/local_gate.py` group **P8.reset_semantics** fails the gate if the sealed evidence stops supporting P2's premise.

---

## 3. THE MECHANISM

### 3.1 `attempt(seq)` — a sandbox helper, sibling of `action()`

```
attempt(seq) -> {"level_completed": bool, "reward": float, "actions_taken": int,
                 "terminal_reason": str, "board_delta": "<~200 char summary>"}
```
Executes the action list from the **current level start**, captures the outcome, then issues a **RESET** which returns the game to that same level start, and returns the compact summary. **Costs actions; costs no turn.** One LLM turn can therefore evaluate **K independent candidate plans** instead of committing to one. At 17 turns/game and K = 5 that is **85 episodes/game against 17 today**.

### 3.2 The stuck trigger — harness-side, zero prompt text
After **H consecutive acting turns on the same level with no `level_completed`**, the `python` tool result carries one structured line:
`retry_mode: on, episodes_available: K`
No prompt paragraph, no per-turn obligation. The affordance is announced in the tool description (which is where `action()` is announced too) and its *availability* is signalled by a ~40-character structured field.

### 3.3 Parameters — fixed pre-data, verbatim from perturn §5.5
| parameter | value | why |
|---|---|---|
| **H** (stuck threshold) | **4** consecutive acting turns on one level without a clear | the productive rate is 6.96 turns/level and 3.0 turns/level on the games we actually know; 4 is inside the productive regime's tail, so retry arms while the level is still plausibly winnable, not only after it is hopeless |
| **K** (episodes per retry turn) | **5** | 17 turns/game × 5 = 85 episodes; at ≤40 actions each the worst case is bounded by §3.4 |
| episode length cap | **40 actions** | keeps one retry turn inside the §2.5 allowance on a k ≤ 1 game |
| retry disabled once | **k ≥ 4** on that game | the allowance tightens to r < 1.29 and the quadratic starts eating the depth |
| RESET-after-WIN | **never** | the engine full-resets after WIN; `attempt()` must refuse to RESET when the episode ended in a level clear, and instead return `level_completed: True` and leave the game advanced |

### 3.4 Build recipe — the P1 patch machinery, reused
Same vehicle and same insertion point as P1 (`p1_notes_prereg_2026-08-22.md`): the certified field floor + **ONE inserted patch cell at position 6** — after cell 5's `_run_shell_commands("setup_commands.json")`, **before cell 7's `benchmark_initial.pkl` load, which is the first import of `inference`** — asserted at build **and** at runtime (`assert "inference" not in sys.modules`). The patch is a working-copy shadow of `ARC3-Inference` applied by **exact anchors, each asserted `count == 1` at runtime; any bundle drift dies LOUDLY as an INFRA DEATH, never a silent stock run.**

Anchor sites (all in the vendored 08-15 bundle, verified unique offline before the build):
- `inference/agent/python_tool_sandbox.py` — `def action(actions):` and `runtime_globals["action"] = action` (the sibling definition and its export);
- `inference/agent/python_tool_sandbox.py` — `_refresh_state(state_payload)` (the post-episode state refresh `attempt()` reuses);
- `inference/agent/tool_agent.py` — the `python` tool description string, and the tool-result assembly where `retry_mode` is appended;
- `inference/agent/tool_agent.py` — the per-turn bookkeeping that owns `current_level` (where the consecutive-turns-on-this-level counter lives).

Every patched file must `ast`-compile at build. Cell count 11 → **12**, insertion **declared** so `local_gate` N1b/N6a and `preflight` D4 compare against the base with the declared cell removed.

---

## 4. RUNTIME CERTIFICATION (before any number; any failure ⇒ INFRA DEATH, never NULL)
1. Kernel COMPLETE, `benchmark.json`, **n_games = 25**.
2. Served **`Qwen/Qwen3.8-27B-FP8`**; model_sources EXACT at pull-back.
3. `reasoning_effort` **ABSENT** (⇒ template `xhigh`, the floor's setting).
4. `anim-20260807` bundle; solver banner echoes **`max_runtime_s_per_game=7920.0`**.
5. **`[p2] reset-retry armed H=4 K=5 cap=40`** present, and **`[p2] patch applied: N anchors`** with N equal to the sealed anchor count.
6. **`[p2] reset semantics OK`** — a boot-time self-check that `os.environ["ONLY_RESET_LEVELS"] == "true"` **before** `arcade.make`, so the level-0 footgun cannot be live.
7. **FORBIDDEN markers, all absent:** `EDGE1`/`EDGE2`, `= 3960.0`, `= 23760.0`, `[notes]` (P1), `[cadence]` / `reasoning_effort` / `LOCAL_ANALYZER_MAX_OUTPUT` (C1/C2), and the graft token set.

---

## 5. DELIVERY — read BEFORE the effect, own verdict class

**Seed 1 alone certifies DELIVERY. Seed 1 alone NEVER produces an lc verdict** (n=1 MDE 11.1).

| gate | threshold |
|---|---|
| **D1 trigger fires** | ≥ 15 of 25 games enter `retry_mode` at least once (the floor has 15/25 games ≥80% dead — if the trigger does not arm on those, H is mis-set) |
| **D2 `attempt()` call-rate** | **≥ 25% of retry-mode turns call `attempt()`** — *the program's own kill line, verbatim* |
| **D3 episodes are real** | median episodes per calling turn ≥ 2, and ≥ 90% of episodes are followed by a RESET that returns the level-start frame signature |

**Failing D1, D2 or D3 with the arm ARMED ⇒ `DELIVERY FAILURE`** — a distinct verdict class, **not** a null, **not** an infra death. It does not count against §7's kills, licenses no statement about whether extra attempts buy levels, and triggers **re-scope, never re-read**. Logged in KAOS as `DELIVERY FAILURE` with the measured D1/D2/D3.

This is the exp 9 / exp 36 / P1-mechanism-C lesson made procedural: the campaign has four times read "delivery without use" as a mechanism null. The P1-suppressor reached **96.3% delivery with no behaviour change**; that is the outcome this clause exists to name correctly.

---

## 6. SEALED READ (n = 2 seeds; comparator = field floor 28 + Arm A base 30, mean 29.0, pooled sd 2.80, n=2 MDE 7.84)

- **lc_total mean ≥ 37** ⇒ **SIGNAL**
- **22 ≤ mean ≤ 36** ⇒ **NULL**
- **mean ≤ 21** ⇒ **HARM** (flag off permanently)

Co-primary **`trim1`** reported; an lc/trim1 disagreement is reported, never silently resolved.

**Secondary, pre-registered, explicitly non-inferential** — and the most informative thing this arm can produce:
- **dead-phase turn count** (turns whose action index ≥ Σ`actions_per_level[:k]`), floor = **229 of 424**;
- **terminal-level action ratio** to `base_actions_per_level`, floor median **0.41**;
- episodes per game, and the share of cleared levels first reached inside an `attempt()`.

> **The dissociation is the point.** P2's *signature* is a fall in dead-phase turns. If dead-phase turns fall sharply and lc does not move, that is not a weak null — it is the strong result (§7).

**Break-even note carried from §1:** the +10% token tax costs ≈0.8 lc, so a true effect of +0.5 lc reads as a NULL. **n = 2 cannot separate "small win eaten by its own tax" from "no effect".** Acknowledged limitation, not resolvable at our power; it is written here so it is not discovered in the read.

---

## 7. KILLS — stated pre-data

1. **P0.1 fails** ⇒ arm never built. **DISCHARGED: it passed (§2).**
2. **`attempt()` call-rate < 25%** in retry-mode turns ⇒ **DELIVERY FAILURE**; the mechanism was never tested; re-scope, do not re-read.
3. **lc mean ≤ 21 at n = 2** ⇒ **HARM**; the token tax plus action inflation outweigh the extra episodes; flag off permanently.
4. **The quadratic eats the depth:** action ratio on cleared levels exceeds `sqrt((k+2)/k)` on ≥50% of scoring games ⇒ re-scope to k ≤ 1 games only (readable pre-hoc from `actions_per_level`).
5. **★ THE DECISIVE ONE — dissociation in the wrong direction: dead-phase turns fall sharply AND lc does not move.** That proves the attempts happen, are independent, and still fail ⇒ **the entire ATTEMPTS family dies together: search, MCTS, Go-Explore, retry, cross-clone transfer.** perturn §5.5 calls this *"the single most valuable negative result available after exp 39"*, and I agree: it would mean the failures are **world-model** failures, not **exploration** failures, and it would redirect the remaining ten weeks in one build.
6. **Two infra deaths** ⇒ arm parked.

---

## 8. COST
2 builds × ~2.3 GPU-h = **~4.6 GPU-h. 0 submission slots.** Arm 0's nightly field-floor redraw is untouched and continues to bank the ~1.6 floor for free.

---

## 9. INTERACTION WITH THE CADENCE ARMS — they do not conflict

**They are separate kernels on separate slugs, with separate bundles, separate working directories and no shared state. They can be built on the same day without any confounding, and I am stating that explicitly so the next session does not serialise them out of caution.**

| | C1 `cadence-effort` | C2 `cadence-cap` | P2 `reset-retry` |
|---|---|---|---|
| what changes | vLLM server default `reasoning_effort=medium` | `LOCAL_ANALYZER_MAX_OUTPUT=768` (+ ctx companion) | sandbox `attempt()` + harness stuck trigger |
| layer | chat template | analyzer request | tool sandbox / turn loop |
| inserted cell | position 4 | position 6 | position 6 |
| marker | `[cadence] effort pin armed` | `[cadence] max_output armed` | `[p2] reset-retry armed` |
| forbids | `[p2]`, `[notes]`, the other cadence marker | idem | `[cadence]`, `reasoning_effort`, `[notes]` |

Each arm's certifier **forbids every sibling's marker**, so a compound can never be read as a single-variable arm (exp 34's standing rule). The three arms share the identical comparator (mean 29.0, sd 2.80) and identical bands, so their reads are directly comparable.

**The only real interaction is arithmetic, and it is a scheduling fact rather than a scientific one:** 2 pushes/day against the weekly GPU pool, ~2.3 GPU-h per build, and five owed builds (P1 seed 2, C1 ×2, P2 ×2) ≈ **11.5 GPU-h** — comfortably inside a 30 GPU-h week, but three days of slots.

**A note on eventual compounding, pre-registered so it is not decided post-hoc:** P2 costs ≈+10% tokens and the cadence arms cut tokens. If both SIGNAL they are complementary and a compound is the natural week-5/6 build. **They must not be compounded before each has read at n = 2** — compounding a token-adding mechanism with a token-cutting one before either is measured would make both unreadable, which is precisely how edge-1 and edge-2 were priced too late.

---

## 10. ADVERSARIAL SELF-REVIEW (carried from perturn §6, not softened)

**A1 — "K independent attempts from the same wrong world model are K correlated failures."** The strongest objection and **not fully answered.** Against P2: the 20-clone reference shows `tr87` and `wa30` at 0/20 — twenty genuinely independent whole-game attempts, all failing. For P2: the same dataset shows 11 of 25 games with per-game score sd ≥ mean and max/mean = 2.97, so for most games the outcome *is* stochastic given the policy. **Honest scope: P2 helps the stochastic middle and does nothing for the hard walls, which are 2 of 25 games.** *(Amendment, 2026-08-22: that clone dataset is **Qwen3.6** — see `reference_config_provenance_2026-08-22.md` §5.1. Its variance structure is evidence that per-clone variance is large in this environment, not a measurement of our config's. P5a measures the real one.)*

**A3 — "`attempt()` is a prompt addition wearing a costume."** Partly fair: the helper must be named in the tool description. The claim is about the *kind* of text — an affordance, not a per-turn obligation. **If P1 and P2 both NULL with delivery proven, the correct conclusion is that the distinction is not real and the whole harness-modification program is dead.** That is itself decision-grade, reached in four builds.

**A4 — "6.96 turns/level will not generalise to the dead phase."** It will not, and it is flagged: the marginal games run 16–20 turns/level (`cn04` 16, `s5i5` 18, `m0r0` 19, `ka59` 20). Under that rate the honest ceiling for this arm drops to lc ≈ 34–36, i.e. **the SIGNAL band is at the top of the plausible range, not in its middle.** P(SIGNAL) ≈ 25–30%.
