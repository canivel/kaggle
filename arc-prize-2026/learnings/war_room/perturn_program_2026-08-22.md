# THE PER-TURN-VALUE PROGRAM — 2026-08-22
**Order (principal):** design the per-turn-value program. Round 1's first arm (budget elasticity, exp 39) returned a program-level result: more turns buy almost nothing above our operating point, so the 2.5+ tier is not getting more turns — it is getting more value per turn.
**Owner:** strategy-perturn. **Mode: READ-ONLY on campaign operations** — no pushes, no submissions, no queue edits, no writes to any other lane's staged artifacts. Lane lock registered.
**Predecessor:** `learnings/war_room/original_program_2026-08-22.md` (round 1). Its establishments are not re-derived here; two of them are **corrected** (§1.6, §1.7) on new evidence.
**Tags:** **[V]** computed/verified from artifacts in this session · **[V-doc]** verbatim in a verified artifact · **[INF]** inference · **[UNK]** unknown.
**New evidence pulled this session (read-only, zero GPU, zero slots):** the full transcript corpus of the **certified field floor** (`canivel/arc3-q38-field-eval`, 25 games) and of the **T3 3×-budget arm** (`canivel/arc3-budget-t3-eval`, 25 games). The pulled `benchmark.json` for the field floor is **byte-identical to `runs/kernel_pulls/q38_field_v1/benchmark.json`** (dict equality checked) — so every transcript number below attaches to the certified lc-28 / 6.173 artifact. **[V]**

---

## §0 — THE VERDICT IN SIX LINES

1. **We measured the wasted turn. It is real, it is large, and recovering it is not enough.** 54.0% of the field floor's 424 acting turns are spent on a level that is never cleared; 62.1% of all LLM generations take no environment action; 49.7% of all generated characters sit in those action-free generations. **[V]**
2. **But the honest arithmetic says waste-recovery caps out at lc ≈ 32 (LB ≈ 1.8).** Recovered waste behaves like extra budget, and above our operating point ε = 0.17. Even the *physically impossible* upper bound — zero inspection, zero tracebacks, zero request timeouts — is **+4 lc**. **Say it loudly: the target requires better decisions, not merely non-wasted ones.** **[V arithmetic]**
3. **The target, in the only currency that matters:** we clear a level every **6.96 productive turns**; 229 of 424 turns produce nothing. LB 2.50 (lc 45) requires **52% of the dead phase to become productive**, i.e. **×1.60 value per turn at a constant 17 turns/game**. LB 3.5 requires ×2.19. **[V]**
4. **Actions are in SURPLUS; independent attempts are scarce.** T3 proved we can spend 2184 extra actions for the same score. Therefore the strictly favoured mechanism class is **anything that converts surplus actions into extra independent attempts at no token cost** — and `ONLY_RESET_LEVELS=true` makes every level a restartable episode that the agent uses **9 times in 1639 actions (0.55%)**. **[V]**
5. **Two instruments are broken and one round-1 premise is wrong.** The `hard_noop_guard` is armed and has **never fired** in 1639 + 3616 actions — it keys on a blake2b of the *full 64×64 grid* while the games carry ticking HUD strips, so its key can essentially never recur (`noop_guard.py:16-21`). And the agent **does** query `transitions` — in 16.3% of field generations, not "never". **[V]**
6. **Honest ceiling: lc 33-38, LB 1.9-2.1 by 2026-11-02** [INF]. P(LB ≥ 2.50) ≈ 10-15%. P(top-10) ≈ 2-4%. The board's gold line moves +0.14/day; a top-10 finish on 11-02 plausibly needs LB ≈ 4.0 ⇒ lc ≈ 72. **Nothing on this list reaches that, and I will not pretend otherwise.**

---

## §1 — THE WASTED-TURN TAXONOMY, MEASURED

### 1.0 The unit of accounting (get this right first)

Three different "turns" exist and the campaign has conflated them. On the certified field floor **[V]**:

| unit | count | definition |
|---|---|---|
| **environment actions** | **1639** | entries in `game_run.history` |
| **acting turns** | **424** | history entries with `generated_tokens > 0` (a batch's tokens land on its last action) |
| **analyzer invocations** | **449** | `--- analysis_step=N ---` blocks with a fresh system prompt (the extra ≈25 are the per-game give-up turns, `final_generated_tokens`) |
| **LLM generations** | **1181** | `[MODEL RESPONSE META]` records = 2.63 per acting turn |

The **decision** is the acting turn: 424 of them, **17.0 per game**. Everything below is per-turn or per-generation.

### 1.1 The taxonomy, with percentages

**A. TURN-LEVEL (n = 424 acting turns, field floor)** **[V]**

| # | category | share | measure |
|---|---|---|---|
| A1 | **turns on a level that is never cleared** ("the dead phase") | **54.0%** (229) | turns whose action index ≥ `sum(actions_per_level[:k])` |
| A2 | turns on levels that *were* cleared ("the productive phase") | 46.0% (195) | 28 levels ⇒ **6.96 turns/level** |
| A3 | turns that fire exactly **one** action | **53.5%** | batch-size distribution |
| A4 | turns that fire ≥ 8 actions ("blind bursts") | 12.2% (54 turns) | but they carry **46.9% of all 1639 actions** |
| A5 | turns lost to a client-side `Read timed out` request error | **7.3%** (33 of 449 blocks) | tokens generated then discarded |

**B. GENERATION-LEVEL (n = 1181 LLM generations)** **[V]**

| # | category | share of generations | share of generated chars |
|---|---|---|---|
| B1 | **generations that execute no environment action** (inspection-only) | **62.1%** (733) | **49.7%** (3.18M of 6.40M) |
| B2 | generations that execute actions | 37.4% (442) | 49.5% (3.17M) |
| B3 | generations with **no tool call at all** | 0.5% (6) | 0.8% (53k) |
| B4 | tool calls that raised a **Python traceback** | **3.8%** (45) | 2.8% (176k) |
| B5 | tool-call **format** failures / markup recovery | **0.0%** (0) | — |
| B6 | responses truncated by `finish_reason: length` | **0.0%** (0) | — |

**B5 and B6 are dead hypotheses. Formatting failure and truncation are not costing us anything.** **[V]**

**C. ACTION-LEVEL (n = 1639 actions)** **[V]**

| # | category | share |
|---|---|---|
| C1 | actions on a never-cleared level | **56.7%** (930) |
| C2 | actions that immediately repeat the identical previous action (same id + same `data`) | **45.2%** |
| C3 | `ACTION6` clicks re-using a coordinate already clicked in that game | **44.8%** of 524 clicks (clicks are 32.0% of all actions) |
| C4 | A-B-A-B thrash pairs | 2.1% |
| C5 | `RESET` | **0.55%** (9 in 25 games) |
| C6 | actions blocked by the armed `hard_noop_guard` | **0.0%** (0) |

**D. CODE-BEHAVIOUR (what the agent does with the interpreter, n = 1181 generations)** **[V]**

| idiom | field floor | T3 | older Q3.6 configs (n=3139 cells) |
|---|---|---|---|
| `.ascii` scan (the prompt says never do this for the whole board) | **50.3%** | 35.3% | 30.3% |
| `segmentation` | 40.7% | 48.7% | 54.7% |
| `transitions` / `last_transition` | **16.3%** | 20.1% | 9.9% |
| `animation()` (free, costs no action) | 13.5% | 10.9% | 1.0% |
| **search idioms** (`deque`/`heapq`/`bfs`/`frontier`/`visited`/`shortest`/`itertools.product`/`beam`) | **5.8%** | 3.7% | **0.8%** |
| explicit candidate scoring (`candidat`/`score_`/`simulate`) | — | — | **0.2%** |

> **The system prompt tells the agent, verbatim, "it is usually safer to write an explicit search algorithm such as BFS … pathfinding, flood fill, BFS, DFS, beam search, limited action-sequence search … are all valid." The agent writes search code in 5.8% of generations and scores candidate sequences in 0.2%. It uses the Python interpreter as a printf debugger.** **[V]**

### 1.2 The dead phase, located in time

For each game: how much of its wallclock elapses **after its last level clear**? **[V]**

| run | median dead-time share | games ≥80% dead | games ≤30% dead |
|---|---|---|---|
| **field floor (T1, 7920 s)** | **88%** | **15/25** | 7/25 |
| T3 (23,760 s) | 78% | 12/25 | 2/25 |
| T0.5 (3960 s) | 100% | 17/25 | 4/25 |

Field-floor per game (`game:levels/dead-share`): `sp80:0L/100% bp35:0L/100% cd82:0L/100% dc22:0L/100% g50t:0L/100% r11l:0L/100% sk48:0L/100% tr87:0L/100% ar25:1L/96% vc33:1L/91% tu93:1L/91% su15:1L/89% ft09:2L/88% tn36:1L/83% ls20:1L/81% lf52:1L/64% s5i5:1L/62% wa30:1L/55% re86:2L/25% sc25:2L/19% m0r0:1L/17% sb26:7L/15% cn04:1L/13% ka59:1L/7% lp85:3L/1%` **[V]**

Read: **eight games never score at all and spend 100% of a 2h13m clock producing nothing; a further seven score once and then idle for ≥80% of the remaining clock.** The two ends of the distribution are `sb26` (7 levels, 15% dead, 3.0 turns/level) and `dc22` (26 turns, 0 levels, 100% dead).

### 1.3 The grind, priced against the designers' pace

Terminal-level actions divided by that level's `base_actions_per_level` **[V]**:

| run | median ratio | games spending ≥1× baseline on the level they fail | ≥2× |
|---|---|---|---|
| field floor | **0.41** | 10/25 | 2/25 |
| T3 | 0.83 | 10/25 | **7/25** |
| T0.5 | 0.48 | 3/25 | 1/25 |

**Two distinct failure populations, and they need opposite treatments** **[V]**:

- **TRUNCATED (≈8 games).** Cut off within a handful of actions of a fresh level: `lp85` 0, `re86` 0, `ka59` 1, `cn04` 2, `sb26` 4, `tn36` 6, `r11l` 7, `sc25` 8 terminal actions. These were still converting when the clock fired. **Budget helps them** — this is the ε ≈ 1.05 segment below our operating point.
- **GRINDERS (≈10 games).** `ls20` 192 actions (1.56× baseline), `dc22` 92, `cd82` 71, `ar25` 69, `sp80` 62, `tr87` 60, `su15` 59, `wa30` 58, `bp35` 55. **Budget is pure waste on them** — and T3 proves it: `wa30` burned **447 actions = 6.30× baseline for 0 levels**, `ls20` 407 = 3.31× for 1 level, `r11l` 5.03×, `ka59` 4.54×. **[V]**

### 1.4 Long-run degeneration: the mechanism behind the saturation

T3's transcripts show the agent **getting worse as the run lengthens**, not merely running out of ideas **[V]**:

| statistic | field floor (T1) | T3 (3× clock) |
|---|---|---|
| LLM generations per acting turn | 2.63 | **3.50** |
| generations with **no tool call** | 6 (0.5%) | **197 (6.0%)** |
| generated chars in no-tool-call responses | 53k (**0.8%**) | 3.20M (**16.5%**) |
| **exact-duplicate** code cells re-run within a game | 25 (2.1%) | **354 (11.5%)** |
| near-duplicate (>0.85 similarity) code cells | 13.0% | 12.4% |
| tracebacks | 3.8% | 3.7% |
| request-error turns | 7.3% | 8.2% |

**93% of T3's no-tool-call responses occur in the last third of the run** (median relative position 1.00). **[V]**

> This is the missing mechanism for exp 39's ε = 0.17. The marginal turn is not merely low-value: **at 3× budget the agent begins emitting responses with no tool call at all and re-running code it already ran.** 16.5% of T3's entire generated-character budget went into responses that did nothing. Combined with `context_budget_tokens: 31744`, `history_messages: 18` (an 18-message conversation window) and the level-transition **wipe** of the carried world model (exp 17/18), the picture is a policy whose working memory does not survive the length of the run it is given. **[V for the counts; INF for the causal chain]**

### 1.5 Clone variance: the same policy is a lottery

`runs/tufa_example_run/benchmark.json` — a **20-pass × 25-game** reference run (2026-06-02, label `0-history-turns`, 500 game-runs) that nobody in this campaign has read. It is the campaign's **only true clone-replicate dataset**, and it is free. **[V]**

| statistic | value |
|---|---|
| mean-over-clones, summed to a benchmark score | **1.600** |
| max-over-clones, summed | **4.750** |
| **max / mean** | **2.97** |
| games with per-game score sd ≥ mean | 11 of 25 |
| hard walls (0 levels in 20/20 clones) | `tr87`, `wa30` |
| games this **weaker** config clears ≥65% of the time that our field floor scored **0** on | `bp35` (0.90), `r11l` (0.90), `sp80` (0.65) |

Two consequences.
- **[V]** Our field floor lost three near-certain level-1s (`bp35`, `r11l`, `sp80`) to draw noise. That is ~+3 lc sitting in *consistency*, not capability.
- **[INF, strong]** `make_benchmark_kaggle_official_110` builds 110 **independent** `GameAPI` entries at `n_passes=1` and the benchmark aggregates them with equal weights. So the scored rail reports the **mean over clones**. Variance is therefore **not** an asset to be harvested by max-taking; it is a pure loss, and "make every clone behave like our best clone" is worth up to ~3×. That restates the per-turn-value thesis exactly: **consistency IS decision quality.**

### 1.6 CORRECTION to round 1: the agent does query `transitions`

Round 1 recorded "the harness exposes `transitions`, the agent never queries [it]". **Measured: 16.3% of field-floor generations reference `transitions`/`last_transition`; 20.1% at T3; 9.9% across the older Q3.6 corpus.** The finding is not absence of access to transitions — it is **absence of search over them** (5.8%/0.2%, §1.1D). Any arm premised on "expose transitions" is aimed at the wrong gap. **[V]**

### 1.7 INSTRUMENT DEFECT: a guard that cannot fire

`hard_noop_guard: True` in every analyzer status line. `NoopGuard` blocks re-executing an action already proven to have no effect **in the exact same board state**, keyed by
`board_signature(grid) = blake2b(repr(full 64×64 grid))` (`inference/agent/noop_guard.py:16-21`). **[V, source]**

It has **never fired**: zero occurrences of `"action(s) in this batch were blocked before execution"` in 1639 field-floor actions or 3616 T3 actions. **[V]**

Root cause **[INF, strong]**: the games render **HUD/timer strips that change every step** — the system prompt devotes a paragraph to warning the model about exactly these ("a long horizontal or vertical line near an edge is a timer or remaining-steps bar… it often shrinks or changes each step"). A full-grid hash therefore takes a **new value on essentially every step**, so the `(level, board_before_sig, action_sig)` key can never recur and the memo can never match. This is `feedback_guard_never_fired` instance N: *a guard that never fired may be one that cannot.* The module docstring says its predecessor (K1 auto-memory) left "~12% no-op repeats"; **the replacement blocks 0.0%.** **[V-doc for the 12%; V for the 0.0%]**

### 1.8 What is NOT wasted (negative results worth recording)

- **Tool-call formatting: 0 failures in 1181 generations** (1 in 1223 across edge-2). **[V]**
- **Context truncation: 0 `finish_reason: length`.** **[V]**
- **Actions blocked by the harness: 0.** **[V]**
- **Engine-refused actions after GAME_OVER: 0** in the field log. **[V]**
- **Action efficiency on cleared levels: 709 actions vs 790 baseline = 0.90.** We already beat the designers' pace where we succeed. **[V, round 1, re-confirmed]**

---

## §2 — THE PER-TURN ARITHMETIC (the honest verdict)

### 2.1 The base rates

```
424 acting turns      ->  28 levels
195 productive turns  ->  28 levels   =   6.96 turns per level cleared
229 dead turns        ->   0 levels
average turn value = 28/424 = 0.0660 levels/turn
```
**[V]**

Turns per level, per game, where a level was cleared: `sb26 3.0 · ft09 2.5 · tu93 3.0 · ar25 3.0 · lp85 4.0 · ls20 4.0 · vc33 5.0 · tn36 6.0 · su15 7.0 · re86 8.0 · lf52 9.0 · sc25 9.5 · wa30 12.0 · cn04 16.0 · s5i5 18.0 · m0r0 19.0 · ka59 20.0` **[V]**.
**When the agent has the game's mechanic, it clears a level every 3 turns. When it doesn't, it burns 20 and gets nothing. There is no middle.**

### 2.2 The marginal turn, measured (not assumed)

T1 → T3: turns 424 → 921 (+497), levels 28 → 35 (+7).

```
marginal turn value = 7 / 497   = 0.0141 levels/turn
average  turn value = 28 / 424  = 0.0660 levels/turn
marginal / average  = 0.21
```
**The marginal turn is worth 21% of the average turn.** **[V]**

### 2.3 What recovering the waste actually buys

Recovered waste is, by construction, **extra budget** — so it must be priced on exp 39's curve, not at the average rate. Above our operating point **ε = 0.17**, so `lc ∝ B^0.17`.

| recovery (each an optimistic upper bound) | budget multiplier | Δlc | lc | LB (lc-linear 0.062+0.0538·lc) |
|---|---|---|---|---|
| eliminate **all** request-error turns (7.3%) | ×1.079 | +0.35 | 28.4 | 1.59 |
| eliminate **all** tracebacks (2.8% of chars) | ×1.029 | +0.14 | 28.1 | 1.57 |
| eliminate **all** inspection-only generation (49.7% of chars) | ×1.99 | **+3.5** | 31.5 | 1.76 |
| **all three, simultaneously and perfectly** | ×2.16 | **+4.0** | **32.0** | **1.78** |

> **VERDICT — stated as loudly as the order requires: recovering wasted turns is worth at most +4 lc and lands at LB ≈ 1.8. That is inside the redraw ceiling we already own. The 2.5 tier is NOT reachable by making our turns non-wasted. It is reachable only by making our decisions better.** **[V arithmetic on a measured elasticity]**

Sanity cross-check against exp 39 from the other direction: at ε = 0.17 the budget multiplier needed for lc 28 → 45 is `(45/28)^(1/0.17) = 17.8×`. Round 1 said 1.9× at baseline pace; **the measured curve says 17.8×.** Both are correct — the difference is exactly the per-turn-value gap. **[V]**

### 2.4 The target, restated in the only currency that matters

Hold turns fixed at 424 (17/game — competition-legal, no extra wallclock).

| target | LB | required lc | required lc/turn | ×current | required conversion of the 229 dead turns |
|---|---|---|---|---|---|
| our floor | 1.59 | 28 | 0.0660 | 1.00× | — |
| redraw ceiling | ~1.90 | 34 | 0.0802 | 1.22× | 42 turns (18%) |
| public-kernel ceiling | 2.23 | 40 | 0.0943 | 1.43× | 84 turns (37%) |
| **top-13 line today** | **2.50** | **45** | **0.1061** | **1.60×** | **118 turns (52%)** |
| Tufa | 3.04 | 55 | 0.1297 | 1.96× | 188 turns (82%) |
| cstl | 3.57 | 65 | 0.1533 | 2.32× | **impossible** (needs 258 of 229) |

*(conversion computed at the measured 6.96 turns/level)*

**LB 2.50 = converting 52% of the dead phase into cleared levels. LB 3.57 is unreachable at 17 turns/game even if every single dead turn became productive** — that scenario tops out at `424/6.96 = 60.9 levels`, LB ≈ 3.34. **[V]**

### 2.5 The resource inversion — the single most actionable consequence

Because ε = 0.17 above our point, **actions are nearly worthless at the margin — which means they are also nearly FREE at the margin.** T3 demonstrated 2184 extra actions bought +0.04 mean_score. **[V]**

And the score arithmetic bounds how freely we may spend them. Going from `k` to `k+1` cleared levels at a uniform action ratio `r` relative to baseline beats staying at `k` iff

```
W_{k+1} / r²  >  W_k        with  W_k = k(k+1)/2
  =>  r  <  sqrt( (k+2) / k )
```

| k (levels already cleared) | max affordable action ratio r |
|---|---|
| 0 → 1 | **unbounded** (any first level is pure gain) |
| 1 → 2 | 1.73 |
| 2 → 3 | 1.41 |
| 3 → 4 | 1.29 |
| 7 → 8 | 1.13 |

**20 of our 25 games sit at k ≤ 1** (8 at k=0, 12 at k=1) **[V]**, and our realised ratio on cleared levels is already **0.90** — so relative to what we do now the retry allowance at k ≤ 1 is **≈1.6-1.9×** and at k = 0 it is unlimited. **The action budget for an aggressive retry policy exists precisely where the levels are.** **[V arithmetic]**

> **Therefore the strictly favoured mechanism class is: convert surplus ACTIONS into extra independent ATTEMPTS, at zero token cost and zero turn cost.**

---

## §3 — PER-TURN-VALUE MECHANISMS (none is a prompt/context addition)

The dead class, restated so no one re-proposes it: **exp 3, 4, 9, 33, 36, 37 — six additions, six failures.** Every single one added an *obligation to each turn* (restate ground truth; note your action count and the scoring rule; emit your world model visibly) or widened a window. The two that were *proven to deliver* (P1-suppressor 96.3%, capture-contract H3) still nulled or harmed. **An addition that taxes the turn cannot win here.**

### P1 — PERSISTENT TOOL NAMESPACE (carry-over scratch)

**Mechanism.** `inference/agent/python_tool_sandbox.py` builds `runtime_globals` fresh in a subprocess on every call; the system prompt states "Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need." Add a single JSON-serialisable dict, `notes`, that is injected into the sandbox globals, returned in the sandbox result, and re-injected next call **within the same game** (bounded, e.g. 8 KB, dropped on `run_complete`). ~30 lines across `python_tool_sandbox.py` and `tool_agent.py`, plus **~12 words of prompt change** — the deletion of "The snippet is ephemeral and is not saved across tool calls" and one clause naming `notes`.
**Why it can beat the floor when six additions didn't.** It is the only candidate that **removes** work rather than adding an obligation. It is also the only memory channel that avoids both proven-harmful routes: it does not go through the **visible message** (edge-2: HARM at −10 lc with the plumbing proven) and it does not widen the **context window** (edge-1: HARM at −12 lc). State lives in the interpreter, not in tokens.
**Evidence it addresses something real.** 62.1% of generations take no action; 13.0% of code cells are near-duplicates of an earlier cell in the same game (11.5% *exact* duplicates at T3); the carried world model is wiped on every level transition (exp 17/18) and 97.6% of the model's content is hidden from the only capture path. **[V]**
**Budget price.** Turns: **0**. Tokens/turn: **negative** (a re-derivation avoided is a tool call avoided). Actions: **0**. This is the only candidate that is budget-**positive**.
**Expected.** +2 to +6 lc [INF, wide]. It is not a decision-quality mechanism in itself; it is the enabler that makes one affordable.
**How it fails.** (i) *Delivery-without-use* (exp 9's species): the model never writes to `notes`. Detectable directly — instrument the write rate; a <30% write rate is the kill. (ii) The saved tokens are re-spent on longer reasoning and turns don't rise. (iii) Serialisation cost on large blobs.

### P2 — RESET-ANCHORED EPISODIC RETRY (the search that is actually affordable)

**Mechanism.** `ONLY_RESET_LEVELS=true` is set in our deployed notebook (`taaf_kaggle_run.ipynb:339`) and `game_api.py:215-222` documents that with it set, a mid-play `RESET` returns to the **start of the current level** rather than to level 0. **Every level is therefore a restartable episode.** Add one harness helper available inside the `python` tool:

```
attempt(seq)  ->  execute the action list, capture {level_completed, board_delta_summary,
                  reward, terminal_reason}, then RESET back to this level's start,
                  and return the compact summary.  Costs actions; costs no turn.
```
plus a **stuck trigger** (harness-side, no prompt text): after H consecutive acting turns on the same level with no `level_completed`, the tool result carries a one-line structured field `retry_mode: on, episodes_available: K`.
One LLM turn then evaluates **K independent candidate plans** instead of executing one. At 17 turns/game and K = 5 that is 85 episodes/game against 17 today.
**Why it can beat the floor when six additions didn't.** It is not an addition *to a turn's obligations* — it changes what a turn can *buy*. It spends the resource we have in surplus (actions, ε = 0.17) to buy the resource we lack (independent attempts). And it attacks precisely the measured pathology: 229 dead turns spent **mutating a progressively more broken state** instead of sampling from a known-good one, with RESET used 0.55% of the time.
**Budget price.** Turns: **0** (episodes run inside one turn). Tokens/turn: **+small** — K compact summaries at ~200 chars each ≈ +1000 chars on a 5400-char average generation ⇒ ~+18% tokens on retry turns only, which are ≤54% of turns ⇒ ≈ **+10% tokens overall** ⇒ ×0.91 budget ⇒ **−0.8 lc** as the standing tax. Actions: **+++**, and §2.5 shows the score allowance covers it at k ≤ 1 where 20/25 games sit.
**Expected.** The dead phase is 229 turns at 6.96 turns/level. Converting 20% of it is **+6.6 lc** (lc 35, LB ~1.95); 52% is **+17 lc** (lc 45, LB 2.50). Somewhere between "nothing" and "the target" — which is exactly why it is worth a slot: it is the only candidate on the board whose *upper* branch reaches the goal.
**How it fails.** (i) **RESET does not restore cleanly** — must be verified before any build (see P0). (ii) The model ignores the helper (0-for-6 species) — instrument the `attempt()` call rate; <25% of retry-mode turns is the kill. (iii) Retry actions accumulate against the level's score and the depth gained is eaten by the quadratic — bounded by §2.5 and readable pre-hoc from `actions_per_level`. (iv) The candidate plans are all drawn from the same wrong world model, in which case K independent attempts are K correlated failures — **this is the real risk and P2 does not address it**; see §6/A1.

### P3 — REPAIR THE NO-OP GUARD (an armed instrument that cannot fire)

**Mechanism.** Re-key `board_signature` on the **gameplay interior** — drop any row/column strip flush to a border whose contents change on ≥50% of observed steps (computable online from the first ~10 frames), or hash the multiset of `segmentation` node `(color, pixels, hash)` tuples instead of the raw grid. Keep the existing animation exemption verbatim (the docstring records that recording animated actions as no-ops "hard-blocked actions that had clearly worked, on exactly the games with the most animations" — do not regress that). Then **surface a `known_noops_here: N` integer in the tool result** rather than only hard-blocking.
**Why it can beat the floor.** It is a **repair of an existing armed mechanism**, the cheapest class of change in this campaign's history (exp 6, 22, 34, 35 are all instrument corrections that outranked arms). Zero prompt text.
**Budget price.** Turns 0 · tokens ~0 (one integer) · actions **negative**.
**Expected.** Small: **+0.5 to +1.5 lc** [INF]. Actions are cheap, so blocking them is worth little; the value is in stopping the 53.5% single-action probe turns that re-fire a known-dead action.
**How it fails.** The interior no-op rate is genuinely low (currently **[UNK]** — `intermediate_states.pkl` holds the frames and the grids are reachable with a small unpickling shim; measuring this costs one CPU hour and should be done before the arm, not after). **Rider, not a lead arm.**

### P4 — A LEARNED COMPONENT ON OUR OWN CORPUS

**The asset, inventoried this session [V]:** 606 transcripts on disk across 15 pulls + 50 pulled fresh today = **656 fully-instrumented game transcripts**; `benchmark.json` action histories for **725 game-runs** across 60 artifacts; ~7,000 harness turns; ~19,000 LLM generations carrying code, hidden reasoning, tool output and outcome; plus `intermediate_states.pkl` frame sequences for at least 3 pulls.
**Product (a) — the stuck classifier.** Label is free and now well-defined: *did this game ever clear another level after turn t?* Features: turns since last clear, tokens/turn trend, duplicate-code rate, board-delta entropy, terminal-action ratio to baseline. This is what triggers P2 causally rather than with a hand-set H. Trainable on the 3080 in an afternoon. Target AUC ≥ 0.70; below that, P2 keeps a fixed H.
**Product (b) — an action prior / LoRA policy.** **I rank this LOW and recommend against it this cycle.** Positive events campaign-wide are ~700 level clears; a LoRA on a 27B trained on 700 positives is thin; serving a LoRA on our rail already died once (exp 8, authoring defect, never re-attempted); no cloud spend is authorised; and any learned policy change is a policy change of exactly the class that has gone 0-for-6. The corpus's honest highest use is **diagnosis and triggering**, not policy.
**Budget price.** Turns 0 · tokens 0 (runs harness-side) · actions 0.

### P5 — THE COMPETITION-SHAPE RAIL + CONSISTENCY

Round 1's Arm 3, now with hard data behind it (§1.5): the scored rail is 25 games × ~4.4 clones aggregated as a **mean**, and our only clone-replicate artifact shows **max/mean = 2.97**. Two deliverables: (a) move screening to `--simulate-competition-arcade --competition-clone-runs`, which **triples statistical power for the same GPU-hours** (§5.2); (b) treat *consistency* as the objective — a mechanism that lifts the 10th-percentile clone is worth more on the LB than one that lifts the best clone. Instrument-first; unconditional.

### P6 — DEAD-GAME TRIAGE (round 1's C3), now priced

8 zero-games consume 33.7% of tokens. Reallocating to 17 survivors is +47% throughput share ⇒ `×1.47^0.17 = +6.7% lc = +1.9 levels`. **Priced and small. Demote to a week-7 rider.** [V arithmetic]

### P7 — FRAME REPRESENTATION / ACTION-SPACE CHANGES

Auto-supplying a better default view is a **context addition**. Dead class. The one exception worth keeping is P3's structured integer, which adds ~10 characters.

---

## §4 — THE BUDGET INTERACTION, PRICED FOR EVERY CANDIDATE

Every candidate must pay in decision budget. Additions that spend it have harmed 2-for-2 (edge-1 −12 lc, edge-2 −10 lc).

| candidate | Δ turns | Δ tokens/turn | Δ actions | net budget effect | standing tax in lc | verdict |
|---|---|---|---|---|---|---|
| **P1 persistent namespace** | 0 | **−** (re-derivation avoided) | 0 | **positive** | **+0.2 to +0.6** | budget-safe |
| **P2 reset-retry** | 0 | +10% overall (+18% on retry turns) | **+++** (surplus) | −9% budget | **−0.8** | affordable; must clear +0.8 lc to break even |
| **P3 noop-guard repair** | 0 | ~0 | **−** | positive | ~+0.1 | free |
| **P4a stuck classifier** | 0 | 0 | 0 | neutral | 0 | free |
| **P4b LoRA policy** | 0 | 0 | 0 | neutral (serving risk) | 0 | not this cycle |
| **P5 competition rail** | n/a | n/a | n/a | instrument | 0 | free power |
| **P6 dead-game triage** | 0 | 0 | reallocated | +47% to survivors | **+1.9** | small, real |
| edge-1 class (widen context) | — | **+100%** | — | **−48% throughput** | **−12 (measured)** | **dead** |
| edge-2 class (visible contract) | — | **+** | +6% | direct token tax | **−10 (measured)** | **dead** |

**Worked example of the rule, for P2:** a mechanism that improves per-turn value but costs 30% of the budget must clear `1 − 0.70^0.17 = 5.9%` of lc, i.e. **+1.65 levels**, before it breaks even. At P2's estimated 10% token cost the bar is **+0.8 levels**. A 30%-cost mechanism is *not* automatically net-negative at ε = 0.17 — the low elasticity that kills the budget program also **cheapens budget-spending mechanisms**. This is the second half of the resource inversion, and it is why P2 is affordable while edge-1 (which cost 48% of *throughput*, not 10% of tokens) was not.

---

## §5 — THE TEN-WEEK PROGRAM

Constraints honoured: 2 kernel pushes/day · 30 GPU-h/week (Monday reset) · 1 submission/UTC-day (**Arm 0 nightly field-floor redraw continues untouched and banks the ~1.6 floor for free**) · $50 RunPod for matched-GPU screening · RTX 5090 32 GB from ~08-28 · `scripts/local_gate.py` for the 40-second CPU pre-push gate including the real-agent-loop smoke · no cloud spend on GPU evals.

### 5.1 Methodology carried forward and added

- Primary screening statistic **`trim1`**, co-primary **`lc_total`**; raw `mean_score` retired (50.4% one game).
- **Two seeds per arm by default.**
- **New: every arm ships a DELIVERY INSTRUMENT measured before the effect is read** — write-rate for P1, `attempt()` call-rate for P2, block-rate for P3. Exp 9 and exp 36 both turned on *delivery vs use*; the campaign should never again read an effect without knowing whether the mechanism ran.
- **New:** every arm's certifier gains the forbidden-marker pair for every richer sibling (exp 34 standing rule).

### 5.2 Power analysis — how many seeds each arm needs

Pooled within-config seed sd on `lc_total` = **2.80** (exp 35). MDE at 80% power, α = 0.05 two-sided, two independent arms:

```
MDE = 2.802 · 2.80 · sqrt(2/n)
n=1 -> 11.09   n=2 -> 7.84   n=3 -> 6.40   n=4 -> 5.55   n=5 -> 4.96
```

| arm | target effect | seeds required | GPU-h |
|---|---|---|---|
| P1 persistent namespace | ≥ +8 lc | **n = 2** | 4.4 |
| P2 reset-retry | ≥ +8 lc | **n = 2** | 4.4 |
| P3 noop repair | +1 lc | **unreadable at any affordable n** (n ≈ 123) — ship as a rider inside another arm, never as a standalone read | 0 |
| P6 triage | +1.9 lc | n ≈ 34 — same verdict: rider only | 0 |

**This is the number that has bitten us and it must govern slot allocation: our screen cannot see anything smaller than ~8 lc at n=2. Any arm whose honest expectation is under +8 lc must ride inside a larger one, not consume a read.**

**The cheap power upgrade (P5a), stated with its risk.** 5 clones × 25 games at `max_runtime_s_per_game = 3960` = 125 runs / concurrency 28 = 4.5 waves × 1.1 h ≈ **5.0 h, inside a 9 h kernel**, giving per-game n = 5 ⇒ per-game sd/√5 and roughly a **3× power gain for the same GPU cost**. **Risk, stated pre-data:** 3960 s is *below* our operating point, where exp 39 measured segment elasticity ≈ 1.05. A mechanism that mainly buys budget will read **inflated** there; a mechanism that buys decisions should read approximately true. Use it as a **discriminator, never as the sealing read.**

### 5.3 ARM P0 — INSTRUMENT WEEK (Week 1, zero GPU, prereq for everything)

Four CPU-only checks, all on `scripts/local_gate.py`'s existing stub-LLM + real-competition-simulator harness:

1. **RESET semantics, verified not assumed.** On the local simulator, drive a game to level ≥1, issue `RESET`, and assert the returned frame equals the recorded level-start frame and `current_frame.level` is unchanged. **P2 does not get built until this passes.** (`game_api.py:215-222` documents the snap-back-to-level-0 footgun that `ONLY_RESET_LEVELS` prevents; the flag is set at `taaf_kaggle_run.ipynb:339`, but "the flag is set" is not "the behaviour is right", and this campaign has been burned four times by exactly that gap.)
2. **Interior no-op rate.** Unpickle `intermediate_states.pkl` for the field floor with a ghost-class shim and compute, per action, whether the **interior** board changed. Produces the number P3 needs and the campaign has never had.
3. **Guard non-firing, reproduced.** Show `NoopGuard` returns `is_known_noop == False` on a replayed field-floor action stream, and returns `True` once the signature drops a ticking border strip. *Prove the instrument can refuse before repairing it.*
4. **Delivery instruments compile** into `local_gate` as gate-group checks.

**Cost: 0 GPU-h, 0 slots, ~1 day. Teaches on failure:** if RESET does not restore cleanly, P2 dies in week 1 for free and P1 becomes the sole lead arm.

### 5.4 ARM P1 — PERSISTENT TOOL NAMESPACE (Week 1-2, first push-ready arm)

**Question.** Does giving the policy a zero-token memory across tool calls raise levels?
**Vehicle.** The certified field floor, byte-identical, at `max_runtime_s_per_game = 7920`. One variable.
**Change.** `notes: dict` injected into sandbox globals, round-tripped through the sandbox result, bounded at 8 KB, cleared on `run_complete`; the prompt line "The snippet is ephemeral and is not saved across tool calls" replaced by "`notes` is a dict that persists across tool calls within this game; everything else is ephemeral." **Net prompt delta ≈ +12 words, and it removes an obligation rather than adding one.**
**Certification, before any number is read.** Served `Qwen/Qwen3.8-27B-FP8`; `reasoning_effort` ABSENT; 08-07 anim bundle; n = 25; `max_runtime_s_per_game` echoed at 7920; zero graft markers; edge-1 / edge-2 / T0.5 / T3 forbidden markers absent; **`notes` marker present in the analyzer status line**.
**Delivery instrument, read BEFORE the effect (mandatory).** Fraction of acting turns whose code writes to `notes`. **If write-rate < 30%, the arm is a DELIVERY FAILURE, not a mechanism null** — and it is logged as such.
**Sealed read (n = 2 seeds; comparator = field floor 28 + ArmA base 30, mean 29.0, pooled sd 2.80).**
- `lc_total` mean ≥ **37** ⇒ **SIGNAL** (+8, the n=2 MDE).
- 22 ≤ mean < 37 ⇒ **NULL** — the memory channel is not worth ≥8 lc.
- mean ≤ 21 ⇒ **HARM**, flag off permanently.
- Co-primary `trim1` reported; disagreement between lc and trim1 is itself reported, never silently resolved.
**Kill criteria.** Two infra deaths ⇒ parked. Write-rate < 30% ⇒ delivery failure, re-scope not re-read.
**What it teaches on failure.** A NULL with delivery *proven* would be the campaign's fourth "delivery-without-use" and would establish that **the policy's limitation is not memory** — which retires the entire memory family (exp 17/18's mechanism, goalkeep, the capture contract, cross-turn carry) in one build. That is worth the slot on its own.
**Cost.** 2 builds ≈ 4.4 GPU-h. **0 submission slots.**

### 5.5 ARM P2 — RESET-ANCHORED EPISODIC RETRY (Week 2-3, gated on P0.1)

**Question.** Does converting surplus actions into independent in-turn episodes convert dead turns into levels?
**Vehicle.** Field floor + (P1 if P1 SIGNALs, else base). One variable: the `attempt()` helper + the stuck trigger.
**Parameters, fixed pre-data:** H = 4 consecutive acting turns on the same level without a clear; K = 5 episodes per retry turn; episode length cap 40 actions; retry disabled once `k ≥ 4` on that game (the §2.5 allowance tightens to r < 1.29).
**Delivery instrument, read first.** `attempt()` call-rate among retry-mode turns. **< 25% ⇒ delivery failure.**
**Sealed read (n = 2).** lc mean ≥ 37 ⇒ SIGNAL · 22-36 ⇒ NULL · ≤ 21 ⇒ HARM.
**Secondary, non-inferential but pre-registered:** dead-phase turn count and terminal-level action ratio. The mechanism's *signature* is a fall in dead-phase turns even if lc does not move; that dissociation is the informative outcome.
**Kill.** P0.1 fails ⇒ arm never built. Action ratio on cleared levels exceeds `sqrt((k+2)/k)` on ≥50% of scoring games ⇒ the quadratic is eating the depth; re-scope to k ≤ 1 games only.
**What it teaches on failure.** If K independent attempts from a clean state do not raise levels, then the failures are **not** exploration failures — they are world-model failures, and the entire "give it more attempts" family (search, MCTS, Go-Explore, retry, cross-clone transfer) dies together. That is the single most valuable negative result available after exp 39.
**Cost.** 2 builds ≈ 4.4 GPU-h. 0 slots.

### 5.6 Weeks 3-10

| week | plan | contingency |
|---|---|---|
| 3 | **P5a competition-shape rail** at 5 clones × 25 games (~5.0 h) — power upgrade + the clone-variance read on the *modern* config | simulator will not start inline in 2 attempts ⇒ park; local_gate stub path still validates plumbing at 0 GPU |
| 4 | **P4a stuck classifier** offline on the 656-transcript corpus (3080/5090, 0 Kaggle GPU-h); **P3 noop repair** built as a rider | AUC < 0.70 ⇒ P2 keeps fixed H |
| 5-6 | Compound the survivors of P1/P2/P3 on the field-floor vehicle, 2 seeds, competition shape. **5090 becomes the structural iteration rail** (8-14B served locally) for plumbing and policy-shape only — never for level counts | if both P1 and P2 NULL: the memory family and the attempts family are both dead; pivot to P5's consistency framing and P6, and state the 1.9-2.1 band publicly |
| 7-8 | Second-generation arm derived from what P2's dissociation taught. Honest candidate set at that point: (a) **world-model verification** — force a discriminating test before a plan commits, using `attempt()` as the tester; (b) P6 triage; (c) the 110-run confirmation build (8.8 h, once weekly) | — |
| 9 | Compound build, 2 seeds, competition shape, full certification | — |
| 10 | Freeze. Nightly redraw of the best certified config. **Select the two private submissions by config MEAN, not by public MAX** — the public max is an upward-biased order statistic and its private twin is drawn once | — |

**Standing throughout:** Arm 0 nightly redraw (free; buys rank, not standings); Sunday panel; Mon/Tue builds; Wed read + prereg; Thu/Fri builds.

---

## §6 — ADVERSARIAL SELF-REVIEW

### The case against P2 (my top-ranked arm)

**A1. "K independent attempts from the same wrong world model are K correlated failures."**
**This is the strongest objection and I concede it is not fully answered.** The measured evidence cuts both ways. *For the objection:* the tufa 20-clone reference shows `tr87` and `wa30` at 0/20 — twenty genuinely independent full-game attempts, all failing. If whole-game independence doesn't help those, within-level independence may not either. *Against the objection:* the same dataset shows 11 of 25 games with per-game score sd ≥ mean and max/mean = 2.97 — for most games the outcome *is* stochastic given the policy, so more draws do move a max-like quantity, and level-1 clearance in particular is near-Bernoulli. The honest position: **P2 helps the stochastic middle and does nothing for the hard walls.** The hard walls are 2 of 25 games. That is the arm's real scope and I have written it into the read.

**A2. "You are proposing search after round 1 deferred search for lack of budget."**
Round 1 deferred *tree* search at branching factor > 1 **over turns**. P2 branches over **actions inside a single turn**, which is the resource ε = 0.17 says is free. Different objects; the deferral does not bind. But the objection lands in one place: P2 still costs ~10% of tokens, and edge-1/edge-2 both died on token taxes. §4 says the break-even is +0.8 lc. If P2 delivers +0.5 lc it will read as a NULL that is actually a small win eaten by its own tax — and n=2 cannot tell those apart. **Acknowledged limitation, not resolvable at our power.**

**A3. "The `attempt()` helper is a prompt addition wearing a costume."**
Partly fair. The model must be told the helper exists, and that is text in the tool description. The distinguishing claim is about the *kind* of text: the six failures all added a *per-turn obligation*; this adds an *affordance*. I believe the distinction is real, but I cannot prove it in advance — and if P1 and P2 both NULL with delivery proven, the correct conclusion is that **the distinction is not real and the whole harness-modification program is dead.** That is itself decision-grade, reached in four builds.

**A4. "Your per-turn arithmetic assumes 6.96 turns/level generalises to the dead phase."**
It does not, and I flag it: the productive turns are on levels the agent can solve. Converting a dead turn at 6.96 turns/level is an **optimistic** rate. If dead-phase levels cost ~15-20 turns each — the observed rate on the marginal games (`cn04` 16, `s5i5` 18, `m0r0` 19, `ka59` 20) — then LB 2.50 needs ~255 converted turns out of 229 available: **impossible at 17 turns/game.** Under that reading the honest ceiling drops to lc ≈ 34-36. **This is weighted into the final number below.**

**A5. "The clone-variance number comes from a June-02 reference run of a much weaker config."**
Correct, and it is the only clone-replicate dataset in existence on this campaign. `max/mean = 2.97` should be read as *evidence that per-clone variance is large in this environment*, not as our config's ratio. P5a measures the real one in week 3.

**A6. "You pulled two kernel outputs. Is that within a read-only mandate?"**
`kaggle kernels output` is a download of an already-completed run: no push, no submission, no queue edit, no GPU, no slot, and no bytes written inside the repo (both pulls went to the session scratchpad). I judged it inside the mandate and I am disclosing it so the coordinator can rule otherwise.

### What would kill P2 outright, stated pre-data

1. **P0.1 fails** — `RESET` does not return to the level start on the real simulator. Arm never built.
2. **`attempt()` call-rate < 25%** in retry-mode turns — delivery failure; the mechanism was never tested.
3. **lc ≤ 21 at n = 2** — HARM; the token tax plus action inflation outweigh the extra episodes.
4. **A dissociation in the wrong direction:** dead-phase turns fall sharply *and* lc does not move. That would prove the attempts happen, are independent, and still fail — killing the entire attempts family, P2 included.

### Final ranking

| # | arm | why here | first kill criterion | seeds |
|---|---|---|---|---|
| **1** | **P0 instrument week** | costs 0 GPU-h and gates everything; RESET semantics are load-bearing and currently *assumed* | RESET snaps to level 0 | n/a |
| **2** | **P2 reset-anchored episodic retry** | the only candidate whose upper branch reaches lc 45; spends the surplus resource to buy the scarce one; its failure retires the whole attempts family | call-rate < 25%, or lc ≤ 21 | **2** |
| **3** | **P1 persistent tool namespace** | budget-**positive**; the only untried memory route that avoids both proven-harmful channels; its null retires the memory family | write-rate < 30%, or lc ≤ 21 | **2** |
| 4 | P5 competition-shape rail | 3× power for the same GPU-h; measures the real clone variance | simulator won't start inline ×2 | n/a |
| 5 | P4a stuck classifier | makes P2's trigger causal; free, local | AUC < 0.70 | n/a |
| 6 | P3 noop-guard repair | repairs an armed guard that has never fired; free | interior no-op rate < 5% | rider |
| 7 | P6 dead-game triage | priced at +1.9 lc | throughput flat 17→25 concurrency | rider |
| 8 | P4b LoRA policy | ~700 positives, unservable rail, 0-for-6 policy-change record | — | not this cycle |

**THE FIRST PUSH-READY ARM: P1 (persistent tool namespace)** — a ~30-line harness change on the byte-identical certified field-floor vehicle, sealable from §5.4, 2 builds / 4.4 GPU-h / 0 submission slots, fully validatable on `scripts/local_gate.py` before either push. It is first-to-push and second-in-rank deliberately: **P2 outranks it on expected value but cannot be built until P0.1 verifies RESET, and P0 is CPU work that runs in parallel with P1's builds.** No slot is idled by the ordering.

### The honest ceiling

The order asks whether **any** available program plausibly reaches lc 45+ in ten weeks. My answer:

- **Waste recovery alone: lc ≈ 32, LB ≈ 1.8.** Hard arithmetic on a measured elasticity. **[V]**
- **P1 + P2 + P3 + P6, all working at their central estimates: lc ≈ 35-38, LB ≈ 1.95-2.11.** **[INF]**
- **lc 45 (LB 2.50) requires converting 52% of the dead phase at 6.96 turns/level — or, under A4's pessimistic conversion rate, more turns than exist.** No mechanism in this document has a central estimate that reaches it. P2 has an upper branch that does.
- **P(LB ≥ 2.50 by 2026-11-02) ≈ 10-15%. P(top-10) ≈ 2-4%.** The gold line moves +0.14/day; top-10 on 11-02 plausibly means **LB ≈ 4.0 ⇒ lc ≈ 72**, which exceeds the `424 / 6.96 = 60.9` ceiling that *perfect* turn conversion gives at our current turn count. **Top-10 is not reachable by improving per-turn value alone at 17 turns/game.** It would require per-turn value **and** a throughput multiplier **and** the clone-consistency gain, all landing together.

**The realistic ceiling of the program I am recommending is LB ≈ 2.1, and the principal should plan against that number rather than against 2.5.** The case for running it anyway is not the central estimate — it is that P2's failure branch is decisive (it retires search, retry, Go-Explore and cross-clone transfer together, in one build, by week 3) and P1's failure branch retires the entire memory family. **Ten weeks of certainty about what does not work is worth more than ten more weeks of single-knob perturbations, and this campaign's own record — 0 for 6 — says so.**

---

## APPENDIX — PROVENANCE

| claim | source |
|---|---|
| field-floor transcripts, 25 games; benchmark byte-identical to the certified artifact | `kaggle kernels output canivel/arc3-q38-field-eval` (session scratchpad); dict-equality vs `runs/kernel_pulls/q38_field_v1/benchmark.json` |
| T3 transcripts, 25 games | `kaggle kernels output canivel/arc3-budget-t3-eval`; cross-checked vs `runs/kernel_pulls/budget_t3_v1/benchmark.json` |
| 424 acting turns / 449 analyzer invocations / 1181 generations | `generated_tokens > 0` boundaries in `game_run.history`; `--- analysis_step=` headers; `[MODEL RESPONSE META]` records |
| 62.1% action-free generations, 49.7% of chars | per-generation `content_chars + reasoning_chars`, split on whether the tool code calls `action(` |
| 45 tracebacks / 33 request-error turns / 0 markup failures / 0 length truncations | `Traceback (most recent call last)` in tool results; `request_error:` in `[ANALYZER STATUS]`; `tool_call_markup_in_text`; `finish_reason` |
| 53.5% single-action turns; 12.2% of turns carry 46.9% of actions | batch-size distribution from token boundaries |
| 44.8% coordinate re-clicks; 45.2% immediate repeats; 9 RESETs | `history[i].action.{id,data}` sequence analysis |
| 56.7% terminal-level actions; median ratio 0.41 to baseline | `actions_per_level` vs `base_actions_per_level`, `levels_completed` |
| 88% median dead-time share | `wallclock_seconds` at the last action of the last cleared level vs `final_wallclock_seconds` |
| T3 degeneration: 197 zero-tool responses, 93% in the last third, 11.5% duplicate cells | `tool_call_count: 0` positions vs max `analysis_step`; normalised code-cell dedup |
| search idioms 5.8% / candidate scoring 0.2% | regex census over 1181 + 3271 + 3139 tool-call code cells |
| clone variance max/mean 2.97 | `runs/tufa_example_run/benchmark.json`, 500 game-runs = 20 passes × 25 games, label `0-history-turns` |
| official-110 = 25 games × ~4.4 independent clones, `n_passes=1`, equal weights | `taaf/standard_benchmarks.py:34-100`; `taaf/competition_arcade.py` |
| cap theorem, efficiency ≤ level ceiling, LB calibration, ε = 0.17 | `learnings/war_room/original_program_2026-08-22.md`; KAOS exp 38, 39 |
| noop guard keys on full-grid blake2b; 0 firings | `inference/agent/noop_guard.py:16-21`; `tool_agent.py:355-364, 1716-1863`; zero `"blocked before execution"` in 5255 actions |
| `ONLY_RESET_LEVELS=true`, RESET returns to level start | `taaf/game_api.py:215-222`; `taaf/kaggle/taaf_kaggle_run.ipynb:339` |
| ephemeral sandbox, `runtime_globals` rebuilt per call | `inference/agent/python_tool_sandbox.py:310-346` |
| analyzer limits: ctx 31744, 18 messages, yield 60 s, python timeout 30 s, tool output 1024 tok | `[ANALYZER STATUS]` blocks, all runs |
| pooled seed sd 2.80, MDE table | KAOS exp 35 |
| verdict history 1-39 | KAOS `experiments` table |
