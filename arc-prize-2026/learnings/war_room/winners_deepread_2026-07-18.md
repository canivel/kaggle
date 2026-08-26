# Milestone winners deep-read — 2026-07-18

The never-executed deep-read flagged in `discussions_2026-07-16.md` (Actions 1+3). All three
winner notebooks pulled and read end-to-end; current public Tufa source dataset downloaded and
diffed against our bundle. Naming note: these are the **June-30 Milestone-1** winners announced
in disc 725002 (the only milestone whose winner code is public; Milestone-2 closes Sept 30 —
no M2 winner artifacts exist yet). Read-only ops only; no pushes, no submissions, no API spend.

Artifacts (all local):
- `runs/winner_pulls/duck_public/` — `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` (1st, milestone 1.21)
- `runs/winner_pulls/reki/` — `ruichardliu/milestone1-2nd-solution` (2nd, 0.86)
- `runs/winner_pulls/forge3rd/` — `mbmmurad/arc-agi-3-lb-0-86-3rd-place-candidate-milestone` (3rd, 0.86)
- `runs/winner_pulls/*/[name].py` — flattened cell dumps; `runs/winner_pulls/our_duckwar.py` — our fork flattened
- scratchpad `taaf_current/` — current public `jeroencottaar/taaf-kaggle-source-share` (diffed, then disposable)

---

## 1. Drift-diff verdict on our fork: **NO UPSTREAM DRIFT — CLEAN**

Three-way comparison (public notebook vs `notebooks/duckwar/arc3-duck-war.ipynb`; public
dataset vs `duck_eval/taaf_bundle/`):

1. **Notebook:** cell-for-cell identical except our four intentional grafts — (i) `RUN_HEAVY`
   fast-submit gate (R1a), (ii) warpack customization hook in cell 12, (iii) `soft_end` =
   start+11h20m in a real rerun (R1b; vanilla passes `None`), (iv) extra dataset
   `canivel/arc-war-kit`. Nothing upstream changed under us.
2. **Dataset:** every file in the current public `taaf-kaggle-source-share` is dated
   **2026-06-12** (same version we forked). The only diffs vs `duck_eval/taaf_bundle` are OUR
   local additions flowing the other way: the GPT-5.6 probe spend guard + `openai-api`
   provider branch in `tool_agent.py`/`openai_compat.py`, and our
   `requires-python` widening in `pyproject.toml`. Public upstream = pristine base.
3. Kaggle-side risk is nil anyway: our kernel mounts the upstream dataset slug directly and
   patches at runtime, so a silent upstream dataset update is the only vector — none occurred.
4. Caveat carried from Tufa's own note: the notebook that actually SCORED 1.21 is the messier
   `jeroencottaar/taaf-duck-harness-kaggle`; the shared readable one "hasn't had the same
   lucky result". Their 1.21 was itself an order-stats draw, by their own admission.

**Standing action retired:** discussions_2026-07-16 Action 1 (drift check) is DONE — no change needed.

---

## 2. Per-winner mechanism inventory

### 2.1 Tufa Duck (1st, milestone 1.21) — our own base
Notebook is pure infrastructure (wheel install, bundle import, pickled benchmark, gateway wait,
teardown); solver = the dataset we already run and have distilled extensively (`duck_eval/`,
`gpt56_distill_*`). Confirmed from `setup_commands.json`: **Qwen3.6-27B-FP8**
(`vrfai/Qwen3.6-27B-FP8`), vLLM 0.19.0 + torch 2.10 + flashinfer 0.6.6 wheelhouse,
`max_model_len` 65536, analyzer context window 32768, TP=1, RTX Pro 6000. Mechanisms (from
writeup 717133 + source): REPL/python-tool over ASCII game state, multimodal perception
(image + ASCII + segmentation zoom), infinite play via message eviction, `ONLY_RESET_LEVELS`
pinned. Their stated gains: "multimodality + better base models, not hand-built tools" —
i.e., the 1st-place team's own attribution is **model tier**, matching our probe decomposition.
Nothing new to extract here; the value of this pull was the drift certification (§1).

### 2.2 Reki (2nd, 0.86 = v5; v7 0.00 was infra per his own post)
Different harness family from the duck: official ARC-AGI-3-Agents template + Swarm (one agent
thread per game), single-file `MyAgent`, **Gemma-4-31B-it vision policy** under vLLM 0.23,
bf16 ("auto"), 32k ctx, `max_num_seqs` 20, prefix caching, JSON `response_format`, images =
last 4 frames scaled 8x with red STEP labels.

| # | mechanism | detail |
|---|---|---|
| R1 | JSON action plan | one user message (4 images + text prompt) → JSON `{board_change_assessment, plan_summary, actions[≤4]}`; plan queue dequeued action-by-action; queue invalidated on reflection, level change, or unavailable action |
| R2 | JSON repair pass | on parse failure, second `/no_think` call (256 tok) with the bad output quoted; robust brace-scan JSON extractor |
| R3 | Reflection memory | every 10 completed transitions: `/no_think` call (10k tok budget) rewrites a **bounded 1800-char** markdown memory with fixed headings (Rules/Goal/Progress/Avoid); REPLACES previous memory (revisable), persisted per-game file; injected into every action prompt as "authoritative but revisable" |
| R4 | Per-state no-effect refutation (base) | `_observe_frame` diff-attributes every executed action: `changed_pixels` (border-trimmed grid compare) + `levels_delta`; zero-effect ⇒ `failed_state_actions[frame_hash] += failure_key` (`click@x,y` or action name); enforced 3 ways: prompt line "Ineffective in this exact state", hard skip of planned actions, fallback exclusion. Per-level reset |
| R5 | **Dead-signature tracker (V25-fusion, his 0.64→0.86 delta)** | see §4a — structural-class click suppression |
| R6 | Salient-click fallback (V25-fusion) | fallback clicks pick most "button-like" component: 4-connected same-color components, saliency = 0.5·color-rarity + 0.5·size-tier (≤4 px best); skips dead-sig classes and per-state failed coords; falls through to random non-zero pixel |
| R7 | Budget management | global 9h limit with **20-min shutdown reserve** shared across the swarm (`_SUBMISSION_STARTED_AT` module global); per-game 8h; per-request timeout = min(configured, remaining); `FIRST_ACTION_DEADLINE_S` 14min |
| R8 | First-action RESET before model boot | action 0 is always RESET, returned BEFORE `_ensure_vllm_available()` — "the gateway needs early activity"; vLLM startup failure is sticky (`_vllm_startup_error`) and the agent then lives on fallback cycling instead of hanging |
| R9 | GAME_OVER handling | `NOT_PLAYED`/`GAME_OVER` ⇒ clear plan queue + immediate RESET; no stop-acting language anywhere (no deadlock class) |
| R10 | Sampling | temp 0.6 think / 0.2 no-think (actions default `/no_think`), top_p 0.95, top_k 20, **repetition_penalty 1.08**; `n>1` unused |
| R11 | Seeding | `random.seed(time*1e6 + hash(game_id))` — **nondeterministic** (his own variance: byte-identical logic ranged 0.00–0.86; he attributes 0.00 to infra) |
| R12 | Legality | fully game-agnostic; per-level resets only; no per-game branches. LEGAL |

Footgun observed: his current public kernel-metadata has `enable_gpu: false, machine_shape: None`
— forking it verbatim reruns on CPU and dies (our `feedback_kaggle_env_match` 5x lesson, now
visible in a winner's own artifact).

### 2.3 forge / mbmmurad (3rd, 0.86)
**Same code family as Reki's base** (identical class skeleton, prompts, observe/refute loop,
reflection, budgets — clearly a shared public lineage). Profile that SCORED:
`forge_v46_gemma31b_public_single` = **everything off**: `LLM_ACTION_CANDIDATES=1`,
`LLM_CANDIDATE_ARBITER=0`, `LLM_CONFIDENCE_PROMPT=0`, `LLM_INCLUDE_FRAME_DESCRIPTOR=0`,
`LLM_CLICK_FAILURE_RADIUS=0`. The machinery exists in the file (3-candidate sampling via
`n`, static score = confidence + effect-priors, LLM arbiter pass, JSON frame descriptor,
click-failure radius veto) and was all disabled for the milestone run. Same Gemma-4-31B, same
sampling, same budgets. Deviations from Reki:

| # | mechanism | detail |
|---|---|---|
| F1 | **Deterministic RNG** | `sha1(agent_name:game_id:AGENT_RANDOM_SEED)` seeds a per-instance `random.Random` — the whole run is reproducible modulo LLM sampling; directly answers the #726552 unseeded-ACTION6 variance thread |
| F2 | Deterministic fallback clicks | no random pick: fallback cycles **component centers sorted by area (smallest first)**, index = action_counter mod n — a deterministic sweep over clickable objects |
| F3 | Commit-mode smoke test | non-rerun commit path: imports the agent under the official framework, boots vLLM, runs one image-generation smoke request, then tears down — a built-in `feedback_test_before_submit` |
| F4 | Simplicity as the win condition | his own writeup: local public-suite checks "were not a perfect leaderboard proxy"; the scored config is the one with all cleverness off |

`MAX_ACTIONS = 200` is defined but dead code in both (never checked) — neither 2nd nor 3rd
place has an action-budget sentinel; their budget discipline is wall-clock only.

---

## 3. ADOPT / ADAPT / IGNORE ledger (grounded in our evidence base)

Failure-pathology buckets per GSME adoption (research_2026-07-18): budget-death /
verbatim-resubmit / game-over-deadlock / stuck-loop / infra-death.

| mechanism | verdict | bucket | reason (our evidence) | counting Δ (rail) |
|---|---|---|---|---|
| R5 dead-signature class suppression | **ADAPT** | verbatim-resubmit + stuck-loop | Superset of our (c) fingerprint + (d) no-effect FACTs; class key covers the 16–32 same-coord re-clicks AND their twins (`transcript_forensics.md`); ships inside the already-registered (c)+(d) single flag as an extra counter, not a new window | ceiling unchanged **+0.10** (conversion channel identical: reclaimed actions pay only via clear-faster/clear-at-all, §2 grinder_cracking_design); expectation +0.02–0.05 |
| R4 per-state failure keys w/ 3-point enforcement (prompt + plan veto + fallback filter) | **ADOPT (design detail)** | verbatim-resubmit | This is exactly (d)'s no-effect FACT, independently converged on by BOTH runner-up teams and enforced mechanically, not by prose — external validation that the (c)+(d) flag is the right first window; hard-veto enforcement is what our ledger lacked (1552 digests / 0 escalations = prose-only trigger) | inside (d) ceiling +0.08 |
| R8 first-action RESET before model boot + sticky startup-failure + fallback survival | **ADOPT (hygiene)** | infra-death | Host post #727119: ~1/3 of all failures = silent stuck; our v38 0.00 and Reki's v7 0.00 are this class; costs nothing, claims nothing | ≈0.00 per-draw mean; removes a 0.00-draw class |
| R7 global shutdown reserve + per-request timeout=min(remaining) | **ADOPT (hygiene)** | infra-death | Same class as our R1b soft_end 11h20m — extend to the request layer (a hung last request can still eat the drain window) | ≈0.00 |
| F1/F2 deterministic seeded RNG + deterministic fallback sweep | **ADAPT (rail only)** | measurement | N5: env is frame-deterministic; agent-side seeding closes the last nondeterminism gap → gate power (build-rail 3-seed MDE). Do **NOT** ship to LB draws: card score = max over plays, so draw variance has positive order-stats value (prereg 07-14: ceiling 1.11@k=110 is the *limit* of that value, not a reason to forfeit it) | 0.00 mean; instrument value only |
| F3 commit-mode smoke test | **ADOPT (checklist)** | infra-death | Literally our `feedback_test_before_submit` + preflight.py, already policy; confirm preflight covers vLLM-boot smoke | 0.00 |
| R3 bounded self-REPLACING reflection memory (1800 chars, fixed headings) | **IGNORE (note the design)** | — | Our ledger-as-built was REFUTED 07-17 (always-on context tax, 0 escalations); Reki's differs in one way worth remembering — replace-not-accumulate with a hard char bound — but it is still prompt-mediated advice, and `feedback_prompt_is_noise` + the distill's "exhortation is inert" finding say Qwen won't convert it | unbounded claims rejected |
| R6 salient-click fallback | **IGNORE** | stuck-loop | Fires only on fallback/empty-plan turns; the duck REPL agent computes coordinates deliberately — trigger count on our rail ≈ 0, and A10 forbids windows whose trigger can't fire | no firing trigger |
| F multicandidate + arbiter + confidence + descriptor | **IGNORE** | — | The 3rd place's own winning move was turning these OFF; external confirmation of `feedback_simplicity_wins` and of gating (b)'s token tax behind non-inferiority | negative-risk avoided |
| R1 plan queue (≤4 actions/call) | **PARK for war-v4** | throughput | Duck's REPL already batches; but for the 72B line, actions-per-LLM-call is the exact lever the throughput guard watches — note as a v4 mitigation if decode throughput binds | v4-only |
| R10 repetition_penalty 1.08 | **IGNORE** | — | Sampling-param tinkering = prompt-noise class; `feedback_vllm_params` says match reference exactly, and our reference is the duck's own config | — |

---

## 4. Special-attention items

### 4a. Reki's dead-signature suppression — exact implementation, vs our (c)
Implementation (verbatim from `reki.py` lines ~1505–1560):
- **Key** = structural signature of the 4-connected component under the click:
  `(color, size, is_rect, twins)` where `is_rect` = pixels fill the bbox and `twins` = count
  of other components with identical (color,size,is_rect). Background clicks → no key.
- **Update rule**: after every click, diff-attribute (`changed_pixels`, `levels_delta`).
  Any effect ⇒ signature goes into a **protected set forever** (a class that EVER worked is
  never suppressible — protects the win-button class). Zero effect and not protected ⇒
  counter++; at **K=2** the class enters `_dead_sigs`.
- **Enforcement**: (i) saliency fallback picker skips dead classes; (ii) separately flagged
  hard **veto of the LLM's own planned clicks** on dead classes (`USE_DEADSIG_VETO_LLM`,
  kept separable because twin-generalization can over-suppress in tile/maze games).
- **Scope**: per-level reset of all three sets (an L0-inert class can be the L1 win class).
- **Attribution**: his notebook states the deadsig+salient pair is the delta over a
  "byte-identical 0.64 base" → **+0.22 LB at Gemma tier** (caveat: single-draw LB deltas at
  his own sd are weak evidence; but the mechanism direction matches our forensics exactly).

**Mapping to our (c) submission-fingerprint:** it is the same refutation idea, one level of
abstraction up — ours keys on exact byte-identical submissions/coordinates; his keys on a
structural **equivalence class**, which is what converts the forensics' "same slot re-clicked
16–32×/seed *plus its twins*" into one suppression each. His protected-class rule and
per-level reset are the two safety valves ours lacked. Verdict: **his is better as the click
component of our (c)+(d) flag**; adopt the class key as an additional refutation-record type
(fingerprint-block counter stays separate per the pre-registered counter split, so a gate pass
still decomposes mechanically). It does NOT replace (c)'s arrangement-resubmit fingerprint
(sb26's arrangements aren't clicks on one component) — the two are complementary within the
same code path (harness diff engine writing FACT records).

### 4b. What the 1.44+ band actually does — synthesis
The deep-read's sharpest finding is negative: **the entire public winner tier tops out at
0.86–1.21, and nothing in any public artifact explains 1.44–1.86.** Current LB (pulled today):
1.86 / 1.61 / 1.60 / 3×1.56 / 2×1.54 / 1.50 / 1.48 / 1.47 / 2×1.46 / 1.44×4 — ~14 teams at or
above the wall, none of the three winners among them, none sharing code above 0.86. What the
public evidence does triangulate:
1. The 1st place's own attribution for their tier: **multimodality + better base models, not
   hand-built tools** (writeup 717133).
2. Both runner-up teams independently converged on **mechanical no-effect refutation** as their
   only positive-delta scaffold — the same mechanism class as our v3 (c)+(d), worth +0.1–0.2 LB
   at their tier, not +0.6.
3. The wall-breakers are all high-volume submitters on an opaque private set (YUTO 40 entries;
   host confirms nobody can see anyone's code pre-open-source).
Conclusion: 0.86→1.21 is scaffolding+luck; **1.21→1.44+ is per-draw mean (model tier and/or a
non-public mechanism) compounded by draw volume**, consistent with the 07-14 reconciliation
(order-stats ceiling ~1.11) and with grinder_cracking_design's verdict that war-v4 (72B-tier
model swap) is the only registered wall-closer. Nothing found here changes that; two winners'
artifacts independently corroborate it.

### 4c. Executable-world-model sign (OPINE-World family)
**None.** Reki/forge are pure vision-policy + text reflection (no transition model, no replay
check, no program synthesis). The duck's python-REPL-over-state is the closest public relative
of an executable substrate but has no learned transition function and no CEGIS loop. No public
competitor artifact implements anything OPINE-shaped — if OPINE-World (20/25 games, arXiv
2607.01531) transfers to the private set, its replay-check extraction into our (c)/(d) ledger
remains an uncontested edge, not a catch-up move.

---

## 5. Final statement

**Top 3 adoptable mechanisms (counting bounds in rail units, scorer arithmetic from
`grinder_cracking_design.md` §2):**
1. **Class-signature refutation graft** (Reki R5+R4: structural-class key, protected-class
   rule, K=2, per-level reset, hard plan-veto) into the already-registered (c)+(d)
   mechanical-refutation flag — ceiling **+0.10 rail ≈ +0.06 LB**, honest expectation
   +0.02–0.05 rail; no new window consumed; two independent winners validate the mechanism
   class and the hard-enforcement (not prose) design our failed ledger lacked.
2. **Infra-death hygiene set** (R8+R7+F3: first-action RESET before heavy boot, sticky
   startup-failure with fallback survival, request-level timeout = min(remaining), commit-mode
   smoke) — **≈0.00 per-draw mean, removes the 0.00-draw class** that is ~1/3 of all failed
   submissions (host data) and both of our and Reki's observed 0.00s; ships with the (f)-class
   unflagged hygiene window.
3. **Deterministic agent seeding on the build rail only** (F1/F2: sha1(agent,game,seed) RNG +
   deterministic fallback sweep) — **0.00 mean, pure instrument value**: closes the last
   agent-side nondeterminism the N5 audit couldn't control, tightening the 3-seed gate; kept
   OFF for LB draws where variance retains order-stats value.

**Drift verdict: CLEAN.** Public duck notebook and source dataset are byte-equivalent to our
fork base (dataset frozen at 2026-06-12); every observed diff is our own intentional graft or
local probe instrumentation. No re-fork, no rebase, no action required — the eternal-fallback
frozen fork remains valid.
