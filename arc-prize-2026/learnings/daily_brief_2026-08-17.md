# DAILY BRIEF — 2026-08-17 (Monday)
**Protocol:** canonical daily loop. Weekday ⇒ **no panel round** (full panel is Sundays only, per the 07-27 addendum). Weekly fingerprint/dream ⇒ Sunday only, not today.
**Slots:** 08-17 was **2 of 2 SPENT** by 09:00 (the Q38-low push + the misfire push to `arc3-q38-engine-eval`). **Nothing was pushed today after that ruling, and nothing was submitted beyond the 00:07 auto-refill.**
**Provenance tags:** **[V]** verified by direct read/sha/download this session · **[V-doc]** verbatim claim inside a verified artifact · **[INF]** inference · **[UNK]** unknown.

---

## 1a. RESULT DEEP-DIVE

### The draw
**0.80** (`canivel/arc3-duck-repro` v3, frozen-fork filler, 2026-08-17T00:07:11Z, COMPLETE). **[V]**
Ledger re-derived from the API: **n=34, mean 0.9382, s 0.1559**, **z(0.80) = −0.91**, trailing-4 0.89, range 0.65..1.33, sealed mean-of-4 promotion bar **1.0777** (was 1.0826 at n=33 — *the bar drifts; re-read `runs/ledger.json` at prereg time, never cache it*). Public max **unchanged at 1.33 since 07-18**. **[V]**

**Interpretation, not the number:** an interior draw from the frozen fork's own distribution, −0.9σ. It carries **zero information** about anything we built — it is the fifth consecutive day where the only thing on the board is the eternal fallback. The pre-registered expectation for a filler draw ("interior, no signal") was met. **No action follows from it, and none should.**

### The Q38 low arm — INFRA DEATH, and the root cause is the day's most transferable lesson
Sealed and journaled as exp_id 10 (`VOID: INSTRUMENT MIS-SPECIFICATION, not a mechanism finding`). Died at t=415s, zero games, ~7 min GPU. **The kernel died because the pin WORKED:** `--default-chat-template-kwargs` bound `reasoning_effort=low` exactly as designed, the served prompt therefore contained "Reasoning effort is set to low", and the pin-verification gate — still asserting the MEDIUM arm's *silence* signature — read the arm's own intended behaviour as "did not bind" and raised FATAL. **[V]**

**The lesson, stated generally because it recurred today in a second form:** *a gate's LOGIC must be invariant across arms, but its EXPECTED VALUES are a function of the arm.* Freezing both converts a correct poisoning gate into a landmine that fires precisely when the arm works. The smoke's replay stub agreed with the gate and both disagreed with reality — **internal consistency is not correctness.** Fixes shipped and negative-controlled (arm-aware markers; silence is no longer the definition of success; 113/0 both arms). **v2 artifact `7c0de2a6fcc121cf` is built and NOT pushed** — it needs tomorrow's slot 2 and the three deliberate steps.

### ★★★ THE DAY'S REAL RESULT: the authorized 08-18 arm was falsified before it was built
Yesterday's conversion trace (exp_id 11) traced the field's 2.5+ recipe to a public CC0 graft stack and the coordinator authorized an 08-18 separation test: `banking + transfer + shortcircuit`, SCORE-primary. **Today's build-blocking audit (`duck_eval/graft/bundle_audit_2026-08-17.md`) killed that arm on reachability, and the evidence is entirely our own:**

| finding | evidence |
|---|---|
| `banking` gates on `run.state == "won"` (`banking_solver.py:180`) | **[V]** |
| runs reaching `won` across **every eval artifact this campaign has ever produced** — 23 pulls, **470 game-runs** | **ZERO** **[V]** |
| best `levels_completed` on any single game, ever | **4**, of 6–10 **[V]** |
| `transfer` needs clone siblings; our eval rail is `n_passes=1`, 25 games, **25 unique `game_id`s** | **[V]** |
| its own docstring on a non-clone set: *"turns the entire stack into a measured no-op"* | **[V-doc]** |

⇒ **The authorized arm reduces by construction to `shortcircuit` alone.** It would have returned a REFUTE carrying no information about the mechanism — *the A9/warpack error verbatim* ("our gate measured LEVELS on an offline bench where banking's conditions never fired"). **We were one slot away from booking the same mistake a second time, against the same mechanism, five weeks later.**

**★ And it retro-explains our July result with far more force than N5 did:** `war_eval_v1` — our own warpack banking arm — posted the **highest lc of any run on disk (22)** and still banked nothing, **because it had zero wins to bank, not because the lane was wrong.**

**★★ THE STRATEGIC REFRAME, which outranks the arm.** Banking multiplies the score of cards you **already win**; it is a denominator exploit on cleared content. We clear 1–2 levels of 6–10 and have never won a single card. **So the field's recipe is not a shortcut past our problem — its payoff is gated behind exactly the capability we lack.** For us the 1.33 → 2.00 gap remains, first, a **capability** gap; the exploit becomes monetizable only after capability arrives. *This does not contradict the trace's attribution of the field's jumps (still [INF ~85%]) — the 2.0+ teams presumably do win cards. It changes what is liftable BY US.* **[INF on the reframe, [V] on every input]**

### What was built today instead (all local, nothing pushed)
Amended arm **`arc3-graft-floor-eval`** = the **reachable** public floor, thtennant's published v19 set: `{efficiency, retry_guard, shortcircuit, goalkeep, hudmask}`, `banking`/`transfer` OFF **and asserted absent**, incumbent Qwen3.6, LEVELS-primary. Prereg sealed pre-push: `learnings/war_room/graft_floor_prereg_2026-08-17.md`.
Gates, all green: **builder** (fail-closed, idempotent, code sha `3c047dff2e6c02fd`, differing cells exactly [2,6,12]) · **smoke 36/36** · **sealed scorer selftest 22/22** · **bundle re-audit OK on a fresh download** (byte-identical to the audited 08-17 00:26 version). Preflight is post-push by design (it pulls from Kaggle; the slug does not exist yet). **[V]**

Three defects the audit caught that would each have produced a *silent* wrong answer:
1. **The bundle-marker collision.** Cell 6 resolves the bundle by first `rglob` hit of `taaf-kaggle-bundle.json`; the fork carries that marker too. Attaching both bundles ⇒ ambiguous `BUNDLE_DIR`, and a stock resolution runs **stock** while looking like a clean arm. Resolution: **REPLACE**, corroborated by thtennant's own v18/v19 metadata (exactly 3 datasets, fork replacing stock). **[V]**
2. **`install()` silently ignores unknown flag names** — no validation. A typo = a clean-looking stock run. Hence the scorer certifies the **runtime banner**, never the source. **[V]**
3. **A sealed constant that disagreed with its own derivation:** the Q38 prereg's `0.286320` ≠ C(3)·σ̂ = 2.02 × 0.141740 = **0.2863148**. 5×10⁻⁶ lc/game, moves no boundary — but `graft_score.py` now **derives** it, so the class cannot recur. **[V]**

Also corrected from the trace: `install()` lives in `taaf_grafts.composite`, **not** `__init__` (which exports only `BankingHarnessSolver`) — the wrong import path is a silent stock fallback; `hudmask` is **nested under `goalkeep`** (arming it alone does nothing); and `transfer` **implies** `banking` (`composite.py:172`, confirming the trace).

### ★ INSTRUMENT/INFRA FINDING: the KAOS-native rail had never actually run
The 08-16 mandate routes every substantial task through `kaos run -m fable-panel`. **The first two spawns under it failed with `claude-agent-sdk not installed`, produced 0 bytes, and exited 0.** **[V]** Fixed (`uv pip install claude-agent-sdk` → 0.2.139) and the rail now answers. **But two limits are now measured, not assumed:**
- `fable-panel`/`opus5-panel` use the `agent_sdk` provider, which is **text-only**: *"agent_sdk provider cannot forward 8 OpenAI-style tool schema(s); the call proceeds text-only."* A probe agent correctly reported it could not read a file and **refused to fabricate a line count** — good behaviour, but it means these agents **cannot** do file-diffing, sha work, kaggle-CLI pulls or web sweeps. **[V]**
- The only tool-capable model (`claude-sonnet`, `claude_code` provider) **also failed**: `claude produced no output for 60.0s`. **[V]**

⇒ **Today's audit and sweep were executed inline** (verification is the orchestrator's own sanctioned role), and that is why. **This is a live blocker on the KAOS-native mandate's mechanism, not its intent** — the journal/experiment/bench half works and was used (exp_id 10, 11). Fourth instance this week of *silence from an automation is not success* (refused scheduled task → untested scorer → the differ's cp1252 crash → now a 0-byte agent exiting 0).

## 1b. DISCUSSIONS SWEEP

Route: `kaggle competitions topics list -c arc-prize-2026-arc-agi-3` (CLI **2.2.2**; note the subcommand is `topics list -c <comp>`, not `topics <comp>`). **No topic newer than 2026-08-17 03:31** ⇒ **no genuinely new thread** since yesterday's full sweep. **[V]** *(The CLI's `topics show` hits the same cp1252 crash our own differ did — `PYTHONIOENCODING=utf-8` works around it.)*

| topic | date | verdict + reason |
|---|---|---|
| **734843 — "Potential persistent memory issue with the Tufa Duck harness"** (Jason Feng) | 08-12 | **★ ADOPT-AS-EVIDENCE / ADAPT.** *"The Tufa Duck harness only captures persistent memory updates from visible outputs, but the model might put the updates in hidden reasoning instead… A significant portion of the updates were put in hidden reasoning and were never captured. Tufa's prompt does not explicitly tell the model to put the updates in visible output."* **[V-doc]** A **second, independent, and DIFFERENT** mechanism for our own "the agent FORGOT" root cause: `goalkeep` fixes the model being *wiped* on game-over/level-change; this says updates are never *captured* because they are emitted as reasoning content. **Our config runs `LOCAL_ANALYZER_ENABLE_THINKING='true'` [V], so the failure mode is live for us.** Public fix notebook exists (`iamjasonfeng/tufa-duck-visible-updates`). Their evidence is DeepSeek V4 Flash in online mode, so engine-transfer is **[INF]** — but the defect is harness-side and engine-independent. **Queued as the leading candidate for the next capability arm.** |
| 735590 — "run went backwards on the leaderboard" (Pengyi Peng1) | 08-17 03:31 | **IGNORE for plan, KEEP as corroboration.** A 3-sub team asking for advice; no disclosure. But their numbers are a clean independent replication of the divergence we live with: offline eval **+121%** then **+21.6%**, public **0.25 → 0.74 → 0.28**. **[V-doc]** Directly relevant to how much weight a single build-rail eval may carry. Nobody has answered with anything substantive. |
| 735479 / 735381 / 735243 / 735147 | 08-14..16 | **IGNORE** — already swept and dispositioned in yesterday's trace; nothing new. |
| — | — | **The forum still discloses NOTHING about banking/transfer/grafts.** Re-verified. The recipe continues to move through the kernel-fork/dataset-attach graph, silently. **[V]** |

## 1c. RESEARCH SWEEP

| item | verdict + reason |
|---|---|
| **Prime Intellect "Prime Agent" — 95.5% on ARC-AGI-3, "beats the 95.4% human baseline", all 183 levels** (Aug 5) | **IGNORE — NON-COMPARABLE, and the harness hypothesis is already refuted.** Our standing finding holds: every published ARC-AGI-3 headline measures something other than the Kaggle scored rail (cf. MAP's "22/25" = beat-ReAct rate). Separately, the trace **[V]**-refuted "a different harness (Tycho/Prime port)" — *every* observed public artifact in this competition is the same duck/TAAF family. A 95.5% on the public 183-level suite is not a 2.76 on a private, 8h-GPU, no-internet rail. **Do not cite it as a target.** |
| **WebClipper — graph-based trajectory pruning for web agents** (2602.12852) | **ADAPT (parked, named).** The closest published analogue to banking's prune-and-replay, and it confirms the technique is a studied one rather than folklore. Relevant **only** to the generalization the public graft code does *not* implement — pruning a **partial** clear rather than a win. That is the one route by which banking's mechanism could ever apply to us at current capability, and it is a build lane, not a lift. |
| **AgentDiet / Trajectory Reduction** (2509.23586) — removes waste context, **−39.9…59.7% input tokens** at equal performance | **ADAPT, low priority.** Aims at our documented 31,744-token ceiling. But our binding constraint is levels, not token cost, and our own `effnote` lane already returned NO-PROMOTE on an efficiency-note arm. Keep on the shelf. |
| **AgentHER — hindsight experience replay, trajectory relabeling** (2603.21357) | **IGNORE.** Training-time method (+7.1–11.7pp via SFT). We have **no training rail** — `feedback_arc_zero_budget` forbids cloud spend and there is no fine-tuning path on the free build rail. |
| **Causal Agent Replay — counterfactual failure attribution** (2606.08275) | **IGNORE for now.** Diagnostic tooling; our diagnosis is not the bottleneck (we know the agent forgets and never wins a card). Revisit if a capability arm SIGNALs and we need per-game attribution. |
| Targeted question — *is prune-then-replay published?* | **Answered: yes for pruning/reduction (WebClipper, AgentDiet); NO published work found on pruning-then-replaying to bank a lower ACTION COUNT in a scored environment.** **[V on the searches, [UNK] on exhaustiveness]** The field's graft stack appears to be genuinely ahead of the literature on this specific exploit. |

**Net: 1 ADOPT-as-evidence (734843), 3 ADAPT (2 parked), 5 IGNORE.** Consistent with the standing finding that most sweep items are IGNORE; the one live item came from the **forum**, not arXiv.

## 2. OPEN QUESTIONS
1. **The ruling the coordinator owes tomorrow:** run the amended reachable arm (`graft-floor`, built and gated today), or the literal authorized `banking+transfer` arm knowing it can only measure `shortcircuit`? §7 of the prereg states exactly what each does and does not settle. **My recommendation: the amended arm.**
2. **Does the agent ever reach `state=="won"` on the COMPETITION 110-run rerun?** **[UNK]** — we retain no rerun logs. This is the single cheap question that would reopen banking, and it is answerable from a rerun's log, not from an eval build. **Worth a deliberate log pull on the next rerun.**
3. **Visible-vs-hidden memory updates (734843) — is it live on Qwen3.6 in our harness?** Cheap to check offline against retained request logs before it ever costs a slot.
4. **Disposition of `arc3-q38-engine-eval` v3** (this morning's misfire, still RUNNING at last check): it is an unplanned second medium seed + decode probe. May PRIMARY-B's comparator become n=2? **Not folded in unilaterally — it was not pre-registered.** Coordinator's call.
5. **Q38-low v2** (`7c0de2a6fcc121cf`) is built, negative-controlled and unpushed. It competes with nothing for slot 1; it needs slot 2.
6. **KAOS agent spawning is blocked** (both providers). Fix, or formally amend the mandate's mechanism to "KAOS for journal/experiment/bench + memory; inline for tool work"? The intent is being met; the mechanism is not.
7. Unchanged: cstl 2.70 flat 4 days and dark; per-team attribution for every 2.0+ team **[UNK]**; the `lb_diff.py` merge-blind Δ/draw divisor is **named but not fixed**.
