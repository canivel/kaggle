# A17′′ (v2) — war-v4 72B capability screen: REPAIRED amendment (post-R16)

Filed 2026-07-22. Supersedes the gate/comparator/ρ/cost sections of
`a17_72b_screen_scope.md` (§1b, §3, GATE BOOLEAN, §5-quota) in response to the R16
panel's REJECTED-as-drafted ruling on Q6 (4/5 blocking: llm-agents N13/N14,
prog-synthesis N1/N4, methodology R1, systems #15/#16/#17). Everything NOT amended
here — weights choice (Qwen2.5-VL-72B-Instruct-AWQ, 43.0 GB, Kaggle Model source),
bench-kernel design (serve-config diff, hermes parser, max-model-len 32768,
awq_marlin), risks A–E, and the Jul 22–30 schedule skeleton — carries forward
unchanged from the v1 doc. §8.2's "materially different" definition is retained
verbatim: it drew no objection from any reviewer.

All numbers below computed from data on disk ($0, 0 GPU-h):
`runs/kernel_pulls/{war_eval_v1,war_eval_v2,war_eval_v3,w0_eval_s1}/benchmark.json`.
Compute script + artifacts: `runs/a17_repair/a17_repair_compute.py`,
`runs/a17_repair/per_seed_table.json`, `runs/a17_repair/false_nogo_bootstrap.json`.

---

## 1. THE GATE — exactly one boolean (fixes systems #15 / prog-synthesis N1)

The R16 circulation carried two contradictory booleans (Part 1 §13 vs A17′ §4), and
the §4 form made null_adj dead code: CAPABILITY(Σ≥8) as a conjunct of branch 2
implies the throughput-adjusted bar (8 ≥ 5 ≥ 4), so the gate collapsed to "Σ 72B
MAX ≥ 8" and a 72B clearing Σ=6–7 against a throttled null of 3–4 — demonstrably
superior per action — was NO-GO. **Repair: adopt systems' option (i) — drop the ≥8
conjunct from branch 2 — which is exactly Part 1 §13's own form.** Part 1 §13 and
this section are now character-identical on the boolean; the A17′ §4 text is
SUPERSEDED and void. This is the only gate boolean in the record.

```
GO  iff
    ( CAPABILITY:      Σ_g max_{s∈S72} lc72(g,s)  ≥  Σ_g max_{r∈R27} lc27(g,r) + 2    # = ≥ 8
      AND
      ACTION-PARITY:   Σ_g N72B(g)  ≥  0.90 · Σ_g N27B(g) )
  OR
    ( THROUGHPUT-ADJ:  Σ_g max_{s∈S72} lc72(g,s)  ≥  Σ_g null_adj(g; ρ_action) + 1 )  # MARGIN = +1
NO-GO otherwise.
```

- **Comparator populations (fixes llm-agents N14 / prog-synthesis N4):**
  R27 = the **4 certified 27B runs** {war_eval_v1, war_eval_v2, war_eval_v3,
  w0_eval_s1} — the v1 doc's "3 certified seeds (v1/v2/v3 + W0)" wording listed four
  runs while saying three; the set IS four runs (3 war_eval seeds + W0), consistent
  with §11's n=4 control-band language. Recomputed from disk with the 4-run set:
  per-game 27B MAX = ft09 2 / sb26 1 / lp85 1 / vc33 2, **Σ 27B MAX = 6** —
  unchanged, so the capability bar stays ≥ 8.
  S72 = **all certified 72B seeds** (budgeted n=2 from the start, §4 remedy; plus
  the marginal seed if triggered).
- **One statistic, every prong (fixes prog-synthesis N4's underspecification):**
  both the CAPABILITY prong and the THROUGHPUT-ADJ prong use the SAME statistic —
  per-game MAX over S72, then Σ over the 4 screen games. Not first-seed-only; the
  marginal seed, if run, enters the MAX on both prongs.
- **Dead-code check (the defect class, tested):** branch 2 no longer contains any
  capability conjunct, so at ρ_action ≥ 2.5 (parity fails) branch 2 is the live
  branch and null_adj binds; at ρ_action ≤ 1.11 branch 1 is live. A 72B at Σ=5–7
  beating null_adj+1 is now GO — the exact case §15 exhibited as wrongly NO-GO.
- **MARGIN stays +1** (not the +2 throttled-capability variant systems offered as
  option (ii)): the §4 sensitivity table shows +2 would zero the gate's power
  against a real single-game gain at k≤2 seeds, aggravating exactly the
  false-NO-GO defect methodology R1 blocks on. +1 retains the ρ-step protection
  rationale from v1.
- **Marginal-seed rule, re-keyed (part of the N13 remedy):** run one additional 72B
  seed if CAPABILITY lands at Σ ∈ {6, 7} **or** THROUGHPUT-ADJ lands within 1 level
  of its bar (at bar−1 or bar). Extended from v1's "exactly Σ=7" per llm-agents'
  remedy menu. One extra seed max; still short → NO-GO.
- **NO-GO consequence** unchanged (§8.1 closes war-v4; §8.2 governs re-screens) —
  but NO-GO may not be declared before the §6 FP8 fallback branch resolves.

## 2. Comparator data + per-seed Σ table (R17 checklist item 5; llm-agents Q4)

Computed from the four `benchmark.json` files (artifact: `per_seed_table.json`).
Versioned game ids, identical in all 4 runs: ft09-0d8bbf25, sb26-7fbdac44,
lp85-305b61c3, vc33-5430563c.

**Full-budget lc and per-seed Σ (the table llm-agents Q4 demanded):**

| run | ft09 | sb26 | lp85 | vc33 | **Σ (4-game lc)** | shortfall vs Σ MAX = 6 |
|---|---:|---:|---:|---:|---:|---:|
| war_eval_v1 | 1 | 1 | 1 | 2 | **5** | 1 |
| war_eval_v2 | 2 | 1 | 1 | 1 | **5** | 1 |
| war_eval_v3 | 0 | 1 | 1 | 2 | **4** | 2 |
| w0_eval_s1 (W0) | 2 | 1 | 1 | 2 | **6** | 0 |
| **per-game MAX** | **2** | **1** | **1** | **2** | **Σ MAX = 6** | |

The implied single-27B-seed shortfall distribution vs the frozen MAX Σ=6 is
{0, 1, 1, 2}: a single 27B seed re-drawn against its own 4-run MAX falls short by
1–2 levels 75% of the time. This is the seed-luck asymmetry N13 flagged, now
quantified and remedied (§4).

**Per-seed THROTTLED lc rows** (each run's own actions_per_level walked to
⌊N_r(g)/ρ⌋; the empirical null for branch 2 — see §4):

| run | ρ=2.5: ft09/sb26/lp85/vc33 (Σ) | ρ=3.0: ft09/sb26/lp85/vc33 (Σ) |
|---|---|---|
| war_eval_v1 | 0/1/1/1 (**3**) | 0/1/1/1 (**3**) |
| war_eval_v2 | 0/1/1/1 (**3**) | 0/1/0/1 (**2**) |
| war_eval_v3 | 0/1/1/1 (**3**) | 0/1/1/1 (**3**) |
| w0_eval_s1 | 0/1/1/2 (**4**) | 0/1/1/1 (**3**) |

W0 row = the sealed null_adj: **Σ null_adj = 4 (ρ≤2.5) / 3 (2.5<ρ≤3.0)** —
re-verified from disk, identical to v1's frozen table. Full ρ grid 1.5–3.5 in the
artifact; the walk procedure (not the anchor numbers) is what seals, unchanged.

## 3. ρ := ρ_action — the corrected measurement (fixes systems #16)

`generated tokens/sec` is decode-only; each action in this harness prefills a fresh
4×-upscaled grid image plus context, and the 72B/27B prefill ratio need not equal
the decode ratio (and the 2.4–3.1 decode prior is itself suspect given the 27B is
served FP8, not BF16). **Sealed definition:**

- **ρ := ρ_action = (actions/s, 27B) / (actions/s, 72B)**, measured on the same
  4 games, same SKU, same fixed 7920 s per-game window. Because the windows are
  identical, this reduces to an action-count ratio:
  **ρ_action = Σ_g N27B_W0(g) / Σ_g N72B_canary(g)**, pooled over the 4 screen
  games (pooled, not per-game, to keep canary game-luck out of the null).
- **27B numerator, frozen now from disk:** N27B_W0 = 39 (ft09) + 225 (sb26) +
  147 (lp85) + 69 (vc33) = **480 actions / 7920 s = 0.0606 actions/s**.
- **72B denominator:** measured from the Jul-24 canary push, which runs the 4
  games at the FULL 7920 s window (same cost as a short canary, ~2.5 GPU-h, and it
  is what makes ρ_action measurable). The `summary.txt` tokens/s line is still
  recorded — as a diagnostic only, no longer the gate input.
- **Sequencing preserves seal-before-measure:** ρ_action is measured on the canary;
  null_adj is then frozen at measured ρ_action by the unchanged §3.5 cumulative
  walk BEFORE the scored bench (seed 1) is pushed. If measured ρ_action lands
  outside [2.4, 3.1] nothing re-opens: the frozen procedure simply evaluates at the
  measured value (per-seed throttled table across ρ ∈ [1.5, 3.5] already published
  in the artifact).
- **Recorded bias (stated, accepted):** N27B was measured at 25-game concurrency;
  the canary and scored 72B benches run 4-game concurrency (more GPU share per
  game). Canary and scored bench share the same concurrency, so the null's N72B
  prediction is internally consistent for the scored run; the residual asymmetry
  (72B enjoys 4-game concurrency while the 27B comparator lc came from 25-game
  runs) is pro-72B on the capability prong. Accepted for a capability-existence
  screen: it biases toward GO, so a NO-GO is conservative; a GO must anyway
  republish the 25-game × 3-seed × full-budget ledger at measured ρ_action in the
  v4 registration — which, answering rl-planning Q5 directly: **yes, the v4
  registration on GO is REQUIRED to contain that ledger before any v4 window
  opens.** As a published diagnostic (not a gate input), the scored bench also
  reports realized ρ_action from its own action counts next to the canary value.

## 4. False-NO-GO probability — quantified, and the chosen remedy (fixes methodology R1 / llm-agents N13)

Computed by exact enumeration over the 4-run empirical support (4 and 16 equally
likely draw-combinations at k=1,2 — exact, strictly stronger than MC; a 100k-draw
MC bootstrap cross-check agrees to 3+ decimals). Two regimes, per the reviewers'
specified models: (i) null 72B ≡ 27B, (ii) true +1-level-per-game shift; k = 1 and
2 72B seeds; row-wise draws (preserve within-seed correlation) and
independent-per-game draws (conservative). Artifact: `false_nogo_bootstrap.json`.

**(a) Repaired gate, throttled regime (the expected world, ρ ∈ [2.4, 3.1]) —
branch 2 live, bar = Σ null_adj + 1 (= 5 at ρ=2.5, 4 at ρ=3.0):**

| quantity | k=1 | k=2 |
|---|---:|---:|
| P(false GO), null 72B≡27B, either ρ anchor, either draw scheme | **0.000** | **0.000** |
| P(false NO-GO), +1/game shift, either ρ anchor, either scheme | **0.000** | **0.000** |

Under the null the throttled pseudo-72B rows sum to 2–4 < bar always; under the
+1/game alternative every row clears the bar with certainty. The repaired branch 2
separates these two hypotheses perfectly on the empirical support — this is the
direct consequence of un-killing the branch.

**(b) Capability prong alone, parity regime (methodology R1's literal model — 72B
lc drawn from the 27B full-budget distributions), bar Σ ≥ 8:**

| quantity | k=1 | k=2 |
|---|---:|---:|
| P(GO), null 72B≡27B (any scheme) | 0.000 | 0.000 |
| P(false NO-GO), +1/game, row-wise | 0.000 | 0.000 |
| P(false NO-GO), +1/game, independent-per-game (conservative) | **0.0625** | **0.0039** |

So the headline number the panel asked for: **under the sealed gate, P(false
NO-GO | true +1/game shift) = 0.000 on the empirical support in the expected
throttled regime, and ≤ 0.0625 (1 seed) / ≤ 0.0039 (2 seeds) under the most
conservative independent-draw capability-prong model.** The capability prong also
has zero false-GO under the null (max attainable null Σ = 6 < 8).

**Sensitivity, not an error rate (published so the panel ratifies a range):**
against a *weaker* true improvement — +1 level on a single game only — branch 2
detects it with P(GO) = 0.25 (k=1) → 0.44 (k=2) at ρ=2.5, and 0.75 → 0.94 at
ρ=3.0. This is the computation that fixes MARGIN at +1: at MARGIN=+2 these
detection rates drop to ~0 at ρ=2.5, i.e. the +2 variant would re-manufacture the
false-NO-GO defect.

**Caveat, stated plainly:** the support is n=4 rows; "0.000" means zero on the
empirical distribution the reviewers specified, not a structural impossibility.
Hence the remedy below is adopted anyway.

**Chosen remedy (llm-agents N13 — "I will not sign Q6 until one is chosen"): we
choose TWO of the offered options, jointly:** (1) **budget 2 72B seeds from the
start** (ledger headroom exists — §5; k=2 cuts the conservative-model false-NO-GO
to 0.0039), and (2) **extend the marginal-seed trigger to capability Σ ∈ {6, 7}
and to branch-2-within-1** (§1). The "seal the coin-flip as accepted risk" option
is NOT taken — no coin-flip remains to accept at these error rates.

## 5. Cost figures — reconciled to one budget (fixes systems #17 / llm-agents N14 / rl-planning MINOR)

**What 7920 s bounds — resolved from the logs, not asserted:** the eval config line
in `runs/kernel_pulls/w0_eval_s1/arc3-duck-w0-continuation-eval.log` reads
`max_runtime_s_per_game=7920.0, concurrency=28` — 7920 s is a **per-game soft
deadline, with all games running concurrently**. Evidence: in every certified run
all 25 games' `started_at` fall within ~1 s of each other and every
`final_wallclock_seconds` ≈ 7920–7969; kernel start→end = 2h12–13m (e.g. war_eval_v1
12:49:36→15:01:56). So a 25-game bench and a 4-game bench both cost ONE ~2.2 h
window — "4 × 7920 s = 8.8 GPU-h" was a mis-reading of overlapped windows, and is
retracted. The `max_runtime_minutes: 45` figure is the TAAF bundle default in
`duck_eval/taaf_bundle/src/ARC3-Inference/configs/inference.json`
(`environment.max_runtime_minutes`), which the eval path overrides with the
per-game 7920 s value above; it does not bound anything in this screen and is
struck from the risk-A text.

**One push = 2.2 h window + 72B load/init (~0.2–0.3 h) ≈ 2.5 GPU-h. The single
reconciled budget:**

| item | pushes | GPU-h | week attribution |
|---|---:|---:|---|
| canary (full-window 4-game, measures ρ_action) | 1 | 2.5 | Jul 21–27 |
| scored bench seed 1 | 1 | 2.5 | Jul 21–27 |
| scored bench seed 2 (budgeted from start, §4 remedy) | 1 | 2.5 | Jul 21–27 |
| marginal seed (only if §1 trigger fires) | 0–1 | 0–2.5 | Jul 21–27 (28th at latest) |
| **A17 AWQ arm — booked cap** | **3–4** | **≤ 10.0** (modal 7.5) | |
| FP8 fallback arm (contingent, §6: canary + scored) | 0–2 | 0–5.0 | Jul 28–Aug 3 |

Reconciliation of the four contradictory figures: **§12's "~10" = the AWQ-arm
booked cap (correct, unchanged in the ledger); v1 §6.1's "~7.5" = the modal
3-push path (superseded as a budget line — the budget is the cap); "8.8 GPU-h
scored bench" = retracted (overlapped windows); "45 min" = non-binding bundle
default (struck).** Week fit: the verified §12 ledger already books A17 at ~10 in
Jul 21–27 for a week total ≈ 23.2/30 — this amendment changes no ledger number, so
**the Jul 21–27 week still closes at ≈ 23.2/30**. The contingent FP8 arm (≤ 5.0)
lands in Jul 28–Aug 3 (≈ 24.2/30 on the max path): it fits iff at most one of
{W3 banking, EWM Stage-1, W0 fallback seeds} triggers alongside it (24.2 + 5.0 =
29.2 ≤ 30 even on the full max path, but with only 0.8 slack); the war-room
scheduler owns the ordering, and the FP8 arm yields to the A14 binding look but
takes priority over EWM Stage-1 (which is conditional and, per llm-agents N12,
likely re-prioritized below A17 anyway). Push-count: +2 worst case over the week,
within the ≤14/wk bound systems verified.

## 6. FP8/W8A8 fallback SKU — pre-registered (research_2026-07-21)

2026 quantization consensus (surveyed 07-21): on 70B-class open weights, FP8 lands
within ~0.4 pt of FP16 on reasoning/code benchmarks while AWQ-INT4 degrades within
~1.6 pt — and the INT4 loss concentrates exactly on reasoning, with VL/perception
degradation un-measured on top. Therefore, **pre-registered now, before any 72B
number is observed:**

- **Trigger:** the AWQ screen lands NO-GO **with the capability prong failed**
  (Σ 72B MAX < 8 AND branch 2 failed). If branch 2 passed, the gate is GO and no
  fallback is needed; if the failure is unambiguous on both prongs by wide margin
  (Σ 72B MAX ≤ 4 and branch 2 short by ≥ 2), the fallback may be waived by panel
  as futile — otherwise it runs before NO-GO is declared.
- **Fallback SKU:** Qwen2.5-VL-72B at **FP8/W8A8** on the same 96 GB RTX PRO 6000
  rail, same harness, same serve-config discipline, same gate re-evaluated with a
  fresh ρ_action canary (FP8's ~1.4× decode lift vs AWQ's ~3.1× means ρ_action
  will differ; the frozen walk handles any measured ρ).
- **Status vs §8.2:** a bounded, pre-registered SKU swap of the SAME model at a
  different quantization is **NOT a "materially different artifact"** under §8.2
  (retained unchanged); it is inside this screen's registration. Anything beyond
  this one swap (different base model, different size tier) remains governed by
  §8.2 as sealed.
- **Pre-registered TBDs, stated as such (cannot be computed from data on disk):**
  (i) **artifact availability** — no verified Kaggle-attachable FP8/W8A8 snapshot
  of Qwen2.5-VL-72B is confirmed today; procedure: search Kaggle Models/Datasets
  before the trigger can fire; if none exists, an upload/snapshot dataset push is
  required and must be costed then (upload is $0 GPU-h but consumes a dataset
  push). (ii) **single-GPU serve feasibility** — ~72 GB FP8 weights + VL vision
  tower + KV on 96 GB is tight; procedure: canary must verify serve at
  `max-model-len` 32768, stepping down to 16384 and/or FP8 KV cache if needed; if
  the model cannot serve single-GPU, the fallback is void and NO-GO stands on the
  AWQ evidence **with the quantization caveat recorded in the NO-GO filing**
  (i.e., the panel is told the screen bounded AWQ-72B, not 72B-tier capability).
- **Budget:** ≤ 2 pushes / ≤ 5.0 GPU-h, contingent, booked in §5.

## 7. Preconditions — sealed, in order, before any 72B push

1. **Accelerator-selector check (ops, discussions_2026-07-22 #697720, S. Brodehl):**
   Brodehl reports RTX-6000-Pro instances not currently selectable on his account
   (host silent). Before EVERY GPU build in this screen: verify the kernel
   accelerator dropdown still offers RTX 6000 Pro (`NvidiaRtxPro6000`) — a
   1-minute UI check — and keep the existing preflight GPU assert. If the SKU is
   not selectable, the screen STALLS and goes to the war-room daily loop; no
   substitute accelerator may be used (the §0 recompute-and-discard rule stands:
   any run printing a different GPU name is void).
2. **Versioned-game-id identity (drift guard, mirrors methodology R2):** the 72B
   bench's 4 screen games must resolve to exactly ft09-0d8bbf25, sb26-7fbdac44,
   lp85-305b61c3, vc33-5430563c (the ids of all four certified runs, verified
   identical). A drifted game is DROPPED from both sides of the gate (not counted
   either way) and the screen result is flagged to the panel; ≥ 2 drops void the
   screen.
3. **Serve-config smoke (v1 risk D, unchanged):** tool-call round-trip asserted on
   the canary before the scored bench (hermes parser, no qwen3 reasoning parser,
   thinking off); MM image path confirmed (non-zero MM cache, risk E).
4. **ρ_action measured on the canary and null_adj frozen at it (§3) BEFORE the
   scored seed-1 push.** The gate is evaluated only after seed 2 (and the marginal
   seed if triggered) lands.

## 8. Changelog — each R16 defect → fix

| # | R16 defect (reviewer) | fix in this amendment |
|---|---|---|
| 1 | Dead-code null branch; two contradictory booleans in one circulation (systems #15, prog-synthesis N1) | §1: single boolean = Part 1 §13's form; ≥8 conjunct DROPPED from branch 2 (systems option (i)); A17′ §4 text declared superseded; dead-code check shown explicitly |
| 2 | Unquantified false-NO-GO on a campaign-terminal gate (methodology R1, llm-agents N13) | §4: exact enumeration + bootstrap on disk data, both regimes, k=1,2, both draw schemes; headline P(false NO-GO)=0.000 throttled / ≤0.0625→0.0039 conservative; remedy CHOSEN: 2 seeds budgeted + marginal trigger extended to Σ∈{6,7} and branch-2-within-1 |
| 3 | ρ mismeasured — decode-only tokens/s ignores per-action prefill of a fresh 4× image (systems #16) | §3: ρ := ρ_action = actions/s ratio, pooled over the 4 games at identical 7920 s windows; 27B numerator frozen (480 actions); tokens/s demoted to diagnostic; frozen walk kept; concurrency bias recorded |
| 4 | Comparator set self-inconsistent — "3 certified seeds" lists 4 runs (llm-agents N14, prog-synthesis N4, rl-planning MINOR) | §1/§2: set = 4 certified runs, stated; Σ MAX recomputed from disk = 6 (unchanged); n=4 language reconciled with §11 |
| 5 | 72B-side statistic unspecified (per-game MAX vs first-seed) (prog-synthesis N4) | §1: per-game MAX over all certified 72B seeds, SAME statistic in both prongs, one sentence |
| 6 | Cost figures irreconcilable: ~10 vs ~7.5 vs 8.8 vs 45-min (systems #17, llm-agents N14, rl-planning MINOR) | §5: 7920 s = per-game soft deadline, games concurrent (log-verified `max_runtime_s_per_game=7920.0, concurrency=28`); one budget: AWQ arm cap 10.0 GPU-h (modal 7.5), FP8 arm ≤5.0 contingent; §12 ledger unchanged, Jul 21–27 closes ≈23.2/30; 45-min = non-binding bundle default, struck |
| 7 | Per-seed Σ table demanded (llm-agents Q4, R17 checklist 5) | §2: full table (full-budget and throttled rows), shortfall distribution {0,1,1,2}; artifact `per_seed_table.json` |
| 8 | v4-registration ledger requirement (rl-planning Q5, directives Q6 ruling) | §3: YES — GO obligates the 25-game × 3-seed × full-budget GPU-h ledger at measured ρ_action in the v4 registration before any v4 window opens |
| 9 | FP8 fallback pre-registration (research_2026-07-21) | §6: trigger, SKU, §8.2 compatibility, budget, and two pre-registered TBDs (artifact availability; single-GPU fit) with procedures |
| 10 | Accelerator-selector ops risk (discussions_2026-07-22) | §7.1: explicit precondition before every GPU build |

Not changed, deliberately: §8.2 "materially different" (no reviewer objected);
weights choice and serve-config repairs (v1 §1–§2, ratified structurally by
systems #7/#11); the null_adj walk procedure (verified by prog-synthesis O4:
"worked examples verify"); MARGIN=+1 (now justified quantitatively, §4).

Open items (honest list): FP8 TBDs (i)–(ii) in §6; the FP8 arm's week-fit has only
0.8 GPU-h slack on the full max path (§5, scheduler-owned); the 4-vs-25-game
concurrency bias is recorded, not removed (§3) — but as of the R17 amendment §9 it is
no longer in the binding boolean.

## 9. R17 amendment (2026-07-23) — verbatim gate seal + null_adj at realized 72B actions

Filed 2026-07-23 in response to R17 OBJ-D1 (prog-synthesis N1 residual / methodology Q1 /
rl-planning Q6) and OBJ-D2 (systems #20), per Part 4 D2 of
`learnings/panel/round17/_directives.md`. Append-only; §1 and §3 are unchanged. This section
is the sealing text the four A17 sign-offs were withheld pending.

### 9.1 The single gate boolean, sealed VERBATIM (fixes OBJ-D1 / systems, all A17 sign-offs)

A gate that last round sealed with two published forms cannot this round seal by hash-pointer
(prog-synthesis). The one and only gate boolean, quoted verbatim from §1 (there is no other form
in the record; the A17′ §4 text is SUPERSEDED and void):

> GO  iff
>     ( CAPABILITY:      Σ_g max_{s∈S72} lc72(g,s)  ≥  Σ_g max_{r∈R27} lc27(g,r) + 2    # = ≥ 8
>       AND
>       ACTION-PARITY:   Σ_g N72B(g)  ≥  0.90 · Σ_g N27B(g) )
>   OR
>     ( THROUGHPUT-ADJ:  Σ_g max_{s∈S72} lc72(g,s)  ≥  Σ_g null_adj(g; ρ_action) + 1 )  # MARGIN = +1
> NO-GO otherwise.

R27 = the 4 certified 27B runs {war_eval_v1, war_eval_v2, war_eval_v3, w0_eval_s1}, Σ 27B MAX = 6
(so the capability bar = ≥ 8). S72 = all certified 72B seeds (n=2 budgeted + marginal if triggered).
Both prongs use the SAME statistic: per-game MAX over S72, then Σ over the 4 screen games.

### 9.2 null_adj evaluated at REALIZED 72B per-game actions (fixes OBJ-D2 / systems #20)

Systems #20 (`learnings/panel/round17/systems.md` line 31) is correct: ρ_action as frozen
(27B numerator = 480 actions measured at concurrency=28 across 25 games; 72B screen serves ~4
games) is concurrency-confounded — the pooled actions/s ratio across the two batching regimes
measures serving-stack utilization, not model speed. Direction is conservative (biases toward
NO-GO) but mismeasured, and it is the last free parameter in the binding boolean.

**Sealed fix (verbatim, systems #20): "null_adj is evaluated at the realized 72B per-game N
from the pull; ρ_action is demoted to a pre-run planning diagnostic only."** At gate-evaluation
time the 72B pull already contains its own per-game action counts, so THROUGHPUT-ADJ's
Σ_g null_adj(g; ·) is computed from those realized counts — no ρ-predicted N₇₂B enters the
boolean. ρ_action (§3) remains published as a pre-run planning/sequencing diagnostic and for the
v4 registration ledger, but is NOT a gate input. Zero GPU-h; removes the concurrency confound
from the binding boolean entirely. This lands BEFORE the pre-Aug-1 screen runs.

### 9.3 Kamradt A17-boundary note (R-ADAPTS wording)

Per-game score must never re-enter agent context during the screen (the score is an offline
scoring-oracle read, `duck_eval/scoring_oracle.py`, not an in-context signal). Recorded here per
the R-ADAPTS ruling; no threshold change.
