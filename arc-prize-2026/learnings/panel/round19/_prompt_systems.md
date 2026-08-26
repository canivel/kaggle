You are Professor of ML Systems (GPU inference, vLLM serving, wall-clock budgets, quota economics; kills plans that don't fit the compute envelope).

You are reviewer #3 on a 5-person adversarial review panel evaluating a competition
strategy proposal for ARC-AGI-3 (Kaggle, deadline Nov 2 2026). The proposing team has a
best score of 0.43; the leader is at 1.56; the winning Milestone-1 notebook is public.

PANEL RULES — READ CAREFULLY:
- You are UNBIASED and RIGOROUS. You have no stake in the proposal passing.
- Do NOT be agreeable. A proposal that passes round 1 unchanged indicates REVIEW FAILURE.
- Attack the weakest LOAD-BEARING assumptions, not typos.
- Every objection must be specific and actionable (not "needs more detail").
- Distinguish severity: FATAL (invalidates the plan), MAJOR (must fix before execution),
  MINOR (should fix).
- You review ONLY within your expertise; note explicitly what you cannot judge.
- If evidence is missing for a claim, say exactly what experiment/data would supply it.

YOUR PRIOR-ROUND OBJECTIONS (verify each is resolved in the revision; state
RESOLVED / PARTIALLY-RESOLVED / UNRESOLVED for every one before new comments):
=====================================================================
## Objections

**[Prior FATAL 1 — quota is not free] PARTIALLY-RESOLVED.** The demanded budget table exists, is itemized, and fits 17.5h into 60h with headroom — the structural fix I asked for. But every line is priced against a hardware claim (96GB single-card Blackwell) that is asserted, not cited: no kernel-log path, no `nvidia-smi` artifact. Additionally, the table's load-bearing footnote — "A21 exploration draws use the daily submission window, not GPU quota" — is asserted without evidence; on Kaggle, notebook *commit* runs typically do draw personal quota even when hidden-test reruns do not. If scored 8h windows draw quota, nightly filler alone is 56 GPU-h/week and the whole §2 table is fiction. Fix: attach the kernel-log artifact showing the SKU, and cite the competition rule or an observed quota ledger showing scored-window runs are quota-exempt.

**[Prior FATAL 2 — A17 envelope] PARTIALLY-RESOLVED.** Memory (43GB AWQ on 96GB, KV headroom at 32k) and SKU parity are now stated with numbers, and the self-certifying envelope NO-GO is a genuinely good mechanism — physics shouldn't need ratification. Two gaps remain. First, the 2.5–3× penalty is *scoped, not measured*, and it is framed as a **decode** penalty — but an agentic VL loop is prefill-heavy (image tokens re-ingested per turn, growing context), and prefill cost scales with total FLOPs (~2.7× params vs the 27B), so end-to-end turn latency can exceed 3.5× even when decode is under it. Second, no minimum-viable turns/window floor is derived: "~⅓ the turns" is only meaningful against a stated turns-per-level requirement from existing 27B telemetry. Fix: define the 3.5× envelope threshold on measured end-to-end turn latency (parity prong), and publish the turns/level floor the capability prong must beat.

**[Prior MAJOR — watchdog kills the bench] RESOLVED.** A24-revised adopts my fix verbatim: harness heartbeat ≥20 min, kill only on 60 min heartbeat silence, registered benches exempt from the 6h cap. One implementation note, not an objection: ensure the harness emits heartbeats during model download/load/graph-capture, before the server is up, or the 72B cold start will trip the 60-min kill.

**[Prior MAJOR — headline numbers contradict the briefing] UNRESOLVED.** The internal verification (`runs/verify_2026-07-21/`) completed and corrected four discrepancies — good — but the proposal *still* claims best 1.33 / leader 1.86 while the panel briefing states 0.43 / 1.56, and no line in v2 reconciles them. Worse, the numbers are now mutually impossible: a max-keeping leaderboard with a 15-draw ledger mean of 0.962 cannot coexist with a best of 0.43. Either the briefing and proposal use different metrics/normalizations (say which, with the formula) or one source is stale (say which, with the artifact). Every economic quantity in §1 — the +2.6σ exceedance, the 0.001–0.002 opportunity cost, A21's ΔE[max] — is computed relative to "current best 1.33"; if the true best is 0.43 the sign and magnitude of the exploration trade change entirely. Approval must be conditioned on a one-paragraph reconciliation.

**[Prior MAJOR — RC4/R5 pricing contradiction] RESOLVED.** A21 is exactly the fix I demanded: a bounded (12-window) pre-registered first-draw allowance, quantified regret (−0.01 to −0.02 E[max]), harm-pause, mean-lift rule retired for exploration but retained for promotion. The tail-sensitivity of the regret number is a methodology-panel matter (see below); the *structure* resolves the contradiction.

**[Prior MINOR — no tail model] PARTIALLY-RESOLVED.** Parameters (mean 0.962, σ̂ 0.144, n=15, ~102 draws) are now stated, which permits reproduction. But the family is implicitly Gaussian, the exceedance computation lives at +2.6σ where n=15 gives essentially no tail constraint (σ̂ alone carries ~19% relative error), and no CI is given on P(touch 1.44) ≈ 0.18. Mitigating: the "two orders of magnitude below the old rule" conclusion survives a 5× tail error, so A21 is robust; E[max] ≈ 1.35 as a *strategic* claim ("filler is not top-10") is not. State the fit and a CI, as originally asked.

**[MAJOR — new] The entire §2/§3 case is single-point-dependent on one uncited hardware claim.** "RTX PRO 6000 Blackwell 96GB" is not a historically known Kaggle SKU (T4×2/P100/L4×4 are), and v2 uses it to overturn my round-1 hardware premise while citing only "kernel logs" with no path. If the real rail is 4×L4 (96GB *aggregate*, 24GB/card), the 43GB AWQ model requires TP4 sharding, decode throughput drops several-fold below the scoped penalty, and both §2's 7.5h screen estimate and §3's memory headroom paragraph are wrong. This is cheap to fix and must be fixed before any quota is spent: pre-register the screen with the `nvidia-smi`/`torch.cuda.get_device_properties` output from a scored-rail and a bench-rail kernel log as attached artifacts.

**[MINOR — new] The contingency line cannot re-run the largest deliverable.** Contingency is 5.0h; the A17 screen is 7.5h; with infra incidents on 8 of the last 11 days, a wedged screen is the modal failure, and the table as written cannot absorb it (the 60h ceiling can, but then the table's own contingency line is decorative). Also, the 7.5h screen estimate does not itemize setup overhead (43GB weight download, load, warmup), which on cold Kaggle sessions plausibly costs 45–90 min. Re-line the table: contingency ≥ 1× the largest run, setup itemized.


=====================================================================

THE PROPOSAL (sha256 of the full document: 8c1df549836e36b1; full length 9225 chars;
if the text below does not end with the literal line "## END OF PROPOSAL ##",
your copy is TRUNCATED — file that as an objection and review only what you see):
=====================================================================
# Daily brief — 2026-07-24

## §1a Result deep-dive

### Exploration draw 1/12: sentinel arm = 0.71 — campaign low; HARM-PAUSE executed

**Draw:** `canivel/arc3-duck-sentinel` v1 (vanilla duck + (f) continuation
hygiene-default + budget sentinel v2 @ SENTINEL_BUDGET=150; NO warpack, NO
ledger) scored **0.71** at the 00:07Z fire — the lowest scored draw of the
campaign (previous low: war draw #5 = 0.76; frozen-fork low 0.82).

**Pre-registered expectation vs outcome:** entry was via 2-seed canary PASS +
non-harm screen (pooled Δlc ≈ −0.05/game, direction flipped across seeds).
The sealed A21/C2 rule — draw < 0.80 pauses the arm — **triggered; the pause
was executed pre-loop** by the early interactive session. This is the rule
operating exactly as designed: no inference is claimed from n=1, but exposure
stops immediately.

**Statistics (validated this morning):**
- vs frozen control n=10 (mean 0.975, σ̂ 0.156): **z = −1.70**, one-sided
  p ≈ 0.044.
- vs pooled control n=15 (mean 0.962, σ̂ 0.144): z = −1.75, p ≈ 0.041.
- A single draw at p ≈ 0.04 is suggestive, not conclusive — but it is the
  THIRD independent negative signal for this composition: eval-rail
  Δlog1p(RHAE) was negative on BOTH certified seeds (s1 −0.315 p=0.997; s2
  −0.166 p=0.90). Three aligned negatives across two rails = the prior for
  the sentinel composition is now poor.

**Mechanism evidence:** none pullable from the scored rerun (logs hidden);
the composition's mechanism story is already sealed from the eval rail (W1:
mechanism PASS / behavior "fires, doesn't pay" — warnings changed nothing,
21/22 fired games kept grinding). The 0.71 is consistent with the sealed
story: the sentinel adds FACT lines that at best do nothing and at worst
perturb the policy slightly negative, and under the completion-weighted
scorer an efficiency observable has no depth channel to pay through.

**Cost accounting:** one exploration window of 12 (A21); priced ~0.001–0.002
E[max]-equiv in max currency. 11 windows remain. LB best UNCHANGED at 1.33.

**Disposition question for panel (Q1 below):** the calibrated W2 rule
(sentinel_w2_preregistration.md amendment: z-rule vs legal control w0_s1 =
1.731; two-seed KILL α ≈ 0.02) is denominated on the EVAL rail — the scored
0.71 does not formally enter it. Is the $0 W2 eval build still worth a push
slot to close the line cleanly, or does the harm-pause + sealed W1 evidence
suffice to shelve the sentinel as "certified telemetry, no lift channel"
without W2?

### Ledger

Frozen control unchanged (0.71 was a different composition — it does NOT
enter): n=10 mean 0.975 σ̂ 0.156, band 0.82–1.33. Pooled n=15 mean 0.962 σ̂
0.144.

### LB context

Field compression continues: **20 teams ≥ 1.45** this morning (the old 1.44
wall is fully submerged; yesterday gold ≈ 1.47). Top: KOJIMA 1.86, Tecnod8.AI
1.61, then a dense 1.45–1.60 band. We hold 1.33 (~#45) and continue to erode
without a depth-channel gain. This is the strategic backdrop for the A17
screen and the depth-budget lane: efficiency channels price ≈ 0; a single
frontier depth event prices +0.19 to +0.29 rail (d4_provisional_reprice).

## §1b Discussions sweep (learnings/war_room/discussions_2026-07-24.md)

- **ADAPT (ops-critical, TODAY) — #684625 Scott Le Grand comment:** vLLM
  silently hangs on RTX Pro 6000 with the duck notebook after 15–20 min at
  ≥8–25 concurrent sessions (reproduced off-Kaggle). Directly relevant to
  today's A17 canary (same GPU + vLLM stack). Mitigation posture: the canary
  runs 4-game concurrency (below the reported ≥8 threshold), and our
  certified 27B 25-game runs completed clean at 2h12m — but the canary gains
  a log-heartbeat observable so a silent hang is diagnosable post-run; A24
  watchdog wording already covers the class. Watch thread for root cause.
- **ADOPT (upgraded) — boristown `duck-harness-fast-eval` public 1.47 gold,
  144 upvotes.** The top public artifact is 1.47, not 1.39. Defensive
  fork-diff target for our frozen fork upgrades accordingly — potential
  low-risk +0.14 (fork-never-build discipline; byte-matched metadata
  mandatory; fork-diff analysis is $0 and precedes any decision). New Q5.
- **WATCH:** unscored "Deep Reasoning Agent (179/183 levels)" code-tab entry
  updated 1h ago — hype-smelling; becomes ADOPT-diff only if it scores
  >1.47. Re-check tomorrow.
- **IGNORE:** #728210 (beginner), trailing drive-by comments on 07-23
  threads, #697720 grumbles (dropdown-check ops-ADAPT already carried).
- **LB (live top-49):** gold cutoff ≈ **1.49** (top 13); wall band 1.44–1.60
  dense; **we slid #45 → #49 overnight at 1.33** (~4 ranks/day bleed), now
  tied with Jack Cole. Two low-entry high scores (1.60@6, 1.49@5) exceed all
  public artifacts → real methods, not clones; the public 1.47 is itself
  seeding the 1.44–1.47 band. No arc_agi version-bump news.

## §1c Research sweep (learnings/war_room/research_2026-07-24.md)

- **Zero ADOPT, three ADAPT (all wording/design, no build), two PARK.**
- **ADAPT (low) — arXiv 2607.20972** cue-anchored working memory (Jul 23):
  agents made 0 voluntary memory ops in 114 turns; harness-injected
  deterministic cues worked with zero false alarms. Application: the
  depth-budget lane harness re-injects EWM contract + certificate state on
  deterministic cues (level change / revise-loop entry), never
  agent-volitional.
- **ADAPT (low-med) — arXiv 2607.07196** world-model admissibility ladder:
  action-following robustness ranks ABOVE visual fidelity; one line into EWM
  v1.1 certification wording (admissibility-before-verdict).
- **PARK — 2607.08964** Long-Horizon-Terminal-Bench (dense partial credit,
  best model 15.2%): external support for depth ≫ efficiency. **PARK —
  2601.22129** SWE-Replay banked-prefix replay (+3.8% at −17.4% cost):
  cleanest external statement of banking, but blocked by our latent-state
  audit (branching presumes faithful state restoration; state-aliasing
  breaks it).
- **Schema replication watch: still ZERO** (HF traces frozen at 50
  trajectories; no reproduction anywhere). OPINE-World follow-ups: none.
- **A17 serving:** no direct 72B-VL-AWQ-on-96GB datapoint anywhere;
  vllm-blackwell-guide (lastloop-ai) confirms quantized vLLM serves clean on
  RTX PRO 6000 sm_120 at 27B/35B — **toolchain half de-risked; the
  weights+vision-tower+KV fit half is settled empirically by today's
  canary** (as A23 anticipated: envelope is self-certifying).

## §2 Today's plan

1. **A17 canary push (slot 1, ratified; all sign-offs sealed via scope-v2 §9).**
   4-game full-window Qwen2.5-VL-72B-AWQ canary: measures ρ_action
   denominator (27B numerator frozen 480 actions/7920s), serve-config smoke
   (hermes parser, no thinking flags, MM path), envelope check. Build agent
   running (starts from proven eval builder; loud-fail policy if 72B fails to
   serve — a silent 27B fallback would poison ρ_action). §7 preconditions
   walked in order before push; interpretation discipline per C3
   (a17_error_model): the screen is a ≥+4-to-+5-level capability-existence
   detector; NO-GO at modest lift is the designed outcome.
2. **Panel R19 (routine, reduced bench)** on this brief. Then implement
   directives.
3. **Slot 2 contested — panel rules:** W2 sentinel confirmatory null vs
   tr87 fresh-stream depth-lane confirmation (held from the OBJ-H
   ratification: "rides a later quota slot after the panel sees this
   ratification" — the panel is seeing it today).
4. Queue: frozen-fork filler head (verified pending); daemon 18:37/20:07.

## Open questions (for R19)

1. **Sentinel disposition after harm-pause:** shelve on sealed W1 evidence +
   harm-pause (no W2, save the push), or spend a slot on the $0 W2
   confirmatory null to close the calibrated two-seed KILL cleanly? Note the
   arm cannot re-enter LB regardless without a new A21 entry case.
2. **Slot-2 allocation today:** tr87 fresh-stream confirmation (depth lane,
   OBJ-H ratified GO, kill-switch = held-out null10) vs W2 vs hold. The
   depth lane is the only non-A17 depth-targeting line; concentration risk
   says fund it.
3. **Exploration budget pacing:** 1/12 spent on a paused arm. Should the
   next exploration draw wait for the A17 screen outcome (GO → war-v4
   composition is the obvious next arm) or is there a cheaper
   depth-composition candidate first?
4. **A17 canary contingency:** if the canary shows serve-fit failure at
   32768 (envelope NO-GO per A23 self-certification), does the FP8 fallback
   TBD-(i) search start today or wait for the formal AWQ NO-GO?
5. **boristown 1.47 public fork (NEW, from §1b):** the top public artifact
   now scores 1.47 — above our 1.33 best and at the gold cutoff. A
   byte-matched fork of a public 1.47 is the exact move that produced our
   frozen duck (ashvin 0.42 era) and is low-risk under fork-never-build.
   Questions for panel: (i) does adopting it violate any sealed discipline
   (it is a PUBLIC artifact — generalization-first memory says avoid
   luck-chasing, but this is a floor-raise, not a variance lottery)? (ii)
   fork-diff first ($0, today) then queue as filler-replacement? (iii) how
   does it interact with the exploration-budget accounting — is a public
   fork a "filler" (no window cost) or an arm?

## END OF PROPOSAL ##
=====================================================================

OUTPUT FORMAT (exactly this structure):
## Summary (2 sentences)
## Objections
For each: [SEVERITY] title — body (2-5 sentences, specific)
(minimum 3 objections in round 1; in later rounds, review your prior objections'
resolution first, then add new ones only if real)
## Questions for the authors (numbered)
## What I cannot judge
## Verdict: ACCEPT | MAJOR-REVISION | REJECT
## Score: N/10
