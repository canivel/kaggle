# Daily Brief — 2026-07-19 (Sunday)

## §1a Result deep-dive

### Draw #5 = 0.76 → sealed A5/A8 look = FAIL → war accumulation ENDS
War draw #5 scored **0.76 — campaign low** (observed LB range now 0.76–1.33). The sealed A5/A8 look was executed pre-loop this morning (`runs/a5_a8_look_2026-07-19.json`, deterministic arithmetic, consequences sealed 07-18 before observation): war ledger n=5 {0.91, 1.08, 0.88, 1.05, 0.76}, mean 0.936, σ̂ 0.1309, **χ²-CI-hi 0.376 ≥ 0.25 → FAIL**. Sealed consequence (A8): the war arm is accumulation-only permanently — ineligible as an A/B readout arm at any n; no mechanism may cite war-arm LB deltas as evidence either way. Per A9, war accumulation ends (no draw #6); the frozen fork resumes as filler (already queue head).

Was the pre-registered expectation met? Yes in the meta sense: R13 predicted the gate was near-unpassable (the draw needed [0.955, 1.005] to pass), and the A8 amendment sealed the fail-consequence before observation — the machinery worked exactly as designed. **Validation bonus:** pooled n=11 across both arms → mean 0.9655, σ̂ **0.154** — dead-center in the LB process model's predicted 0.13–0.17 bracket. The σ=0.074 era is formally closed. Window pricing stands: filler E[max@~106] ≈ 1.39; experiments must credibly claim ≥ +0.06–0.12 to price a window.

### W0 continuation eval — mechanism PASS, non-inferior, (f) hygiene CONFIRMED
`canivel/arc3-duck-w0-continuation-eval` v1 completed (2h12m, 25 games). Full screen: `runs/kernel_pulls/w0_eval_s1/screen_report.md`.
- **Tripwires:** all 3 banners present; zero warpack/ledger lines (the only greps are the "NO warpack" banners themselves).
- **Mechanism (pre-registered expectation: 0 idle turns): PASS — 49 GAME_OVER episodes across 12 games, 0 idle post-game-over actions.** Every game over recovered on the immediately following action (bp35 alone had 13 episodes).
- **Non-inferiority (descriptive, no score gate per the 0.00 counting bound):** 16 total levels — inside the ledger-OFF seed band {13, 15, 22}; mean 1.73 vs seeds {1.16, 1.58, 1.62}. No game below the 3-seed floor.
- **Interpretation:** (f) is pure hygiene and it works perfectly. Its value is eliminating idle-turn tail risk, not adding levels (Qwen recovers from game over anyway; the graft makes it deterministic). Author recommendation to R15: adopt continuation as a **default layer in all future builds**; seed-2 unnecessary (mechanism is deterministic, 49/49; there is no score gate a second seed would feed).

### Leaderboard
Leader Yuto Kojima 1.86 (resubmitted 00:02 today). Wall unchanged: 16th place = 1.44; ~14 teams in the 1.44–1.61 band; Tufa Labs 1.45. Our best 1.33 (frozen-fork right tail). No structural shift.

## §1b Discussions sweep (since 07-18)
Three threads active; no adopt-worthy LB technique.
- **#727505 "Constraint Before Control" (Yakunin, new):** verifier-holds-sole-action-authority architecture; author's own result 0.17 and only after disabling nearly everything. **ADAPT (low): conceptual convergence with our EWM plan-execute-verify contract; no validated technique to lift.**
- **#727119 host "500 submissions" thread, new comments:** Yakunin reports **reset-logic fragility** — a 5-resets-per-level cap turned a 9-min working agent into a 1-hour 0-score run. **ADAPT: concrete landmine for the war-v4 72B screen and any experimental window — reset-path changes must be A/B'd against the frozen fork before trusting.** Model-stack chatter (Qwen-3.6-27B/Gemma-4-31B opinions): IGNORE (unvalidated). Zejun_ team-up drama: IGNORE.
- **#726367 AGI-timeline thread:** IGNORE (speculation; mild confirmation the public field is stuck near 1.86).

## §1c Research sweep (since 07-16; back-filled early July)
Two ADAPTs land directly on our EWM step-0 abort problem:
- **OCM — arXiv:2607.02846 (ADAPT, high priority):** coupled object-knowledge + procedure-knowledge codebases; **procedures verified against the refined object model before execution**. Candidate fix for our step-0 aborts: validate the planned procedure against the model pre-execution instead of discovering phase misalignment at live step 0 — decouples verification from timer/hidden-counter phase.
- **World-model collapse phase transition — arXiv:2606.31399 (ADAPT):** "world-state fidelity fails before action validity." Diagnostic frame: our step-0 aborts are likely world-state-fidelity (phase) failures, not planning bugs → argues for an explicit **re-observe/resync step before declaring abort** rather than fail-closed on first mismatch.
- AgentLTL arXiv:2607.02599 (ADAPT low, park): online prefix-gating formalism; verifies ordering not state equality.
- Agentic TTT arXiv:2607.03441: IGNORE (needs training data; Kaggle no-weight-update regime).
- 72B AWQ on 96GB: **no external throughput anchor exists** — memory fit confirmed comfortable (~35–40GB weights), throughput must come from our own A17 bench. We are the reference.

## Weekly fingerprint table (Sunday step ii)
16 incidents, 8 recurring families. Top families: `class:ERROR:none` n=7 (05-26→06-28), `provenance:scratch-built` n=5 (the arc3-final/forge35/jepa/execwm scratch-drift cluster — now blocked by preflight), `slug:canivel/arc3-final` n=4, `class:COMPLETE:0.00` n=3, `slug:arc3-pilot-eval` n=3 (07-07/08), `class:COMPLETE:null-band` n=2. No NEW incidents this week — preflight recurrence WARN + fork-never-build holding.

## Open questions → R15 (full 5-reviewer, circulation per A20 via untruncated @file delivery)
1. **A14 recalibrated gate seal** — this circulation is the sealing one: cumulative stack-vs-W0 look as THE binding score decision; per-window looks = mechanism prong + non-inferiority only. Confirm or object now; seals on this round.
2. **EWM Stage-1 re-pricing** — Stage-0 dry-run showed held-out saturation does NOT transfer on-trajectory (vc33 + s5i5 = 2 of 5 targets abort at step 0). Discount the +0.10–0.30 expectation; and rule on the two candidate step-0 fixes from today's sweep (OCM pre-execution procedure validation; resync-before-abort per 2606.31399).
3. **W0 disposition** — adopt continuation as default hygiene layer in all future builds? Seed-2 needed? (Author: yes-default, no-seed-2.)
4. **W1 owner = (a) budget sentinel** (A6 deadline Jul 20) — confirm scope + A10 compressed-budget canary design; (c) Reki-signature suppression disposition (build order after (a)?).
5. **A17 72B screen scope** — weights dataset + vLLM bench kernel on the free rail; reset-fragility caution from §1b applies; go/no-go = ≥2 levels beyond 27B on ft09/sb26/lp85/vc33 AND throughput-adjusted null formula. Pre-Aug-1 blocking.
6. **state_of_the_war priority ratification** — EWM-execute line vs design-doc ordering, given the Stage-0 discount (Q2) and that EWM remains the only uncontested-edge line.

---
## ADDENDUM (post-panel, same day): R15 outcome
R15 completed on the first fully-untruncated circulation (56.7K via @file delivery; all 5 reviewers confirmed END lines — the R14 truncation defect class is dead). **Verdicts: 0 ACCEPT, 5× MAJOR-REVISION, no new FATALs.** THE convergent directive (5/5): **A14 does NOT seal this round** — A18's (d)-kill was never propagated into §2/§4/P(pass); the republication (post-(d) stack ≈ +0.21 rail ceiling, α re-derivation, dismantle branch, guard calibration, (c) disposition) circulates as R16 and the gate seals there. Full synthesis: `learnings/panel/round15/_directives.md`. A17 scope doc filed same day (`learnings/war_room/a17_72b_screen_scope.md`) incorporating the panel's gate-boolean/comparator/SKU repairs — headline discovery: **the harness is multimodal; the 72B swap must be Qwen2.5-VL-72B-Instruct-AWQ**.
