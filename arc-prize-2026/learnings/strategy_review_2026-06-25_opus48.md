# Verification summary

1. **Rank ~22-28, gap +0.20 to top-10** — [CORRECTED]. Actual rank ~57/1430 (10 teams tied at 0.43); top-10 cutoff 0.63, gap **+0.20** holds; top-30 cutoff is **0.47** (gap +0.04). Prior review's rank was off by 30 places.
2. **Tufa Labs 1.21 via StochasticGoose** — [CORRECTED]. Rank/score VERIFIED (1.21, #1, last submit 06-24). Architecture attribution **WRONG**: SG (Dries Smit, Tufa) scored 12.58% in Preview and **collapsed to 0.25% on the official launch benchmark**. The 1.21 result is from a different Tufa stack ("novel approach", per Tufa public update 0.68→1.17). SG is not what's winning.
3. **Top-10 cutoff 0.63** — [VERIFIED]. Rank 10 = face-of-agi at 0.63.
4. **Rodionov 2605.05138 = 58% RHAE via LLM-written Python sims** — [VERIFIED with caveat]. 58% is the **GPT-5.5** variant (15/25 solved). GPT-4 variant = 41.29% (8/25). v1 of the paper reported only 32.58% — the 58% headline depends on GPT-5.5 access.
5. **ARC Prize 2025 report = "no static method >11%, TTT mandatory"** — [CORRECTED]. Paper VERIFIED (2601.10904, Chollet et al., 2026-01-15). But the headline is **refinement loops** (per-task iterative program optimization), top Kaggle 24% on ARC-AGI-2 private, and zero-pretraining 7M nets. The "TTT mandatory" framing is 2024-vintage; 2025-26 SOTA is refinement loops where TTT is one component, not the mandate.
6. **Rudakov 2512.24156 = visual-feature priors, 3rd private LB** — [VERIFIED]. Training-free graph + visual salience segmentation, median 30/52 levels across 6 games, 3rd on **Preview Challenge** private LB. Code: dolphin-in-a-coma/arc-agi-3-just-explore.
7. **Forum 703990 = SG concurrency timeout bug** — [VERIFIED]. Thread exists; reports local ~1.56 → Kaggle 0.02-0.05 with sequential-lock concurrency. Reply: "most games timeout by the time the agent gets to starting them". STALE_MINUTES=15 in scorecard toolkit. This is real and actionable.
8. **SEVerA 2603.25111 = verifier-gated agent evolution** — [CORRECTED]. Paper VERIFIED but **does NOT target ARC-AGI**. It's constrained learning for agentic codegen on program repair / scientific discovery. Prior review's ARC framing was wrong; the technique is still portable but not an ARC paper.
9. **June 30 milestone with CC0/MIT-0 license gate** — [UNVERIFIED — prior claim]. Milestone date plausible; license-gate specifics not corroborated by the bundle. **Check the rules tab before any submission you intend to count for prize.**
10. **Pivot order: SG → visual priors → executable WM** — [CORRECTED below]. SG-first is wrong now that we know SG collapsed 12.58→0.25 on launch. Order should be **executable WM → visual priors → fix-then-retry-SG-only-if-concurrency-is-the-real-bug**.

# Where we stand

Rank is **~57/1430** at 0.43, not 22-28 — prior review materially overstated our position. We are tied with 9 other teams at 0.43; **top-30 cutoff is 0.47** (gap +0.04, very crackable), top-10 is 0.63 (+0.20), #1 Tufa is 1.21 (runaway, +0.78). Seven-day trajectory: {0.10, 0.22, 0.30, 0.21, 0.23, 0.16, 0.43, ERROR, ERROR, ERROR, 0.22, 0.28} — mean ≈0.23, with the 0.43 sitting in the upper tail of a high-variance distribution (v35 N=10 mean 0.246, std 0.094). We have not produced a non-noise gain since 2026-06-19. Three consecutive ERRORs on the forge35 slug cost a week; v62 ERRORed on fresh slug too. Milestone #1 is in 5 days; the LB will compress.

# What top teams are actually doing (verified)

- **Tufa Labs (1.21, #1)**: not vanilla StochasticGoose — that variant collapsed 12.58%→0.25% on full launch. The 0.68→1.17 jump is a **"novel approach"** publicly hinted at frame-change curiosity + better world model + TTT (Tufa research index, Digg coverage). Architecture not yet open-sourced. [VERIFIED rank, RUMORED architecture]
- **Rodionov executable world model (arXiv:2605.05138)**: LLM coding agent writes, verifies, and MDL-refactors a Python simulator per game, plans through it. **58% RHAE w/ GPT-5.5**, 41% w/ GPT-4. This is the strongest single-paper signal in the field. [VERIFIED]
- **Rudakov graph + visual salience (arXiv:2512.24156)**: training-free, 30/52 median levels, **3rd Preview-Challenge private LB**. Code is public. [VERIFIED]
- **DreamTeam (multi-agent DreamerV3-style)**: 38.4% RHAE on 25-game public set, beats 36% protocol baseline with 31% fewer actions. Role-decomposed workspace (observation/dynamics/strategy/probe/critique/arbiter) sharing contradicted predictions. [VERIFIED, paper id not surfaced — track]
- **ARC Prize 2025 thesis**: **refinement loops** (per-task iterative program optimization) define 2025-26 SOTA; TTT is one ingredient, not the only mandate. Best Kaggle ARC-AGI-2 = 24% private. [VERIFIED, prior review's framing CORRECTED]

# What we are NOT trying

- **Executable LLM-written world models.** Largest verified single-method gap vs our stack. JEPA is a latent representation, not an MDL-refactorable simulator.
- **Refinement loops / per-task program optimization.** The 2025-26 paradigm. We do strategy-mutation across submissions, not per-task program search at inference.
- **Frame-change action curiosity at scale** (the thing that actually got Tufa from 0.68 to 1.17, plausibly). Our CNN scores actions but we never built the prediction-based novelty bonus loop.
- **Test-Time Training.** Zero gradient updates at inference anywhere in v35.
- **Visual-feature action priors on graph explorer.** We have the graph; Rudakov's prior is a 2-day add we keep skipping.
- **Fixing the local→LB collapse before retrying SG.** Thread 703990 names a probable cause (lock-serialized concurrency + per-env timeouts). We never instrumented the gateway.

# Hypothesis check: v35+JEPA path

**Pro.** v63's throttling (n_sims=8, depth=3, ~120ms, cap 60/agent + 8/level, fresh slug) is the right surgical fix for v62's ERROR. If it clears, we get a clean read on whether a latent world model adds anything on the 14 BFS-unreachable games. The fresh slug also disambiguates "JEPA broken" from "slug cursed."

**Con (dominant).** JEPA-XXS at 2.3M params int8 is a representation, not a hypothesis generator. It cannot infer the latent rule of an unseen game — which is exactly what Rodionov's executable sim and Tufa's frame-change predictor do explicitly. Even if v63 clears, expected upside is +0.03 to +0.08 on a noisy floor, not structural. The 0.43→0.63 gap is paradigm-shaped (refinement loops, executable WMs, frame-change curiosity), not budget-shaped. And we have **5 days to milestone** — every submission slot spent on JEPA is one not spent on the actual frontier.

**Verdict.** Ship v63 because it's queued. **Do not iterate past it.** If it lands <0.30, kill the JEPA branch. If 0.30-0.40, hold as safety net. Either way, pivot in parallel **starting tomorrow morning**.

# Recommended next moves (ranked)

1. **Build a minimal executable-world-model loop with Claude as code generator (Rodionov 2605.05138 method, offline).**
   - Budget: **4 days**. Expected delta: **+0.08 to +0.20** if 2-3 BFS-unreachable games become solvable.
   - Risk: medium — Kaggle sandbox is no-internet, so simulators must be generated **offline against the 25 public games** and shipped as static Python modules selected by game-id at runtime. This is exactly what Rodionov's pipeline produces.
   - Evidence: 58% RHAE GPT-5.5, 41% GPT-4 — largest verified single-method gap vs our 0.43.

2. **Add visual-feature action priors to v10/v24 graph explorer (Rudakov 2512.24156 method).**
   - Budget: **2 days**. Expected delta: **+0.05 to +0.10**. Risk: low — code is public, training-free, no env coupling.
   - Evidence: 3rd Preview private LB, median 30/52 levels.

3. **Instrument the Kaggle gateway BEFORE any SG retry.** Log per-game start/elapsed inside the agent; submit a diagnostic-only kernel to confirm whether games are being reached at all (thread 703990 hypothesis).
   - Budget: **1 day**. Expected delta: **0 directly, but unblocks the SG-retry decision**. Risk: low — costs one submission slot.
   - Evidence: thread 703990 + our own 1.56 local → 0.02 LB collapse; STALE_MINUTES=15 hard wall.

4. **Conditional SG-retry only if (3) confirms concurrency starvation.** Rewrite to per-game state isolation, drop module-level lock, re-ship v55 architecture.
   - Budget: **2 days, gated on (3)**. Expected delta: **+0.10 to +0.25 IF the gap is infrastructure; ~0 if SG genuinely collapses on launch games** (per Smit's own 0.25% report). Risk: high without (3) data.

5. **Verify Milestone #1 license/eligibility rules tomorrow morning.** Check every notebook we'd submit for license headers, kernel-metadata.json correctness, and prize-track flags. One missing field zeros the milestone.
   - Budget: **0.5 day**. Expected delta: protects all other work. Risk: doing nothing is the risk.

**Explicitly NOT recommended:** any further v35 single-knob tuning, prompt A/B, or v35-redraw "just to see". Per `feedback_prompt_is_noise.md` and `feedback_simplicity_wins.md`, those are zero-EV.

# Risks & watch-list

- **Milestone #1 in 5 days.** Verify license/rules tomorrow. [UNVERIFIED — prior claim of CC0/MIT-0 — confirm before relying on it.]
- **JEPA slug curse.** v45 (2026-05-26) AND v62 (2026-06-20) both ERRORed; the fresh-slug fix for forge35 is now tested but **v62 was also fresh**. If v63 ERRORs, JEPA is dead regardless of slug.
- **Tufa & Tong Hui Kang will climb.** Expect Tufa 1.30+ and Tong 0.85+ by 06-30. Even +0.20 only moves us to ~top-20, not top-10. Plan for two climbs.
- **SG may genuinely not generalize.** Smit's own number: 12.58% Preview → 0.25% launch. Concurrency fix is **necessary but not sufficient** — don't bet the week on it.
- **Submission quota.** Three slots burned on ERROR in the last 6 days. Don't waste another on a redraw.
- **Watch:** v63 outcome tomorrow AM; any Tufa/Rodionov repo or notebook publish; thread 703990 replies naming a concrete gateway fix.

# Decisive instruction

**Tomorrow morning when v63 lands: regardless of score, do not queue another JEPA variant — spend the day starting the Rodionov-style executable-world-model loop (Claude offline-generating Python sims for the 25 public games) AND verifying Milestone #1 license rules; ship the Rudakov visual-priors v10 patch as the safety submission for the day.**
