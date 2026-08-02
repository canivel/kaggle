# ARC-AGI-3 Discussion Sweep — 2026-08-01

Scope: threads posted or with comment activity NEW since 2026-07-31 (~24h window).
Sort checked: `?sort=new` (Recently Posted), page 1 (rendered via chrome-devtools snapshot; WebFetch returns empty JS shell).
Plan context for verdicts: frozen-fork duck harness (public 1.33 best, #51; ledger n=15, μ0.973 s0.134); LB head KOJIMA 1.86, Andy liu 1.69, gold cutoff ≈1.50; active build lane = preregistered single-diff A/B on boristown's vLLM readiness-gate (1.47 anchor, Sunday R23 ratifies); hard constraints: 9h rerun wall, no-final-rerun (host-confirmed), zero cloud budget, 2 kernel pushes/day, dual-notebook execution mechanic (known), OpenAI compaction 13.3%→38.3% (known).

## New posts / activity

| # | Title | Author | Posted / Activity | Gist | Verdict | Reason (vs plan) |
|---|-------|--------|-------------------|------|---------|------------------|
| 1 | [\[LB 1.17\] ARC-AGI-3 Qwen3.6 Duck + 300-Game Diagnostics](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/731522) | yw8837 (87th) | Posted ~1d ago (07-31); 1 appreciation comment 20h ago by Greg Kamradt (host, "Thank you for sharing") | Full open-source release of a TAAF/Duck-harness fork that scored **1.17 public**, plus a dataset with 11-submission ledger + 300 per-game diagnostic rows. Model = **vrfai/Qwen3.6-27B-FP8** on Kaggle RTX PRO 6000, runtime 2h21m, baseline settings kept (concurrency 28, 7920s/game). Two patches: **P1** = if an identical cardinal move leaves the visible board unchanged inside a batch, stop before repeating it and return control for re-observation (repeated-no-effect guard); **P2** = raise local analyzer yield window 60s→90s for an extra inspect/tool cycle on hard turns. 11 official scores: 1.29, 1.05, 0.71, 0.73, 0.68, 0.75, 0.55, 1.11, 1.17, 0.90, 0.94. Author's own conclusion: **more actions/tokens/levels did NOT reliably raise the hidden score** (high variance, non-monotone). | ADAPT | Same harness family as our frozen fork; a peer's public ledger. Do NOT adopt the whole notebook (1.17 < our frozen 1.33; their model is Qwen3.6-27B-FP8, ours is the duck default — swapping is off-plan and off-budget). Two extractable, low-risk mechanics worth a preregistered single-diff *after* the boristown gate A/B resolves: **(a) P1 repeated-no-effect guard** — cheap idempotent-action suppressor that could cut wasted actions against the 9h wall; **(b) P2 60→90s analyzer yield window**. Both are additive to the readiness-gate lane and should be queued as separate single-diff arms, NOT bundled. Strongest takeaway needs no code change: their 11-run spread (0.55–1.29, σ≈0.24) is **independent external confirmation** of our own high-variance ledger — reinforces the prereg/gate discipline (don't chase a single high public scalar) and the "no final rerun" caution. Their 300-game diagnostic dataset is a free analysis artifact; flag for offline mining under zero-cloud (it's data, not compute). |

## Notes on boundary / bumped items (no new adoptable content in window)

- **Dual Notebook Executions** (731290, Alex Paul) — new comment 17h ago by **CPMP** (Grandmaster). Same known code-comp mechanic (save-version run vs. hidden scoring rerun) covered in the 07-31 sweep; CPMP reply adds only standard confirmation. **IGNORE** — no plan change.
- Pinned host threads (deadline 713634, submission-limits 705405, accelerators 697720/695158, code-reqs 697944, getting-started 684625) — all last activity ≥4d ago; unchanged from prior sweeps. IGNORE.
- MDL neuro-symbolic self-promo (730225, Hayford Kofi Quaye) — still -6 votes, no level-completion evidence, unchanged. IGNORE.
- "Three clarifications on final scoring" (729985) — last comment 4d ago, unchanged. IGNORE.
- "Claude Opus 5 achieves 30% on ARC-AGI-3" (728934, Geremie Yeo) — last comment 7d ago; outside window, already known-class (frontier-model score report, no Kaggle-runnable path under zero-budget). IGNORE.

## Verdict summary
- New posts in window: **1** (yw8837 LB-1.17 Qwen3.6 Duck release, 07-31).
- Bumped older threads with <24h comments: 1 (Alex Paul dual-notebook — CPMP reply; IGNORE).
- Non-IGNORE verdicts: **1 ADAPT** (yw8837).
- Plan change: **none to the active lane.** Boristown readiness-gate A/B remains the current single-diff; frozen duck fork filler nightly continues. Queue two *future* preregistered single-diff arms from yw8837's release — (P1) repeated-no-effect action guard, (P2) 60→90s analyzer yield window — to run AFTER the gate A/B resolves, each as its own arm, never bundled. Their 11-run 0.55–1.29 spread is external confirmation of our variance thesis; no course correction, only reinforcement of prereg/gate discipline. Their 300-game diagnostic dataset flagged for free offline mining (data, not compute).
