# ARC-AGI-3 Decision Brief — External Review 2026-07-06

Advisor: claude-fable-5 · Scope: strategy layer (code forensics on v66 running separately)

# Position

- **Us:** 0.43 (v61, 2026-06-19), rank **275/1626**. 268 teams above; 125 teams ≥1.0.
- **Cutoffs today:** top-10 = 1.35 · top-20 = 1.28 · top-50 = 1.16 · top-100 = 1.04.
- **Milestone #2 (Sept 30) target:** extrapolated top-20 needs **1.8–2.2**. That is 4–5x our best. v35-family tuning cannot close this; the noise band {0.14–0.43} is a ceiling, not a ladder.
- **Process failure:** zero submissions since 6/29 — 7 days of dead queue during the fastest field-wide climb of the competition. Cadence is now a hard requirement (see 7-day plan).

# What changed externally in the last 2 weeks

1. **The field tripled.** Jun 24 frontier ~0.51–0.57 → leader 1.56 today; all top-30 submitted in-window and several brand-new team IDs jumped straight to top-30. This pattern is diagnostic of **public code diffusion**, not 30 independent breakthroughs.
2. **The smoking gun is identified:** Jeroen Cottaar / Tufa Labs won Milestone #1 (June 30) and **open-sourced the winning notebook + writeup** immediately (amplified by Kamradt/Chollet; "only submission beating the competition baseline"). Community describes it as foundation-model prompting + tooling, i.e., **orthogonal to our BFS+MCTS+CNN stack**. The 3rd-place solution is also open. We have absorbed neither.
3. **Research:** Rodionov exec-WM (arXiv:2605.05138) remains public SOTA (15/25 games, 58.12% RHAE) and shipped a **v2 with a leakage audit** — v1 numbers may have partly benefited from now-patched information channels; read v2 before trusting technique transfer. **AERA (arXiv:2605.25931)** argues the public 25-game set is heuristic-solvable and only the private 55-game eval is a genuine test — reinforces our generalization-first stance. TTT-for-interactive remains an open, unclaimed gap.

**Implication:** our single most valuable action this week is not invention — it is **pulling Cottaar's open-sourced notebook via authenticated kaggle CLI, reproducing it with exactly matched kernel metadata** (per the env-match rule: same code + wrong env = wrong score, confirmed 5x), and only then layering our differentiators on top.

# Strategy verdict: exec-WM lane — GATE (do not iterate, do not park)

- **Do not iterate blind.** v66 (v35 + ExecWMHook, 25 opus-generated sims) scored 0.14 — bottom of the v35 noise band. Iterating before forensics completes is spending submissions on an unexplained regression, and Rodionov v2's leakage disclosure means the technique's true headroom is less certain than a month ago.
- **Do not park.** Exec-WM is still the published SOTA family, Tufa is quiet on their current method, and it is our best long-term differentiator for the private eval (generalizes; not a public-set heuristic).
- **Gate criteria to resume:** (a) KAOS forensics identifies why v66 underperformed (hook interference vs. sim quality vs. budget cannibalization); (b) local harness shows an ExecWM-gated build **at or above the v35 band median**, not just inside it; (c) Rodionov v2 leakage section reviewed to confirm our sims don't rely on channels the official harness closes. Until all three: exec-WM ships only as the gated variant below, never as the primary lane.
- **Priority inversion:** the Cottaar reproduction lane outranks exec-WM this week. A public ~1.2+ baseline we haven't absorbed dominates any 0.43-band experiment.

# Tonight's submission recommendation: **v61-equivalent resubmit, fresh slug**

- **Pick: v61-equivalent** (v35 + action-replay + dynamic BFS budget + macro-probe). It is our only build with a verified 0.43 draw; it restores cadence immediately and banks a known-good result while we absorb Cottaar's code.
- **Reject v66-gated tonight:** forensics incomplete, the gated variant has never been runtime-tested end-to-end, and untested pushes have already cost us 0.00s twice. It goes in the queue later this week behind its gate.
- **Reject baseline redraw:** strictly dominated by v61 — same variance mechanism, lower band.
- **Execution requirements:** fresh kernel slug (arc3-forge35 slug is burned; slugs accumulate hidden state), mandatory `scripts/preflight.py`, structure cloned from arc3-baseline.ipynb, explicit `kaggle competitions submit -k <kernel> -v <ver>` (kernels push alone does not submit).

# 7-day plan (queue is never empty — 7 concrete entries)

| Date | Queue entry | Notes |
|---|---|---|
| **Jul 6** | **Q1: v61-equivalent, fresh slug** | Preflight-checked. Restores cadence + 0.43-band draw. |
| **Jul 7** | **Q2: Cottaar Milestone-1 repro, unmodified** | Pull notebook + writeup via kaggle CLI today. Fork with **exact** kernel-metadata (enable_gpu, dataset_sources, docker_image). Goal: confirm we can hit the public ~1.2 band at all. |
| **Jul 8** | **Q3: Cottaar repro, env/params corrected** | Fix whatever Q2's run log shows. Fallback if repro blocked: second v61 draw — the slot ships regardless. |
| **Jul 9** | **Q4: v67 = Cottaar base + our ACTION6 set_data fix + uncapped-BFS fallback on games where prompting stalls** | First hybrid: their frame-reasoning, our long-horizon search. Runtime-tested locally first. |
| **Jul 10** | **Q5: v66-gated (ExecWM fires only on level-up predictions)** — *only if all three gate criteria pass*; else v67 tuned variant | Forensics deadline is Jul 9 EOD to make this slot. |
| **Jul 11** | **Q6: per-game router** — dispatch each game to Cottaar-style vs forge_v35 by early-frame signature | Targets private-eval generalization (AERA's point: don't overfit public heuristics). |
| **Jul 12** | **Q7: week's best re-draw + strategy checkpoint** | Re-submit highest scorer; write iter review; queue AERA-style explore/verify/plan probe for week 2 if capacity. |

**Standing rule:** every evening the daemon must find ≥1 entry in `submission_queue.json`; if an experimental build misses its gate, the slot auto-falls-back to the current best-known build. A wasted draw beats an empty queue.
