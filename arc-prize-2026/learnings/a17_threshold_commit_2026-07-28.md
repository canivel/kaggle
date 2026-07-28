# A17 72B kill-threshold commit — 2026-07-28 (pre-observation seal)

Written and git-committed BEFORE `canivel/arc3-a17-72b-canary` v6 (full-window
bench, pushed 2026-07-28 ~07:26 EDT, status RUNNING at seal time ~08:35 EDT)
goes COMPLETE. Implements panel R22 directives D1 (publish v5-slice projection
+ pre-register the G2-FAIL branch), D3 (authors name Y, not the panel), D4
(no kill verdict off k=1), D5 (hash-commit before v6 COMPLETE). Panel is
advisory (restructure 2026-07-27); the governing instruments are the sealed
arithmetic gates cited below.

## 1. Governing kill rule (already sealed — reaffirmed, not invented today)

`learnings/a17_gates_numeric_2026-07-27.md` (sealed 07-27, BEFORE the v5
slice existed) defines, in the frozen-N form:

- ρ_action := 480 / Σ N₇₂B, where 480 = frozen 27B numerator (39 ft09 + 225
  sb26 + 147 lp85 + 69 vc33, measured actions per 7920 s window, w0_eval_s1)
  and Σ N₇₂B = the same-4-games executed-action sum from v6's
  `benchmark.json` (`len(game_run["history"])` — verified today to be the
  emitter behind `A17-CANARY N(...)`, `build_eval_notebook.py:651`; same
  units as the 480).
- **KILL: ρ_action > 3.5 ⇔ Σ N₇₂B < 138.** (Onepager quote: "ρ_action ≤ 3.5
  ⇔ Σ N72B ≥ 138".)

Direction stated unambiguously to discharge R22's directional-bug objection:
ρ_action is 27B-actions per 72B-action, so HIGH ρ_action = slow 72B = dead.
The daily-brief open-question phrasing "ρ_action < Y ⇒ dead" was a
transcription error in the brief, not in the sealed gates doc.

The panel's proposed collapse onto G2 (Σ N₇₂B < 100) is REJECTED as the
governing value: it is looser than the 07-27 seal, and loosening a sealed
threshold after adverse preliminary data (the v5 slice) exists would be
post-hoc. The stricter, earlier seal (138) governs. G2 (≥100 executed
actions) remains a separate prereg gate; both are evaluated.

## 2. v5-slice throughput projection (R22 D1 — published pre-observation)

v5 boot canary (07-27, 1500 s window, COMPLETE): Σ N₇₂B = 5 (ft09 2, sb26 1,
lp85 0, vc33 2), lc = 0, boot-to-serve ≈ 345 s, per-game window_s ≈ 1512.
Naive linear projection to the v6 7920 s window (≈7575 s post-boot):
**Σ N₇₂B ≈ 26–33**, i.e. ~4–5× BELOW the 138 kill line and ~3–4× below G2.
The five R22 reviewers independently derived the same range.

Known caveats, on the record before v6 is read: (i) k=1; (ii) the 1500 s
slice is warmup-heavy (torch.compile, cold MM cache, first-request effects)
so it plausibly understates steady-state rate — but a 4–5× recovery from
warmup alone would be surprising; (iii) the 07-27 token-arithmetic projection
(Σ N ≈ 414–434, ρ ≈ 1.11–1.16) is ~13× above the direct v5 measurement —
where they conflict, direct measurement governs.

Honest prior: **v6 is expected to FAIL the kill line.** This is stated now so
that a FAIL cannot be narrated as surprise, and a PASS cannot be narrated as
inevitable.

## 3. Pre-registered decision branches (sealed now, read after v6 COMPLETE)

Let S₁ = Σ N₇₂B from v6 seed 1 (games_present must be 4/4; any MISSING game
or window-drift WARN voids the read per amendment §7.2 and triggers the
retry branch, not a verdict).

- **B1 — S₁ ≥ 138:** kill line not triggered. Freeze null_adj at measured
  ρ_action per the 07-26 v4 prereg; seed-2 / scored-bench decision goes to
  the Sunday panel. G1 (recovery ≥ 0.95) and G2 read as preregistered.
- **B2 — S₁ < 138 (any margin):** implements R22 D4 (no kill off k=1).
  Exactly ONE confirmation seed fires as tomorrow's kernel slot-1 (free
  build hours; ~2.5 GPU-h; fits before the Aug 3 C4 deadline). No
  parameter, prompt, or harness change between seeds — a change would reset
  the count.
  - **B2a — seed 2 also < 138: the 72B route is DEAD on throughput.** A17
    lane closes. No third seed, no "one more fix" lane. Scored slots stay
    frozen-filler; the build priority reverts to the boristown
    readiness-gate A/B (R22 D2, 5/5). The fenced-recovery adapter and the
    weights-dataset route remain certified artifacts (reusable if a smaller
    multimodal model is ever benched) — the KILL is of the 72B-at-this-
    throughput screen, not of the multimodal contract.
  - **B2b — seed 2 ≥ 138:** discordant seeds → no verdict; variance is the
    finding. Escalate to Sunday panel with both draws; no scored bench
    before that.

No other branch may be invented after v6 completes. Anything not covered
above → no action, escalate to Sunday panel.

## 4. What v6 must still deliver regardless of branch (measurement only)

G1 parse-recovery rate over all tool-call attempts (numerator/denominator
raw counts, not just the rate), G3 cadence trace, per-game N + lc lines,
mm_cache hit-rate line, concurrency line. These feed the NC-4 offline parse
study and the error model whether or not the route dies.
