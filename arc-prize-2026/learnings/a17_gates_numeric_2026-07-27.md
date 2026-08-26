# A17 numeric PASS/kill thresholds — single-page assembly (2026-07-27)

Discharges **Panel R21 Directive #2** (4/5 reviewers: llm-agents, prog-synthesis,
rl-planning, systems; + methodology Q6) and **NC-5**: publish the numeric
PASS/kill thresholds in one place BEFORE the v6 full-window bench fires.

**This memo ASSEMBLES already-sealed numbers. It introduces no new thresholds.**
Every number carries its source document and sealing date. Where a number does
not exist in the sealed record, that is stated explicitly and it is NOT
invented here. The staged v6 build (`learnings/artifacts/a17_v6_staged_2026-07-27.md`)
is untouched by this memo — it remains the pre-registered window-restore-only
diff (3 hunks, smoke 56/56).

---

## 1. G1–G4, verbatim (sealed 2026-07-26, `learnings/war_room/a17_v4_prereg_2026-07-26.md` §1, filed BEFORE the v4 push; carried to v6 unchanged per ITERATION_LOG 2026-07-27 "QUEUED" and the v6 staging report)

Definitions from the prereg: T = analysis turns with non-empty assistant text
and no native tool call (eligible turns); H = fenced-recovery hits
(`fenced-recovery v1 hits=` counter + per-turn
`tool_calls_recovered_from_markup: yes`); X = executed actions
(benchmark.json `history` lengths summed over the 4 games).

> - **G1 recovery:** H/T ≥ **0.95**. On record: the offline 434/436 (99.5%) is
>   a SAME-SAMPLE CEILING (adapter designed on the turns it was validated on;
>   v4's turn population is not exchangeable with v3's stalled loop).
> - **G2 valid-canary minimum:** X ≥ **100** executed actions summed across the
>   4 screen games (v3: 0). Below 100 the canary is INVALID for any cadence or
>   capability statement regardless of H/T.
> - **G3 cadence (measurement, not gate):** actions/hour computed ONLY over
>   `step_executed=True` turns, reported per game with wallclock. The v3-derived
>   "1.1x" figure is RETIRED (stalled-loop bias, direction = false GO).
> - **G4 capability: NO GO/NO-GO interpretation from v4.** The sealed §9.1 gate
>   boolean stays unevaluated until (a) a v4-class run passes G1+G2 and (b) the
>   matched-concurrency 27B control leg (same 4 games, 4-way concurrent) or a
>   panel-ratified concurrency correction exists.

Numeric summary: **G1 PASS ⇔ H/T ≥ 0.95; G2 PASS ⇔ X ≥ 100; G3 = number filed,
no threshold; G4 = interpretation prohibition, no number by design.**
G1/G2 are *measurement-validity* gates for the canary itself. They are NOT the
performance gate — the performance gate is the sealed §9.1 boolean
(`a17_72b_screen_scope_v2.md` §9.1, sealed 2026-07-23: capability bar Σ ≥ 8
vs 27B Σ MAX = 6 + 2; parity 0.90·ΣN₂₇B; throughput-adj bar Σ null_adj + 1),
which v6 does NOT evaluate (see §5 below). Boot PASS ≠ G1–G4 PASS ≠ GO.

## 2. ρ_action: definition, denominator arithmetic, and the kill number

**Sealed definition** (`a17_72b_screen_scope_v2.md` §3, filed 2026-07-22):

> **ρ := ρ_action = (actions/s, 27B) / (actions/s, 72B)**, measured on the same
> 4 games, same SKU, same fixed 7920 s per-game window. Because the windows are
> identical, this reduces to an action-count ratio:
> **ρ_action = Σ_g N27B_W0(g) / Σ_g N72B_canary(g)**

**Numerator, frozen from disk 2026-07-22** (same §3): N27B_W0 =
39 (ft09) + 225 (sb26) + 147 (lp85) + 69 (vc33) = **480 actions / 7920 s
= 0.0606 actions/s**. So **ρ_action = 480 / Σ N₇₂B**; v6's deliverable is
exactly the denominator Σ N₇₂B.

**Kill threshold — SEALED, it exists; nothing needs proposing:**
ρ_action > **3.5** ⇒ ENVELOPE-INFEASIBLE, self-certifying NO-GO.
Chain of seals, all pre-observation:
- `learnings/stuck_review_v2_2026-07-23.md` §3 (2026-07-23): "measured penalty
  exceeds 3.5× the screen self-reports envelope-infeasible".
- `learnings/a17_error_model.md` (2026-07-23): "Envelope NO-GO (>3.5× penalty)
  self-certifies"; §4: "if ρ > 3.5 the envelope NO-GO self-certifies (C3)".
- `learnings/war_room/a17_envelope_onepager.md` (2026-07-24), verbatim:
  > **If the measured penalty exceeds 3.5× — i.e. the canary's pooled
  > ρ_action = 480 / Σ N72B > 3.5 — the screen self-reports
  > ENVELOPE-INFEASIBLE. That is a valid NO-GO datum by itself (physics), and
  > unlike a capability NO-GO it requires NO panel ratification.**

**In the panel's NC-5 orientation** ("ρ_action < Y ⇒ dead", which reads
ρ_action as a 72B *rate*): the sealed statistic is a 27B/72B *ratio* (higher =
slower 72B), so the sealed kill maps to: **Σ N₇₂B < 138 executed actions in the
7920 s window** (480/3.5 = 137.1; onepager: "ρ_action ≤ 3.5 ⇔ Σ N72B ≥ 138"),
equivalently 72B actions/s < 0.0173. Consequence on kill: 72B route dead =
war-v4 wall-closer closes per scope §8.1; slots revert to the frozen-fork
daily regime (ref_arc_daily_protocol). Note orientation carefully: **the kill
is ρ_action ABOVE 3.5, not below** — the "<" in NC-5's phrasing is the rate
form, not the sealed ratio form.

Two more sealed ρ_action landmarks (context, not gates):
- ρ_action ≤ **1.11** re-opens the ACTION-PARITY branch (scope v2 §1 dead-code
  check; error-model §4) — the 0.90 parity bar in action terms is 432 actions.
- Per R17 amendment §9.2 (sealed 2026-07-23), ρ_action is a **pre-run planning
  diagnostic only** for the gate boolean: null_adj is evaluated at the realized
  72B per-game N from the pull. The 3.5 envelope kill is the one place
  ρ_action itself remains binding (self-certification statistic).
- **The panel's asked-for "ρ_action ≥ X ⇒ expected LB band [a,b]" does NOT
  exist in the sealed record.** No LB-band mapping was ever sealed and this
  memo may not invent one; capability→LB translation is exactly what the
  sealed walk + Sunday panel own (§5). What IS sealed is the null side:
  Σ null_adj = 4 (ρ ≤ 2.5) / 3 (2.5 < ρ ≤ 3.0), frozen walk over
  ρ ∈ [1.5, 3.5] in `runs/a17_repair/per_seed_table.json`.

## 3. Throughput projection (from the v3 canary pull; v5 pull pending)

Measured serve throughput, v3 full-window run
(`runs/kernel_pulls/a17_canary_v3/analysis.md`, run COMPLETE 2026-07-25;
identical serve graft rides v5/v6 untouched — v6 staging report, cell 8):
- **gen_tps mean 34.3 / median 30.9 tok/s aggregate** over 4 concurrent
  streams (n=66 heartbeats; stall_s=0, restarts=0) ≈ **8.5 tok/s per stream**;
  ~67k generated tokens per game over the 7920 s window.
- Turn cadence: **436 analysis turns / window** (1200 LLM responses ≈ 2–3
  calls per observe-plan-act turn) → ≈ 615 generated tok per analysis turn
  (268k/436), ≈ 224 tok per LLM call.
- v5 (boot canary, 1500 s, RUNNING at memo time): boot/serve validation only;
  its short window is NOT ρ_action-eligible by construction (scope v2 §3
  requires the full 7920 s window). Its tok/s will be filed on pull as a
  diagnostic; no number exists yet and none is projected from it here.

**Projected actions in the 7920 s window** (arithmetic on filed numbers, not a
threshold): v6 = v3's turn cadence with the fenced-recovery adapter converting
turns to executed actions. At v3's 436 eligible turns/window and recovery in
[G1 bar 0.95, same-sample ceiling 0.995]: **Σ N₇₂B ≈ 414–434 executed actions
⇒ ρ_action ≈ 480/434–480/414 ≈ 1.11–1.16.** Caveats, both on record: (i)
turn≠action is a unit approximation (v3 analysis, envelope onepager); (ii) the
99.5% is a same-sample ceiling (G1 text) and v4/v6's turn population is not
exchangeable with v3's stalled loop. Reference points this projection clears:
G2 floor 100 (×4), envelope kill floor 138 (×3), and it straddles the sealed
1.11 parity-branch boundary — which is why the number must be *measured*, not
assumed. Addressing systems' "boots but decodes at 4 tok/s" scenario: that
world is exactly ρ_action > 3.5 (Σ N₇₂B < 138) and is covered by the sealed
self-certifying kill in §2 — a slow-but-booting 72B cannot pass silently.

## 4. Runtime bound of the staged v6 (NC-3, answered from the staged build — no modification)

The staged v6 is structurally time-bounded by machinery already in the
notebook (all pre-existing; v6's prereg'd diff is window-restore-only, 3
hunks, per `a17_v6_staged_2026-07-27.md`):
- **A17_WINDOW_S = 7920.0** — per-game soft deadline with all 4 games
  concurrent (log-verified semantics, scope v2 §5: `max_runtime_s_per_game=
  7920.0`; one push = one ~2.2 h window + ~0.2–0.3 h load ≈ **2.5 GPU-h**).
- **Budget-derived soft_end** (restored verbatim from
  `build_v5_boot_canary.py` CELL14_REWRITES originals):
  `soft_end = min(soft_end, NOTEBOOK_START + (budget − min(600, budget/2)))`
  — anchored at notebook start, so model-load time cannot extend the window.
- **Zero-action abort**: heartbeat thread hard-exits 71 (`A17-CANARY
  ZERO-ACTION-ABORT`) if `actions_total == 0` at any heartbeat in the armed
  range **[1800 s, 7320 s]** (kill-disarm = A17_WINDOW_S − 600; v4 prereg §3
  — "saves ~1.7 GPU-h vs v3's full-window burn"); non-empty again at the
  restored window. Plus exit-70 fatal machinery and window-drift WARN vs
  7920 s (smoke S5 verifies all of it present in the staged build).
- A hung vLLM boot dies at the cell-8 FAIL-LOUD boot asserts (`A17-CANARY
  FATAL`, minutes not hours — v1/v2 died at boot for ~0.01 GPU-h total, v4
  prereg §4 table).

Net: worst normal case ≈ 2.5 GPU-h < the panel's 3 GPU-h figure; the
pathological cases (hung boot, zero-action) exit early and LOUD. NC-3 asks
for a "hard resource abort (e.g., kill at 3 GPU-h)": the staged build's bound
is window-derived rather than a literal GPU-h counter. If the panel requires
the literal counter form, that is a v7-class change requiring its own
pre-registration — it is NOT added here, because the v6 diff is sealed as
window-restore-only and this memo does not modify builds.

## 5. Authorization scope (explicit, per directive)

**boot canary PASS (v5) authorizes only the MEASUREMENT run (v6); no GO/NO-GO
or capability reading from either (a17_error_model.md k=1 false-NO-GO=1.0);
interpretation only via sealed walk + Sunday panel.**

Sequencing stays as sealed (v5 memo / scope v2 §7): v6 numbers filed → freeze
null_adj at realized actions (R17 §9.2) → seed-1 scored bench → gate evaluated
only after seed 2 (+ marginal seed if Σ ∈ {6,7} or branch-2-within-1). A v6
push remains NC-1 escalation-relevant (first push of a new artifact version);
this memo is the Directive-#2 numeric publication that NC-5 requires before
that push.
