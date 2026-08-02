# Round 23 Panel Directives (synthesized 2026-08-02)

Panel role: ADVISORY for the build rail under the 07-27 restructure (§2: scored-window
promotion decisions are governed by sealed arithmetic gates, not panels; MAJOR-REVISION is
the known absorbing state and does not block the build rail). **EXCEPTION for THIS cycle:**
§7.3 of the boristown A/B prereg REQUIRES R23 governance ratification before the first gated
draw may fire, so for agenda item 1 the panel's disposition is *binding input* to whether the
seal completes today. The proposal reviewed is `learnings/daily_brief_2026-08-02.md`.

## VERDICT SUMMARY
Unanimous 5/5 MAJOR-REVISION (scores 4,4,5,4,5; 0 FATAL, 24 MAJOR): the brief is
operationally clean — the pre-registered second-sub-0.80 watch-rule fired and was executed
(`stationarity_2026-08-02.md`), and both entry gates carry real evidence artifacts — but the
centerpiece "stationarity guard" for the boristown A/B is statistically incoherent along four
independently-derived axes, and every reviewer concludes option (a) as written must not go to
a vote as a coequal standalone remedy. The four convergent defects: (1) the corrected bar
1.1701 is built on an *external* variance (yw8837's σ≈0.24) that our own 15-draw sealed
control rejects at p≈0.007 and that is used twice in opposite rhetorical directions; (2) the
brief's stated direction-of-bias ("step-down ⇒ biased TOWARD spurious promotion") is
backwards — a step-down biases toward spurious *non*-promotion; (3) the headline tail
probabilities (0.38% / 18.5%) do not reproduce from any stated model (reviewers get ≈0.98%
and ≈5.6%); and (4) no power / value-of-information analysis accompanies any of the three
options, and the readiness gate the A/B tests fired at `vllm_ready_latency_s=0.0`, i.e. it was
never exercised in its actual failure mode. The strict majority remedy is design (b)
(interleaved contemporaneous controls, paired test), optionally preceded by (c).

| Reviewer       | Verdict        | Score | n_fatal | n_major |
|----------------|----------------|-------|---------|---------|
| rl-planning    | MAJOR-REVISION | 4/10  | 0       | 6       |
| llm-agents     | MAJOR-REVISION | 4/10  | 0       | 4       |
| prog-synthesis | MAJOR-REVISION | 5/10  | 0       | 4       |
| methodology    | MAJOR-REVISION | 4/10  | 0       | 6       |
| systems        | MAJOR-REVISION | 5/10  | 0       | 4       |

## TOP DIRECTIVES (ranked by cross-reviewer agreement)

1. **[5/5 reviewers: llm-agents, methodology, prog-synthesis, rl-planning, systems] Do not
   present option (a) as a coequal standalone remedy — adopt the interleaved-control design
   (b) (optionally (c)-then-(b)); the sealed-μ-plus-inflated-σ bar corrects the wrong
   nuisance parameter.** All five reviewers independently reconstruct that 1.1701 = sealed-μ
   0.9727 + a σ=0.24 quantile with μ left pinned to the seal, and all five state the
   brief's own direction-of-bias claim is inverted: methodology — "If the level stepped down
   (0.973→0.665 …) bias toward spurious *non*-promotion"; prog-synthesis — under mean 0.665
   the 4-draw mean clearing 1.1701 is "a +4.2σ event (P≈10⁻⁵) — the A/B becomes a guaranteed
   fail that burns 4 scored draws and yields zero information"; systems — "arm-B draws from a
   degraded regime compared against a stale sealed control mean of 0.9727 produce a biased
   treatment-effect estimate in *either* direction … recommending [(a)] as such is a design
   error"; rl-planning — "Only option (b) … is robust to the level ambiguity"; systems —
   "R23 should mandate (b) and drop (a)". *Implementability:* (b) is a scheduling change to
   the sealed prereg (alternate gated/filler, paired test) — must specify the paired test,
   df, pairing/alternation order, and mid-interleave sub-0.80 stop rule BEFORE sealing
   (methodology Q5, prog-synthesis Q3).

2. **[5/5 reviewers: llm-agents, methodology, prog-synthesis, rl-planning, systems] Attach a
   power / MDE / value-of-information table for each of (a)/(b)/(c) before R23 votes — a
   4-draw gated arm at σ∈[0.13,0.24] is underpowered against any plausible gate effect.**
   methodology: bar 1.17 with n=4, σ=0.24 gives "roughly 30–40% power against a true lift of
   +0.15 … will likely return an uninformative negative"; systems: "SE≈0.067 at n=4 — power
   well under 30% … under σ≈0.24 the arm is hopeless (SE≈0.12)"; rl-planning: "State the
   minimum detectable effect at 80% power … the expected draws-to-decision under (a)/(b)/(c)";
   prog-synthesis Q5 and llm-agents Q5 both demand the power at +0.10/+0.20. *Implementability:*
   one power table (n, σ, bar → power at +0.10/+0.15/+0.20/+0.40) plus a draws-to-decision
   column; blocks the vote, not the build. systems/methodology both note sizing the arm at
   n=8–10 interleaved is cheap against ~92 remaining slots.

3. **[5/5 reviewers: llm-agents, methodology, prog-synthesis, rl-planning, systems] σ≈0.24
   is imported from yw8837 with zero provenance and is inadmissible in a decision rule until
   shown exchangeable with our process — our own data reject it.** prog-synthesis quantifies:
   "χ²₁₄ = 14·(0.1343/0.24)² ≈ 4.38, i.e. P(s ≤ 0.1343 | σ=0.24) ≈ 0.007"; systems demands
   "a variance-ratio test … with n=19 and s=0.159, is σ=0.24 even inside the 95% CI? Show it";
   llm-agents flags "the convenient asymmetry: the imported σ is used to *dismiss* alarming
   evidence and to *justify* proceeding" and requires "a fork-diff + ledger provenance check
   for yw8837 equivalent to `fork_diff_boristown_2026-07-24.md`". methodology and systems both
   prefer estimating σ from our own contemporaneous data — which is exactly what (b)/(c)
   produce. *Implementability:* F-test / bootstrap CI on s from n=19 vs σ=0.24 is a
   ten-minute check; the yw8837 fork-diff memo is a half-day. Either kills or licenses the
   σ=0.24 figure. **→ NC-13.**

4. **[4/5 reviewers: llm-agents, methodology, prog-synthesis, systems] Publish the exact
   event definition and reproduction script for the 0.38% / 18.5% pair probabilities — no
   reviewer can reproduce them.** methodology: "P(draw ≤ 0.80)² ≈ 0.99%, P(≤0.68)·P(≤0.65) ≈
   0.012%; neither is 0.38% … under σ=0.24 … 5.6–7.8%, not 18.5%"; systems: "≈0.98% … and
   ≈5.6%, not 18.5% (to get 18.5% per-pair you need μ≈0.84, which is nowhere stated)";
   prog-synthesis and llm-agents concur these two numbers are "the entire quantitative basis"
   / "the quantitative pivot" of the whole guard. *Implementability:* attach the script
   (thresholds, joint vs. scanned-over-19, Gaussian vs. t, assumed μ) to
   `stationarity_2026-08-02.md`, as was done for the ledger stats. Blocks the vote.

5. **[5/5 reviewers: llm-agents, methodology, prog-synthesis, rl-planning, systems]
   Re-derive the harm-pause as a paired/relative criterion — a fixed <0.80 rule is NOT
   "unaffected" and trips at coin-flip-or-better rates under the very regimes motivating the
   guard.** prog-synthesis computes the exact figures: "P(gated draw < 0.80 | null) = 9.9% …
   P(≥1 spurious harm-pause in 4 draws) ≈ 34%; under σ=0.24 ≈ 66%; under the step-down level
   0.665 it fires almost surely" — "'unaffected' is false"; llm-agents: under the 0.665
   regime "gated draws will trip the <0.80 harm-pause with high probability *regardless of
   the gate's true effect*, and the A/B terminates as spurious 'harm'"; the paired-criterion
   fix is exactly design (b). *Implementability:* fold into the (b) spec — harm defined as
   gated minus contemporaneous filler, not an absolute floor.

6. **[3/5 reviewers: llm-agents, prog-synthesis, systems (+methodology Q4 as conditional
   inference)] The readiness gate fired at 0.0s latency — a null observation, not a green
   marker; supply the in-environment vLLM ready-latency base rate (and a fault-injection leg)
   before A/B-testing what may be inert code.** systems: "A gate that 'fired' at 0.0s latency
   means the server was already up when first polled … the eval confirms only that the gate
   is inert when unneeded … run a fault-injection leg (artificially delay vLLM startup by
   60–150s)"; llm-agents: "you are about to spend 4 scored draws A/B-testing dead code …
   supply the base rate of vLLM cold-start delays/failures across our historical Kaggle
   runs"; prog-synthesis frames the metric mismatch — "its causal mechanism is rescuing
   occasional slow-start/stall runs — a rare-event, left-tail-truncation effect, not a mean
   shift … Specify a mechanism-matched statistic (stall/failure incidence, min-draw, or
   left-tail mass)". *Implementability:* grep historical kernel logs for observed ready
   latency (fast, decides whether the arm has any live mechanism at all); fault-injection leg
   is a build. **→ NC-14.**

7. **[4/5 reviewers: llm-agents, prog-synthesis, rl-planning, systems] The strategic ceiling
   is dominated — the A/B's best case (~1.47) is already below the rising gold cutoff (1.54,
   +0.05/week); do not serialize the compaction lane strictly behind it. Approve a parallel /
   fixed-split slot allocation or a hard cap on A/B draw expenditure.** rl-planning names it
   "a bandit misallocation: the arm with the higher ceiling is queued behind an arm whose
   ceiling is already dominated"; systems: "Justify the serialization with a projected-cutoff
   model or change it"; llm-agents: "State an explicit slot budget split rather than a strict
   priority ordering." *Implementability:* a slot-budget line through Nov 2 (A/B gated / A/B
   controls / compaction / reserve) — rl-planning Q5, systems Q5.

8. **[3/5 reviewers: rl-planning, systems, methodology] Adopt option (a) only as a *sealed
   modified design* — post-hoc bar change after observing the two lows is data-dependent
   modification; if used at all, seal the change-log before the first gated draw.** systems:
   "amending the sealed bar post hoc … is exactly the data-dependent modification
   preregistration exists to prevent; if option (a) is adopted, the A/B should be re-labeled
   as a *modified* design with the change-log sealed before the first gated draw";
   rl-planning Q6 and prog-synthesis Q6 both ask whether the sealed prereg even contains an
   escape clause permitting 1.0970 → 1.1701 or whether ratifying (a) sets an
   "unsealed-amendment precedent." *Implementability:* governance note in the seal commit.

9. **[3/5 reviewers: rl-planning, prog-synthesis, systems] Add a minimum-segment-length
   constraint (≥3) to the pre-registered change-point scan and disclose whether the
   permutation null was the max-over-splits statistic — the n₂=2 right segment is a
   degenerate |t|.** systems: "Welch |t|=8.64 splitting after draw 17 rests on the sd of two
   points (0.65, 0.68 → sd≈0.021), a known edge pathology"; methodology adds the conditional-
   inference defect — "the change-point p=0.0032 is computed as if the test were unconditional,
   but it was run only because the watch-rule fired … simulate ≥10k stationary … ledgers …
   apply the *same* pipeline … report the conditional false-alarm rate." *Implementability:*
   scan-config edit + one simulation; also settles which CUSUM/h and split were pre-registered
   vs chosen this morning (methodology Q3 — "threshold-shopping" if unstated).

10. **[3/5 reviewers: methodology, prog-synthesis, rl-planning] Stop laundering the non-harm
    screen into efficacy language.** methodology: "Δlc = +0.152 with sd 0.537 over 16 paired
    items is t≈1.13, p≈0.27 … label these explicitly as non-significant"; prog-synthesis and
    rl-planning concur ("noise presented as signal"; "so it cannot leak into the panel's
    promotion prior"). *Implementability:* wording — restrict the screen's claim to "no large
    harm detected."

## AGENDA ITEM 1 DISPOSITION

**Panel majority: REJECT option (a) as a standalone remedy; RATIFY on design (b)
(interleaved contemporaneous controls, paired test), which may be preceded by (c) if the team
wants to sharpen the stationarity read first. Before any seal, the panel requires four
vote-blocking deliverables: (i) the pair-probability reproduction script (directive 4); (ii)
the yw8837 σ=0.24 variance-compatibility test / fork-diff (directive 3 / NC-13); (iii) the
power table for the chosen design (directive 2); and (iv) the harm-pause re-derived as a
paired criterion (directive 5).** This is a strict-majority disposition, not a bare plurality:

Explicit per-reviewer statements on the options:

- **llm-agents** — "Only option (b) (interleaved contemporaneous controls, paired test) is
  robust to the level ambiguity; the brief should not present (a) as a coequal standalone
  option, and R23 should either adopt (b)/(c)-then-(b) or explicitly re-derive the harm-pause
  threshold conditional on the stationarity re-check outcome." Adds the prior condition that
  if the gate has *never* waited >0s in-environment, the correct EV comparison is "4 draws of
  ~zero-effect A/B vs. 4 draws advancing the compaction lane."
- **methodology** — Does not name a single letter but its objections force (b)/(c): "rewrite
  §1 with the two scenarios separated," publish the 1.195 (not 1.1701) bar if (a) is kept,
  and "R23 cannot rationally choose among (a)/(b)/(c) without a power table." Prefers
  estimating σ "from our own contemporaneous data — which is exactly what option (b)/(c)
  would produce, and is another reason to prefer them." → effectively (b)/(c).
- **prog-synthesis** — "only the interleaved-control design (b) is defensible, and it needs
  specification before sealing"; "R23 should be presented options (b)/(c) only; (a) as
  derived should be struck."
- **rl-planning** — "The only designs robust to a mean shift are contemporaneous interleaved
  controls with a paired test (b) or holding for re-check (c). The brief should not present
  (a) as a standalone admissible option to R23; recommending it as such is a design error."
- **systems** — "Only option (b) … is robust to both mean shift and variance
  misspecification; R23 should mandate (b) and drop (a) as a standalone remedy."

**Disposition tally: 5/5 reject (a) standalone; 5/5 endorse (b) as the robust design;
(c) accepted by ≥3 (methodology, prog-synthesis, rl-planning, llm-agents) as an acceptable
precursor to (b) but not sufficient alone — prog-synthesis warns (c)'s "two extra fillers
move the post-break segment from n=2 to n=4, which has almost no power to separate step-down
from tail." No reviewer endorses (a) alone. Note on 1.1701 if (a) is nonetheless retained in
any hybrid: methodology's exact re-derivation says the correct (a)-bar under the sealed
formula with σ=0.24 is ≈1.195, not 1.1701 — "loose by ~0.025 under your own model."**

**Dissent / minority notes:** No reviewer dissents from rejecting (a)-standalone — the
disposition is unanimous on that point. The only intra-panel split is on *what must precede
the seal*: methodology, prog-synthesis, systems, and rl-planning treat the power table and
σ-compatibility test as hard vote-blockers; llm-agents and systems add a further, arguably
stronger prior condition — that the gate's causal mechanism be shown live at all (directive 6
/ NC-14) — under which even a perfectly-powered (b) is testing dead code. That condition, if
adopted, would defer the *first fire* regardless of design choice.

## AGENDA ITEMS 2–5 DISPOSITIONS

**Item 2 — A22 compaction eval plan ratification + Living-Harness payload amendment:**
DISPOSITION: ratify the A22 eval plan as already sealed/smoked (41/41), but **DO NOT reopen
the sealed prereg to fold in the Living-Harness graph-state payload as a same-day amendment.**
llm-agents (MINOR): "Amending a sealed prereg one day after sealing, on the strength of an
unreplicated paper summary and before any smoke of the graph-state payload variant, is
exactly the prereg churn the team's own governance exists to prevent. Either run the
amendment as a pre-specified secondary arm or defer it to the next A22 cycle; do not reopen
the seal for a citation." prog-synthesis (MINOR) adds the acceptance-metric gap: the
reframing has "no stated criterion for when the graph-state payload beats the plan-blob
payload — pre-register a measurable comparison (e.g., retained-index hit-rate or
recovery-success delta on the smoked 41-case set) before amending." Majority read: admit
Living-Harness only as a pre-specified secondary arm WITH a falsifiable acceptance metric, or
carry to the next A22 cycle. (Only two reviewers scoped this item; the other three placed it
under "what I cannot judge.")

**Item 3 — compaction-lane strategic endorsement:** DISPOSITION: ENDORSED, and stronger than
the brief frames it — the compaction lane is not merely "necessary regardless of A/B outcome"
but should not be sequenced strictly behind the A/B (see directive 7). rl-planning, systems,
and llm-agents all convert this endorsement into an allocation directive: the A/B ceiling
(~1.47) is already below the gold cutoff (1.54, +0.05/week), so the higher-ceiling arm must
get parallel or fixed-split slots, not the reserve behind a dominated arm. Endorse the lane;
reject the strict serialization.

**Item 4 — preflight hardening from host error list:** DISPOSITION: ADOPT (uncontested). No
reviewer objected; it is zero-cost and non-gating, and the underlying host post (Greg
Kamradt, "500 Submissions Analyzed") validates the fork-never-build + preflight discipline
the team already runs. Fold the 7-item list (silent stalls, forgotten GPU, unattached
datasets, `/kaggle/input` writes, `three.arcprize.org` calls) into `scripts/preflight.py` as
explicit gates. Cross-reference systems' MINOR on gate-timeout fail-open/fail-closed semantics
(180s) — the timeout path is a natural preflight/fault-injection check to add here.

**Item 5 — process-slip mitigation (2 sessions died on monitor waits):** DISPOSITION: ADOPT
(uncontested; placed under "what I cannot judge" by the technical reviewers, no objection).
Write the end-of-day log BEFORE long monitor waits and add the 17:00 backstop task. This is
also self-reinforcing with the panel's own throughput: the brief was produced on schedule and
the watch-rule executed cleanly, so the mitigation is codifying an already-working practice.

## NAMED CONDITIONS

New this round (continuing the R21→R22 sequence, last issued NC-12):

- **NC-13 (prog-synthesis, systems, rl-planning, methodology, llm-agents):** yw8837's σ≈0.24
  may not appear in any decision rule (promote bar, pair-probability exoneration) until it is
  shown exchangeable with our process. Discharge requires EITHER a variance-compatibility
  test on our own n=19 (s=0.159) — F-test / bootstrap CI / χ² — showing σ=0.24 is inside the
  95% CI, OR a yw8837 fork-diff + ledger-provenance memo equivalent to
  `fork_diff_boristown_2026-07-24.md`. If σ=0.24 is rejected on our data (prog-synthesis:
  p≈0.007), it is struck from the guard entirely and σ is estimated from our own
  contemporaneous draws (i.e., design (b)/(c)).
- **NC-14 (systems, llm-agents, prog-synthesis):** No boristown gated draw may fire until the
  readiness gate is shown to have a live causal mechanism in-environment. Discharge requires
  EITHER the historical in-environment vLLM ready-latency base rate (kernel-log grep showing
  the gate has ever waited >0s) OR a fault-injection leg (delay vLLM startup 60–150s) proving
  the gate holds requests and episodes complete, PLUS a specified fail-open/fail-closed
  timeout semantic (systems MINOR). If the gate is never observed to wait >0s, the A/B is
  re-scoped or the slots redirected to the compaction lane per directive 7.
- **NC-15 (methodology, rl-planning, prog-synthesis, systems):** The stationarity memo must
  publish (i) the reproduction script for the 0.38%/18.5% pair probabilities with exact event
  definition and assumed μ; (ii) the conditional (trigger→scan) false-alarm rate under a
  simulated stationary null; and (iii) the pre-registered CUSUM h and split-scan
  configuration (including a ≥3 minimum-segment-length constraint) so the n₂=2 |t|=8.64
  artifact is not re-litigated ad hoc. No stationarity verdict may drive a promotion decision
  while these are unstated.
- **NC-16 (all five, as a vote-gate for agenda item 1):** No gated draw may fire under any
  design until a power / MDE / draws-to-decision table for the chosen design is published
  (directive 2), and — if design (a) is retained in any form — the bar is corrected to the
  self-consistent value (methodology: ≈1.195, not 1.1701) and the post-hoc modification is
  sealed as a change-logged *modified* design before first fire (directive 8).

Carried-forward status of R22 conditions:
- **NC-9 through NC-12 (A17 rail):** DISCHARGED / MOOT. The A17 lane is formally closed
  (07-30, B2a: 72B route DEAD, seed-concordant format livelock; C4 discharged early), so the
  ρ_action metric-direction (NC-9), k≥2-seed kill rule (NC-10), and A17-promotion GPU parity
  (NC-12) conditions no longer gate an active promotion. **NC-12's GPU-parity principle is
  RE-ASSERTED and inherited by NC-14/systems**: systems' objection that RTX PRO 6000
  workstation cold-start timing "makes cold-start timing incomparable to Kaggle's allocation"
  is the same parity concern, now applied to the boristown readiness gate rather than the 72B
  route. NC-11 (sentinel un-shelve error rates) remains dormant with the sentinel arm.
- R21's NC-1–NC-8 carry forward unchanged.

## WHAT FIRES TODAY

**Read: the prereg CANNOT seal today as written, and the first gated draw should NOT fire
tonight.** Rationale under the process rules:

- The 07-27 restructure (§2) routes scored-window *promotion* decisions through sealed
  arithmetic gates rather than panels, and MAJOR-REVISION is advisory to the build rail. But
  boristown's own §7.3 makes R23 governance ratification a *precondition of the seal* for
  this specific A/B — so for this decision the panel's disposition is binding input, and the
  panel unanimously declines to ratify option (a) as sealed and unanimously requires design
  (b) plus four vote-blocking deliverables (NC-13, NC-15, NC-16, and the paired-harm
  re-derivation) that are not in hand as of the brief.
- Two fire conditions the brief lists as "governance-only" are therefore NOT both clearable
  today: §7.1 git-commit seal *could* be executed mechanically, but §7.3 ratification is
  withheld. The brief's own contradiction that rl-planning flags — "governance-only fire
  conditions / no evidence blockers left" versus "the sealed promote bar 1.0970 does NOT
  survive standalone" — resolves in the reviewers' favor: the invalid bar is an evidence
  blocker, so "no evidence blockers left" is false and the entry-gate discharge does not by
  itself license a fire.

**Under which amendment could it seal, and when could the first draw fire:**
- The seal can complete on a **design-(b) amendment** (interleaved contemporaneous controls,
  paired test, harm defined relative to the contemporaneous filler, sealed change-log) once
  the four blockers are discharged: pair-probability script (NC-15), σ-compatibility test or
  yw8837 fork-diff (NC-13), the power table for (b) (NC-16), and the paired harm-pause spec.
  Three of these are fast (script, χ²/F-test, power table are hours of work); the yw8837
  fork-diff, if that route is taken instead of the variance test, is the long pole.
- **First gated draw:** even a clean design-(b) seal does not license a fire tonight if the
  ≥3-reviewer NC-14 (live-gate-mechanism) condition is treated as blocking — llm-agents and
  systems argue the gate may be inert (`latency=0.0`), in which case the correct move is to
  redirect the slot to the compaction lane, not fire the A/B. Minimum realistic path to a
  first gated draw: NC-13 + NC-15 + NC-16 discharged AND at least the kernel-log latency grep
  (cheap leg of NC-14) showing the gate has ever waited >0s. If that grep comes back all-zero,
  the panel's directive 7 (parallel/redirect to compaction) supersedes and no gated draw
  fires at all.

**Bottom line:** today's actionable output is (1) log + backstop process fix and preflight
hardening (items 4–5, adopt now), (2) endorse the compaction lane and un-serialize it (item
3), (3) do NOT seal option (a); prepare the design-(b) amendment plus the four deliverables,
and (4) run the cheap NC-14 latency grep to decide whether the boristown arm is worth any
slots before building the fault-injection leg. The A/B seal and first fire slip to the
earliest day all blockers clear — not tonight.
