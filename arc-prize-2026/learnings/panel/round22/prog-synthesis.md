## Summary (2 sentences)
The revision materially improves measurement hygiene — the canonical ledger now covers every cited draw, the A17 gates are at least summarized inline, and the sentinel/boristown decisions are framed as explicit options — but the two load-bearing falsifiability items I demanded (a numeric ρ_action kill threshold and a ρ_action→LB mapping) remain undischarged *while the experiment they govern is already in flight*, which is pre-registration in name only. Worse, the ρ_action metric as written is directionally ambiguous (the stated definition and the stated kill-direction contradict each other), so even a panel-ratified Y would currently be unfalsifiable.

## Objections

**Resolution of prior-round objections (reviewed first, per panel rules):**

[MAJOR → PARTIALLY-RESOLVED] No falsifiable prediction attached to A17 — G1–G4 now appear in abbreviated form (G1 recovery ≥ 0.95, G2 ≥ 100 executed actions, G3 cadence, G4 no interpretation), which satisfies half my fix. But NC-5 (numeric kill threshold Y) and the ρ_action→expected-LB mapping are explicitly "NOT yet discharged," and v6 was pushed anyway. The brief converts my demand into open question 1 and asks the panel to supply the missing arithmetic — that is outsourcing, not resolution. Mitigant honestly stated: v6 consumes no scored slot.

[MAJOR → PARTIALLY-RESOLVED] Gold-capable arms closed while filler burns slots — The boristown readiness-gate A/B is now on the table (R21 directive #1, 5/5) with a proposed prereg shape (n=3–5, one-sided at the 1.47-anchor effect size), and the brief concedes P(frozen ≥ 1.49) ≈ 2×10⁻⁴. But it is *still unscheduled*, and a fourteenth filler draw (0.90, 07-28) was burned overnight. Acknowledged ≠ executed.

[MAJOR → PARTIALLY-RESOLVED] Sentinel shelved at n=1 — Question 2 now contains exactly the un-shelve rule structure I asked for (draw #2 after v6, n ≥ 4–5, re-shelve on 2 consecutive < 0.80 or mean of first 3 < 0.80). But I asked for "a date, not a deferral," and the disposition is again handed back as OPEN. One more round of deferral makes this UNRESOLVED.

[MAJOR → PARTIALLY-RESOLVED] Non-canonical ledger / mid-week pushes escaping review — The ledger half is fully RESOLVED: `runs/lb_ground_truth.md` was refreshed 07-28 from the live API before this panel, n=14 stats recompute cleanly, and the 07-26/07-27 draws are now canonical. The governance half is UNRESOLVED: my named escalation condition ("first push of a new artifact version triggers ad-hoc panel review") was not adopted; instead v6 — a new artifact version — fired mid-week under user order with the panel demoted to advisory, i.e., precisely the scenario I flagged occurred.

[MINOR → PARTIALLY-RESOLVED] Band as min/max — z-scores against prior-n stats are now reported per draw (good), but the operative criterion is still "interior to band 0.82–1.33." No tolerance interval, no stated trigger for an out-of-interval draw. The band still widens monotonically and can never be falsified.

[MINOR → UNRESOLVED (non-blocking)] Three-way convergence framing — The war-v4/skill-memory material is simply absent from this brief; the convergence narrative was dropped rather than corrected, and no falsifiable war-v4 criteria (skill-reuse rate, per-skill verification pass rate, held-out delta) have been shown. Carries forward to whenever war-v4 resurfaces.

**New objections:**

[MAJOR] ρ_action is directionally ambiguous, making any kill threshold meaningless as written — §2 defines v6 as delivering "the ρ_action denominator (480 / Σ N₇₂B)": if ρ_action = 480/ΣN₇₂B, then ρ *decreases* as the 72B executes more actions, so the stated kill rule "ρ_action < Y ⇒ route dead" (question 1) kills the route for *good* throughput. Either ρ_action = ΣN₇₂B/480 and the parenthetical is wrong, or the kill inequality is inverted. A pre-registered threshold on a metric whose sign convention the document contradicts is not falsifiable. Fix before the sealed walk reads v6: state ρ_action's formula, direction ("higher = better"), the value Y with its inequality, and the frozen-baseline derivation of 480, in one sentence.

[MAJOR] Pre-registration after launch requires a commitment mechanism, and none is specified — v6 is already running; "name Y before the sealed walk reads v6" only counts as pre-observation if the seal is enforceable, and the same principal operates the sealed walk, the kernel pulls, and the brief. Nothing prevents (even unintentionally) glancing at the kernel log before Y is fixed. Fix: hash-commit the threshold document (Y, mapping, G2-fail branch) and record the commit timestamp *before* the v6 kernel status flips to COMPLETE, or have another panel member hold the pull.

[MAJOR] The v5 slice data may already predict a G2 failure, and no G2-fail branch is pre-registered — The 1500 s slice reports N(ft09)=2, N(sb26)=1, N(lp85)=0, N(vc33)=2 (Σ=5). The brief never says whether N counts *executed actions* or *errors* (the citation is `a17_error_model.md`, suggesting errors, but §2's ρ_action usage suggests actions). If N is actions, linear extrapolation gives ~26 actions over 7920 s — a ~4× miss on G2's ≥100 — and the plan should pre-register that branch now (slots revert to which of frozen / gated-A/B / sentinel, in what order?) rather than deliberate post hoc. This is throughput arithmetic, not capability interpretation, so G4 does not shield it. Fix: define N, and write the G2-fail contingency into the same hash-committed document as Y.

[MINOR] Question 3's framed trade-off is false, and the ranking is determinable now — Boristown A/B vs sentinel draw #2 "compete for the same filler slots" only because daily frozen filler draws continue; ending filler (which by the brief's own 2×10⁻⁴ figure cannot climb) funds both. Within my competence to say: the readiness-gate A/B should rank first — it has an external 1.47 anchor, a one-line diffed mechanism, and at s ≈ 0.138 a one-sided n=3 test detects the anchor-implied ~+0.5 effect with power ≈ 1; the sentinel's hypothesized effect is unquantified. The prereg should also state the *minimal* detectable effect at n=3–5 (~+0.25 at 80% power), so a null result is interpretable.

## Questions for the authors (numbered)
1. State ρ_action's exact formula and direction, and confirm which side of Y is "dead." Where does the constant 480 come from — measured frozen-fork median actions per 7920 s window, or a design target?
2. Are the v5 slice N-counts executed actions or parse/tool-call errors per `a17_error_model.md`? If actions, do you dispute the ~26-action full-window extrapolation, and on what mechanism (e.g., warmup-dominated slice)?
3. What is the commitment mechanism ensuring Y and the expected-LB mapping are fixed before anyone observes v6 output? Hash, timestamp, holder?
4. Commit to a date: sentinel draw #2 and boristown A/B draw #1 — which calendar slots, under which prereg files?
5. For question 4 (EWM Stage-1): if the A17 rail occupies build slots through ~Aug 3 and the window closes Aug 3, isn't the honest answer that Stage-1 is already dead for this window? What is the re-priced date?

## What I cannot judge
Kaggle kernel infrastructure specifics (whether the dataset-mount route, AWQ shard layout, and Blackwell serving claims are as reported); the competition-legal status of dataset-hosted 72B weights; the accuracy of the live-API leaderboard pulls (I take the canonical file on trust per panel instruction); and the internal eval-rail evidence cited by the sentinel disposition memo, which I have not seen.

## Verdict: MAJOR-REVISION

## Score: 5/10