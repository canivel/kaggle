## Summary (2 sentences)
The R16 republication is the first circulation in this campaign whose core inferential machinery survives arithmetic check: the α re-derivation, the SE-based guard, the dismantle-branch calibration, and the explicit binomial sketch all do what my prior round demanded, and I verified their numbers (σ̂=0.189, SE=0.154, z-boundary −0.28, familywise 0.097, all critical values in the §3R table). Two load-bearing statistical defects remain — the A17 screen's unquantified false-NO-GO risk under an asymmetric max comparator, and an engine-version-drift confound in the frozen n=4 control band that the team's own audit documents — both of which must be fixed before the A14 seal and the A17 sign-off respectively.

## Objections

**Prior-objection resolution status (required first):**

**[Prior FATAL] Sign-flip prong calibration — RESOLVED.** §5R publishes all four binomial-sketch assumptions explicitly (game as unit, uniform §4R ranges, 0–2 spurious pairs, independence with the adverse direction stated), and with α=0.05 the pass region is arithmetically reachable (5/5 at p=0.031). The internally inconsistent "≈0.05 at 4 positives" figure is gone; B− is now honestly labeled a near-certain FAIL rather than dressed up. Residual concerns move to R4/R5 below.

**[Prior MAJOR] Banking winner's curse — PARTIALLY-RESOLVED (unchanged).** The A16 shrinkage recompute is still promised, not delivered; the architecture is acceptable because W3 cannot open without it, but §5R's B+ P(pass)≈0.2 is computed on the pre-haircut +0.03–0.08 band and must be marked provisional, with a commitment to republish P(pass|B+) after the haircut.

**[Prior PARTIAL] 0.56× conversion / uncalibrated mechanism threshold — RESOLVED.** A20's 0.4–0.8 band is carried through §2R, and the (a) mechanism prong is now the deterministic O5 predicate (fired-before-every-death, 49/0), replacing the same-transcript n=3 "deaths halved" criterion I objected to.

**[N1] Stale alternative at the cumulative gate — RESOLVED.** §2R/§4R/§5R recompute the stack with (d) excluded and banking carried as an explicit B+/B− branch; both branches' expectations, ceilings, and P(pass) are published, with the B− near-certain-FAIL admission made in plain text.

**[N2] α multiplicity rationale — RESOLVED.** §3R names the test family, correctly argues that a conjunctive (AND) decision rule needs no Bonferroni (size ≤ primary's size), and seals α=0.05 one-sided; every entry in the critical-value table checks out (7/8 = 9/256 = 0.035; 8/9 = 10/512 = 0.0195; 9/10 = 11/1024 = 0.0107).

**[N3] Guard false-kill probability — RESOLVED.** §8 publishes the old guard's false-kill (0.26/window, 0.59 familywise — confirming my suspicion), replaces the point threshold with the SE-based boundary z=1.834 achieving familywise 0.097 ≤ 0.10, and adds the honest MDE statement (a true −0.20 trips at only ~30%). Arithmetic verified. The df=2 fragility of σ̂ becomes new objection R3.

**[N4] A17 single-noisy-sample GO/NO-GO — PARTIALLY-RESOLVED.** The comparator is now sealed (per-game MAX over the certified 27B seeds, Σ=6 frozen), a marginal-seed rule exists, and the gate boolean/null_adj walk is fully pre-specified — real repairs. But max-on-both-sides with 3–4 draws on the 27B side versus 1–2 on the 72B side is not distributionally symmetric: the 27B max harvests its lucky tail (ft09 spans 0–2 across seeds) while the 72B side gets one draw, and this bias stacks with the +2 margin. See R1.

**[N5] Budget regime of the binding look — RESOLVED.** §6.1 states FULL budget explicitly and correctly discharges A15's confirmation-replicate requirement by construction (3 full-budget seeds ≥ 1).

**[N6] Post-seal edit channel — PARTIALLY-RESOLVED.** §13 adopts exactly the standing procedure I demanded (thresholds in hash-committed files under `runs/sealed/` before measurement; append-only results). But this very circulation re-instantiates the pattern: §10's 0.99 bar is claimed "sealed before the audit runs" while Part 4 delivers the completed audit in the same circulation and the protocol was "drafted in parallel today" — precedence is again narrative-attested only. Harm is nil here (the audit is diagnostic, no score claim), but the fix must be evidenced by a timestamped hash in the panel record, not asserted.

**[Prior MAJORs from earlier rounds] Truncated circulation, compressed-bench regime — remain RESOLVED** (document ends with the literal END line; A15 rule intact and now concretely discharged by §6.1).

**New objections:**

**[MAJOR] R1: A17's false-NO-GO probability is unquantified, and NO-GO is campaign-terminal for the only wall-closer.** Under the sealed rule, one (possibly two) 72B draws must beat the max of 3–4 27B draws by +2 summed levels; even a genuinely better 72B has substantial probability of failing this on seed luck alone, and §8.1 then closes war-v4 permanently with §8.2 prohibiting re-screens. §4.2 asks the panel not to demand α — I accept no p-value is needed on an existence screen, but I do not accept an unpublished error rate on an irreversible decision. Fix before sign-off: bootstrap the frozen 27B per-game per-seed lc distributions to publish P(NO-GO) under (i) the null 72B≡27B and (ii) a true +1-level-per-game shift, for 1 and 2 72B seeds; alternatively equalize draws (compare max over equal seed counts). This is a one-day computation on data already in hand.

**[MAJOR] R2: Engine-version drift confounds the frozen control band before the binding look.** §11 pools war_eval v1–v3 (weeks old by the Jul 28–Aug 3 look) as controls, but Part 4's own audit documents engine-version drift within the corpus (cn04-65d47d14 vs cn04-2fe56bfb; ka59-9f096b4a UNRESOLVED in the older era) — proof that game engines get updated between pulls. Any engine change between the control pulls and the ON pulls loads onto the paired per-game Δlc as exactly the integer-level signal the sign-flip test detects, in either direction, invisible to the test's size guarantee (which assumes exchangeability within pairs). Fix before the A14 seal: require versioned-game-id identity between each control-band game and its binding-look counterpart, with a sealed rule that a drifted game's pair is DROPPED (not counted either way) and the fallback W0 seeds (§11) triggered if drops exceed a stated count.

**[MINOR] R3: All guard/dismantle operating characteristics rest on a df=2 variance estimate.** With σ̂=0.189 at df=2, the χ² interval on σ is roughly [0.10, 0.6+]; the dismantle trip rate quoted as 0.24 ranges ~0.10–0.33 over σ∈[0.10, 0.30]. The small-df caveat is stated, but publish the sensitivity band for both the §8 guard and the §6.2 dismantle alongside the frozen points, so the panel ratifies a range, not a false-precision scalar.

**[MINOR] R4: Assumption (iii) — "0–2 spurious nonzero pairs" — is in tension with the stated 8–12 games carrying cross-seed lc variance.** A game with cross-seed variance will generically produce a nonzero 3-ON-mean-vs-4-control-mean diff; 4–6 spurious pairs is at least as plausible as 0–2. Test SIZE is unaffected (the exact sign test conditions on n and signs are symmetric under H0), but P(pass|H1) falls as random-sign pairs dilute the count — so the published P(pass) may still be optimistic. Publish the sketch at 4 and 6 spurious pairs.

**[MINOR] R5: The B− binding look has power ≈ size (P(pass|H1) ≈ 0.02–0.10 vs α=0.05), so a B− PASS carries likelihood ratio ≈ 1 and is nearly uninformative.** The authors concede the near-certain FAIL but do not seal what a PASS would mean; under B−, a lucky PASS must not be reported as confirmation. Seal a reporting rule: any B− PASS is cited with its realized p and an LR computed under the sealed sketch, and does not upgrade the stack's evidentiary label on its own.

**[MINOR] R6: SENTINEL_BUDGET=150 rests on an unverified assumption that the scored-regime per-game token envelope equals the build rail's (§12 says "assumption; verifiable by grep").** Since the entire warning ladder mis-scales if the envelope differs, make the tokens/game grep on a scored-run pull a sealed pre-ship check with a stated tolerance (e.g., envelope within ±15% of 63k, else re-derive B by the frozen formula).

## Questions for the authors (numbered)
1. When does the A16 shrinkage recompute run, and will §5R's B+ P(pass) be republished post-haircut before W3 can open?
2. Are the engines behind the war_eval v1–v3 control pulls byte-identical (versioned game ids) to those the Jul-28 binding-look harness will serve? If unknown, why is this not a sealed precondition?
3. How many 72B seeds does the base A17 plan actually fund — §2.3 says "≥1"; is the modal plan n=1? What is P(false NO-GO) under the frozen 27B seed spreads?
4. What is the empirical basis for "0–2 spurious nonzero pairs" given 8–12 games with cross-seed lc variance — has the null pair-count been computed from war_eval v1 vs v2/v3 splits (data in hand)?
5. Can you produce a timestamp-verifiable hash commitment (panel record, not self-narrative) showing the §10 0.99 bar preceded the first `latent_state_audit.py` run?

## What I cannot judge
The sha256 values and file provenance; the existence/properties of the Kaggle 72B-AWQ artifact and the VL-modality claims; vLLM serve-config correctness (hermes parser, thinking flags); GPU-hour and wallclock estimates; game-domain semantics (which games are "grinders," lp85's 60-click mechanic); the relevance and quality of the cited arXiv papers; and whether the latent-state audit's implementation matches its protocol (I reviewed the protocol's statistical design — support guard and selftests are sound — not the code).

## Verdict: MAJOR-REVISION

## Score: 7/10

The inferential core of the A14 gate is, for the first time, statistically sound and honestly calibrated — a genuine repair, and §8/§3R/§5R are close to model pre-registration practice. But the seal cannot proceed with an unverified engine-drift confound sitting inside the control band (R2), and A17 — an irreversible campaign decision — must not sign off with its dominant error mode (false NO-GO) unquantified (R1). Both fixes are cheap, computable from data already on disk, and must land in the sealing text, not a follow-up memo.