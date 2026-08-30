## Summary (2 sentences)

The proposal correctly identifies that the day's only new asset is an instrument (`bench`) rather than a mechanism, and its three handoff decisions (run bench, don't redraw, fix the ledger) are directionally sound and cheap. However, the bench instrument is accepted on the artifact author's own verbal warrant of control validity, with no pre-registered kill threshold, no treatment of a visible token-budget confound between arms, and a ledger-correction step that would write an unverified config identity into the very record the selection rule depends on — all correctable, none yet correct.

## Objections

**[MAJOR] The "placebo control" claim is confounded by the binding token budget, and the proposal's own find #7 proves it.** The validity of arm B rests on the verbatim [V-doc] claim that suppressing graft *text* changes "nothing else." But find #7 establishes — from the same kernel — that every archived game ends on the 7920s wall-clock, i.e. a fixed ~240k generated-token budget, with 97.6–98.3% of tokens spent on reasoning. Graft text in the prompt consumes context and steers generation; suppressing it changes how the binding budget is allocated between arms. So A vs B differs in *at least two* things: information content AND effective compute-per-action under a saturated cap. A B-wins result could mean "grafts hurt" or "grafts cost budget" — different stop decisions. The proposal must state this confound and pre-register which reading a B-win licenses. Same code path ≠ same computational budget when the budget binds.

**[MAJOR] "Kill instrument, only on a large gap" is not a decision rule — no threshold, no noise yardstick, no VOID condition.** The proposal pre-registers the *category* of read but not the read itself. What numeric gap between arm means kills the TV28 arm? The rig itself provides the yardstick: A0-vs-A1 and B0-vs-B1 are within-arm replicate pairs, so the pre-registration should be of the form "kill only if |mean(A)−mean(B)| exceeds k× the pooled within-arm spread; VOID if within-arm spread exceeds the between-arm gap." Given the 1.82/0.00 byte-identical precedent, the false-kill probability at n=2/arm is nontrivial and should be stated before data lands. Without this, "do not let it promote anything" is enforced but "do not let it kill on noise" is not — and a wrong kill throws away a config permanently on a coin flip.

**[MAJOR] Handoff #3 corrects the ledger to "13 grafts" while the proposal's own Gaps section says that claim is [UNK] on the run.** `install()` is blanket-guarded and silently falls back to stock on any error; until the `TAAF_GRAFTS FEATURES={...}` banner is pulled from the 00:19 run log, the 1.62 draw's config identity is unknown — it could be 13 grafts or zero. Writing "13 grafts" into the ledger before that check inverts the order of operations and commits exactly the corruption the section warns against. Sequence must be: pull banner → then write whatever the banner says → and if the banner is absent, the 1.62 has *no* config identity and belongs in no config mean at all.

**[MAJOR] The bench run tonight would itself mount the unpinned, author-mutable slug — pinning is scheduled for the wrong milestone.** Handoff #2 requires version-pinning only "if the fork is redrawn," but handoff #1 runs bench on `canivel/arc3-tv28-fork` tonight against a slug the author republished 11½ hours before the last draw and can republish again mid-session. An A/B on a config with no fixed identity produces a verdict attached to nothing: the kill (or non-kill) cannot be bound to the config the ledger tracks. Pin the dataset version *before* the bench session, not before the next draw.

**[MAJOR] A kill verdict from n=2/arm on ONE hand-picked game does not transfer to the 4-game submission distribution, and the proposal licenses exactly that transfer.** BENCH_GAME=m0r0 was chosen by Tennant, presumably for cheapness or graft-sensitivity — the selection criterion is unstated and matters. Per-game score mechanics (find #7: cap binds, efficiency saturates) can make graft text score-inert on m0r0 while it matters on games where level-completion is marginal. The honest pre-registration is: "bench can kill the stack *on m0r0*; a stop decision for the submission config additionally requires the assumption that m0r0 is representative, which is untested." State the assumption or weaken the licensed conclusion.

**[MINOR] σ-based pricing (+2.01σ draw, +4.27σ to top-10) presumes an approximately Gaussian draw distribution that the document's own evidence contradicts.** The 1.82/0.00 byte-identical pair and our 0.41–2.05 range suggest heavy-tailed or bimodal draws; n=8 sd estimates are unstable and z-scores computed from them overstate precision. Report gaps in raw score units alongside, or use a rank/quantile framing.

**[MINOR] "Zero draws" is not "zero cost."** One ~9h GPU session comes out of a finite weekly quota that also feeds any redraw pipeline; the handoff should state what the session displaces so the "free comparator" framing stays honest.

## Questions for the authors

1. What is the pre-registered numeric kill threshold for bench, expressed relative to the within-arm (A0–A1, B0–B1) spread, and what result VOIDs the instrument itself?
2. If arm B outscores arm A, which hypothesis do you record: "graft information hurts" or "graft text consumes the binding token budget"? How would you distinguish them (e.g., compare generated-token counts and action counts per arm from the artifacts)?
3. Will the `TAAF_GRAFTS` runtime banner be pulled from the 00:19 log *before* the ledger row for 1.62 is written, and what is the ledger entry if the banner shows a fallback-to-stock?
4. Will `kernel-metadata.json` pin the dataset version before tonight's bench session (not merely before a redraw)? Which version — 08-28 v28 or the 08-29 12:51 republish — and why?
5. What is known about why Tennant chose m0r0 as BENCH_GAME, and is there any evidence it is graft-sensitive rather than cap-bound like tn36?
6. Handoff #3 notes `runs/ledger.json` is ten days stale with promotion bar 1.089 computed on the old distribution — is the bar re-derivation blocking for any decision tonight, and who owns it?
7. Does the `not TRUE_SUBMISSION` guard verification (per `feedback_audit_the_instrument`) have a concrete check — e.g., grep of the effective flag path in the pushed notebook — or is it a read-and-nod?

## What I cannot judge

Kaggle API operational details (rate-limit behavior, CLI version quirks, heartbeat tooling); the sociological "ladder hypothesis" and the identity/method inferences about Youssef Nader, Liao Zixu, and other teams; GPU quota economics on this specific account; and the accuracy of the board-scrape numbers themselves, which I take as canonical per the panel brief.

## Verdict: MAJOR-REVISION

## Score: 6/10