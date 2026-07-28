## Summary (2 sentences)
The revision fixes the ledger provenance discipline and is admirably honest about which named conditions it violated (v6 pushed mid-week on a boot-PASS with NC-4/NC-5 undischarged, under user order), but the two highest-value actions — the boristown readiness-gate A/B (a 5/5 panel directive) and a committed, quantified gap-closing plan — remain unscheduled and are again punted to the panel as open questions. Worse, the only 72B throughput data in hand (the v5 1500 s slice: 5 actions across 4 games) extrapolates to ~4× below the v6 G2 gate, and the brief presents it without flagging that the 72B route is already on track to fail its own pre-registration.

## Objections

**Prior-objection resolution status:**

**[MAJOR, prior] No quantified gap-closing plan — PARTIALLY-RESOLVED.** The brief now explicitly concedes filler cannot climb (P(≥1.49) ≈ 2×10⁻⁴, directive #5 acknowledged) and enumerates candidate lift channels (72B route, gated A/B, sentinel). But enumeration in an "open questions" section is not a plan: nothing has a scheduled date, an expected Δscore, or an owner. Three days after a 5/5 MAJOR-REVISION on exactly this point, the strategy is still a monitoring loop with a menu attached.

**[MAJOR, prior] Boristown readiness-gate A/B unclaimed — PARTIALLY-RESOLVED, and the residual is now the worst item in the document.** The diff is named, the directive was 5/5, and the brief admits "still unscheduled." This is a fork-not-build change with a 1.47 public anchor, competing for filler slots that the brief itself certifies carry zero information. There is no stated reason it did not start on 07-25; every day of filler since the R21 directive is a slot spent against the panel's unanimous instruction.

**[MAJOR, prior] Push gated on a boot canary under reduced oversight — PARTIALLY-RESOLVED.** The condition was overridden ("explicit user order, panel advisory"), which the panel cannot un-ring, but the blast radius is genuinely bounded: v6 consumes free kernel GPU-hours, not a scored draw, NC-3 machinery caps the burn, and the brief correctly asks for ρ_action threshold Y to be sealed *before* v6 numbers are read. The residual defect: NC-5 is still un-named, and the brief inverts the burden by asking the panel to invent Y (see new objection below).

**[MAJOR, prior] Fenced-recovery adapter unmeasured — UNRESOLVED.** NC-4 (≥200-replay offline parse study) is admitted undischarged. The canary evidence offered instead is n=1 anecdote in both directions: one successful roundtrip, one recovered fenced call ("hits=1"). The v6 G1/G2 gates (recovery ≥ 0.95 over ≥ 100 actions) *would* deliver equivalent data — but only if v6's output reports raw parse-attempt/failure/recovery counts, which the prereg as quoted does not guarantee, and only if v6 executes ≥100 actions at all, which the throughput data below makes doubtful. Until then, the adapter remains prompt-it-better hand-waving with a sample size of one.

**[MINOR, prior] Ledger provenance for the 0.84 draw — RESOLVED.** `runs/lb_ground_truth.md` refreshed 07-28 from live API; 07-26 and 07-27 draws cross-checked against `submission_log.jsonl`; n=14 arithmetic verifies (13.56/14 = 0.9686 ✓). This is how it should be done.

**[MINOR, prior] Filler slot economics — PARTIALLY-RESOLVED.** The brief now concedes filler holds rank without climbing and frames the sentinel-vs-A/B slot competition, but still never states whether slots bank or expire, and the 07-28 slot went to filler anyway.

**New objections:**

**[MAJOR] The v5 throughput slice already forecasts G2 failure, and the brief hides this behind "MEASUREMENT ONLY."** The 1500 s slice shows 5 total actions across 4 games (N=2,1,0,2), i.e., roughly one action per 230–300 s after the 337 s boot. Linearly extrapolated to the 7920 s window that is ~25–35 actions — a factor of ~3–4 below G2's ≥ 100 executed actions. Refusing to *interpret* at k=1 is defensible doctrine; refusing to *state the arithmetic implication for your own pre-registered gate* is not — the panel should walk into the v6 read expecting a G2 FAIL and with the reversion branch (slots → gated A/B) pre-committed, not deliberating post hoc. Additionally, N(lp85)=0 — zero actions on one game in 25 minutes — is a distinct dead-path signature (stuck episode loop or per-game init failure) that deserves its own grep in the v6 log, not aggregation into a throughput average.

**[MAJOR] Asking the panel to name Y is burden inversion; here is a proposal to remove the excuse.** The authors own the harness and the frozen fork's per-window action counts; the panel does not. Concrete proposal to be ratified or amended before the sealed walk reads v6: **kill the 72B route if executed actions < 100 in the 7920 s window (G2, already registered) OR ρ_action implies < 50% of the frozen fork's measured per-window action count**, with the expected-LB mapping supplied by the authors as: score is monotone in valid actions per episode (their own 0.82–1.05 mass is attributed to wasted early actions), so a 72B at half the frozen action rate has no mechanism to beat 0.97, let alone 1.49. If the authors have the frozen fork's action-count denominator, publishing it is a one-line addition; if they don't, that is itself a gap — they have been running a "throughput canary" program without a baseline throughput number.

**[MINOR] The tool-call roundtrip evidence contains a schema smell.** `args={"action": "submit_action", "x": 3, "y": 7}` duplicates the function name *inside* the argument object. Either the harness schema genuinely requires an `action` field (fine — say so) or the hermes parser is silently tolerating malformed output that the game API may reject downstream. One grep against the actual game-API call site settles it; do not let an n=1 "roundtrip=OK" paper over a latent contract mismatch.

**[MINOR] "Push slot 1 of 2" — slot 2 is never accounted for.** State what today's second push slot was spent on (or that it lapsed). Unaccounted push capacity in a brief that litigates slot economics elsewhere is an audit gap.

## Questions for the authors (numbered)
1. What is the frozen fork's per-window executed-action count (the ρ_action baseline denominator)? If unmeasured, why has a throughput program run for 6 versions without it?
2. If the v5 slice rate holds and v6 fails G2, what — precisely and today, before the read — happens to the A17 build slots and the Aug 3 EWM collision?
3. Why has the boristown A/B not started, given a 5/5 directive, zero-information filler slots, and a fork-not-build implementation? Name the blocking constraint or start it.
4. Will the v6 output include raw parse-attempt / parse-failure / fenced-recovery counts (not just G1 pass/fail), so NC-4 can be discharged from bench data without a scored draw?
5. Does the harness schema require the `action` key inside tool-call args, and does the game API accept the exact object the hermes parser emitted at t=345.8?

## What I cannot judge
The statistical machinery (MK/CUSUM no-trend verdicts, the t-predictive p ≈ 0.07 on the sentinel draw, sequential stopping boundaries in open question 2) — that is reviewer territory for the panel statistician; I take the n=14 arithmetic at face value since it verifies. Likewise Kaggle account/quota mechanics (whether scored slots bank), the EWM Stage-1 latent-state audit content, and the provenance claims I cannot re-derive from the quoted text (fork-diff byte-identity, kernel pull logs).

## Verdict: MAJOR-REVISION

## Score: 4/10