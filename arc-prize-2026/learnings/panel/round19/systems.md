## Summary (2 sentences)

The A21 governance machinery demonstrably works in production (harm-pause fired at 0.71 pre-loop, exactly as sealed), the ρ_action screen design has moved to measured end-to-end action rates as I demanded, and the SKU claim now has two independent external corroborations — but the team is pushing the A17 canary **today** without having attached the first-party hardware artifact or quota-ledger evidence that were the explicit preconditions of my round-2 objections. Worse, the headline-number contradiction (proposal 1.33/1.86 vs briefing 0.43/1.56) enters its third round without a single reconciling sentence, and the new ρ_action construction has a concurrency confound (4-game denominator vs 25-game numerator) that biases the screen toward a false GO.

## Objections

**[Prior FATAL 1 — quota is not free] PARTIALLY-RESOLVED (no change; precondition violated).** The SKU half improved indirectly: two independent external sources (#684625 Scott Le Grand reproducing the vLLM hang on RTX Pro 6000 with the duck notebook; lastloop-ai's vllm-blackwell-guide for sm_120) now corroborate the Blackwell rail. But the demanded first-party artifact (`nvidia-smi` / `get_device_properties` output from the team's own kernel log, with path) is still unattached, and the quota-exemption footnote for scored windows has **zero** new evidence — and Q5(iii) ("is a public fork a 'filler' (no window cost) or an arm?") reveals the team itself does not know the answer. The A17 canary is scheduled for slot 1 today *before* this evidence exists, which inverts the pre-registration ordering I required.

**[Prior FATAL 2 — A17 envelope] PARTIALLY-RESOLVED.** The parity prong is now denominated on measured end-to-end action throughput (ρ_action with 27B numerator frozen at 480 actions/7920s), which is exactly the fix I asked for, and NO-GO-at-modest-lift is stated as the designed outcome. But the numeric GO/NO-GO threshold on ρ_action and the derived turns-per-level floor live in unattached references (C3, `a17_error_model`) — this brief does not let me verify the arithmetic. See also the new concurrency-confound objection below, which attacks the ρ_action construction itself.

**[Prior MAJOR — watchdog kills the bench] RESOLVED.** Remains resolved; the canary's log-heartbeat observable implements my cold-start note. No regression.

**[FATAL, escalated from Prior MAJOR — headline numbers contradict the briefing] UNRESOLVED, third consecutive round.** The brief asserts best 1.33 / KOJIMA 1.86 / gold ≈1.49 / rank #49; the panel briefing states best 0.43 / leader 1.56. Not one line reconciles them, despite this being an explicit condition of approval last round. Every economic quantity in today's brief — the +0.14 fork delta, the "4 ranks/day bleed," the +0.19–0.29 depth-event repricing, the E[max] currency of the A21 ledger — is denominated in the unreconciled metric. I escalate to FATAL: a strategy document whose objective function cannot be tied to the ground-truth leaderboard is unactionable, and repeated silence suggests either a stale artifact or a normalization the team has not audited. One paragraph with the formula or the artifact resolves this; its continued absence is itself the finding.

**[Prior MAJOR — RC4/R5 pricing contradiction] RESOLVED.** Better than resolved: the 0.71 draw is the mechanism's first live firing — pause executed pre-loop, no n=1 inference claimed, cost booked against the 12-window ledger. This is the strongest section of the brief.

**[Prior MINOR — no tail model] PARTIALLY-RESOLVED (no change).** The 0.71 analysis reuses the implicit-Gaussian z-machinery (z=−1.70/−1.75, one-sided p≈0.04) on n=10/15 with no stated fit family or CI. Not load-bearing today (three aligned negatives across two rails is a robust qualitative signal), but the requested fit statement and CI have now been ignored twice.

**[Prior MAJOR — single-point hardware claim] PARTIALLY-RESOLVED.** Merged with FATAL 1's SKU half above: external corroboration is real progress and the canary's "self-certifying envelope" will settle the fit half empirically. But the pre-registration demand — attach the scored-rail and bench-rail device artifacts *before* quota is spent — is being violated by today's slot-1 push. This costs ten minutes; do it before the push, not after.

**[Prior MINOR — contingency line] UNRESOLVED (not addressed).** This brief contains no budget table, so I cannot verify the re-lining (contingency ≥ 1× largest run, setup itemized). Carried forward; note the "full-window" canary description still does not itemize the 43GB download/load/warmup, which directly affects whether the denominator window is 7920s of serving or 7920s minus 45–90 min of setup.

**[MAJOR — new] ρ_action is confounded by concurrency regime; the bias direction produces false GOs.** The 27B numerator (480 actions/7920s) was measured from certified 25-game runs; the 72B denominator will be measured at 4-game concurrency. Under vLLM continuous batching, per-game action latency *improves* at lower concurrency (less queueing, more KV headroom per stream), so the 72B's per-game action rate at 4-way is flattered relative to its rate at deployment concurrency — ρ_action will overestimate 72B parity, and the screen's parity prong can pass a model that is out of envelope at real load. Fix before interpreting the canary: either (a) add a 27B 4-game control leg on the same rail to renormalize the numerator at matched concurrency (cheap — ~1 window or bench-rail), or (b) pre-register a stated correction model with the C3 threshold tightened accordingly.

**[MAJOR — new] The hang-risk mitigation is diagnostic, not preventive, and the "safe" concurrency margin is not transferable to the 72B.** The reported hang threshold (≥8 concurrent sessions, 15–20 min) was observed on the *27B duck* stack; hang thresholds of this class are typically memory-pressure dependent, and 72B-AWQ at ~43GB weights leaves materially less KV headroom, so 4-game concurrency is not demonstrably below the danger zone for *this* config. As written, a silent hang burns the full scored window and slot 1, and is only diagnosed post-run via the heartbeat log. Fix: add an in-run liveness gate (no completed action in N minutes → one server restart, then loud-fail) and/or a 20–30 min bench-rail smoke at 4-way before committing the scored window; watch #684625 for root cause is not a mitigation.

**[MINOR — new] Q5's accounting question must be answered by evidence, not definition.** Whether a boristown fork submission is "filler (no window cost)" is precisely the FATAL-1 quota-exemption question wearing a new hat; declaring it filler by fiat would launder the unevidenced footnote into policy. The fork-diff-first ordering ($0, today, byte-matched metadata) is correct and I endorse it; the *classification* waits on the quota ledger.

## Questions for the authors (numbered)

1. Produce the one-paragraph reconciliation of 1.33/1.86 vs the briefing's 0.43/1.56 (formula or stale-artifact identification). What prevents this from being written today?
2. What is the numeric ρ_action GO/NO-GO threshold in C3, and is it defined at matched concurrency? Will you run a 27B 4-game control leg to fix the numerator?
3. Is the frozen 480 actions/7920s an aggregate over 25 games (~19 actions/game) or per-game? The envelope arithmetic differs by >6× between readings.
4. Does the canary's window arithmetic include weight download/load/warmup inside the 7920s denominator, and is that the deployment-representative accounting?
5. On Q1 (W2 disposition): the eval build is $0 but the push slot is not — what is the slot's opportunity cost in your own E[max] currency vs tr87? If you cannot price it, why does the question reach the panel?
6. Attach the scored-rail kernel-log path with device query output before the slot-1 push; if the push has already fired by panel time, attach it retroactively from that run's log — which is it?

## What I cannot judge

The sentinel mechanism story (RHAE observables, FACT-line policy perturbation semantics), the depth-lane/EWM research adaptations in §1c, the statistical calibration of the W2 two-seed KILL rule, and the competitive-strategy legitimacy of forking boristown's public 1.47 (methodology panel). I also cannot independently verify the leaderboard figures, discussion-thread contents, or the hidden-rerun quota semantics of this specific competition — which is exactly why I demand the ledger artifact.

## Verdict: MAJOR-REVISION

## Score: 5/10