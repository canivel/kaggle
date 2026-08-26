# Where we stand

We are mid-pack and falling. Best ever 0.43 is below the rank-20 cutoff of 0.50 — realistic rank today is **22-28**, with downside to 30+ as April stragglers get bumped and June climbers (Tong Hui Kang 0.79, the last dance 0.71, face-of-agi 0.63) settle in. Tufa Labs jumped to 1.21 in a single week, redefining the ceiling: top-5 is now 0.66+, top-10 is 0.63+. Our gap to top-10 is **+0.20**, and the fresh-slug confirmation draws (0.22, 0.28) sit at the v35 distribution mean, not at the 0.43 tail. Trajectory is flat-to-down: v35 family has reached a structural ceiling (14/25 local games unreachable at BFS-s=60 budget=400), the last 7 days produced 3 ERRORs and 2 mid-distribution draws, and the next milestone deadline (June 30 — five days) will compress the LB further. We have not climbed since 2026-06-03 in any non-noise sense.

# What top teams are doing differently

- **Tufa Labs (1.21) runs StochasticGoose at scale**: 4-layer CNN over 64x64 frames + RL predicting which actions change state, with heavy Test-Time Training. The 12.58% RHAE Preview score they published is the recipe — they have evidently iterated it past 1.0 on private games. Our v36/v38/v55 attempts at SG transfer **scored 0.01-0.06**; we abandoned it after three regressions but it is what is winning.
- **Executable LLM-generated world models** (Rodionov, arXiv:2605.05138): a coding agent writes a Python simulator of each game, verifies it against past observations, MDL-refactors, then plans through it. GPT-5.5 hits **15/25 games solved, 58% RHAE**. This is a different paradigm than ours — code synthesis as the world model.
- **OpenClaw on Anthropic Opus 4.7** holds the Community LB at 5.2% — a harness + frontier LLM, not pure search. The community signal is that scaffolded LLMs beat naked search on novel environments.
- **Graph-based exploration with visual-feature priors** (Rudakov et al., arXiv:2512.24156): training-free, prioritizes untested state-action pairs by visual features — placed **3rd on private LB pre-launch**. Our v10 graph explorer is in this lane but lacks the visual-feature action priors.
- **SOAR-style hindsight fine-tuning** of the proposer on successful trajectories — 52% on ARC-AGI-1 public test by recycling wins back into the LLM. We evolve prompts but never train on rollouts.

# What we are NOT trying

- **Executable world models via code synthesis**. Rodionov's 58% RHAE is the largest reported single-paper gap vs our stack. We do not have an LLM-writes-a-simulator loop. JEPA is a latent world model, not an executable one — it cannot be MDL-refactored or symbolically verified.
- **Test-Time Training**. Every winning ARC-AGI-2 entry used TTT; the ARC Prize 2025 technical report (arXiv:2601.10904) is explicit that **no static method exceeds 11%**. We do zero gradient updates at inference. This is the single biggest paradigm gap.
- **StochasticGoose properly**. We tried it three times (v36, v38, v55) and quit after Kaggle scored 0.01-0.06 despite local RHAE 0.199. We never debugged the local→LB gap — that gap is exactly what Tufa Labs solved on the way to 1.21.
- **Hindsight fine-tuning of proposers** (SOAR) and **verifier-gated agent evolution** (SEVerA, arXiv:2603.25111). Our `evolve_claude.py` mutates strategies but does not formally verify or fine-tune on wins.
- **Frontier-LLM harness**. OpenClaw's 5.2% Community LB shows scaffolded Opus 4.7 beats search-only stacks. We have not shipped a Claude-in-the-loop agent to Kaggle. (Note: sandbox is no-internet for Official LB; Community LB allows it. The harness pattern still informs decision-making logic we can port.)
- **Visual-feature action priors on the graph explorer** (Rudakov). Cheap to add to v10/v24/v25 and is a top-3 private-LB technique we are 80% of the way to already.

# Hypothesis check: is v35+JEPA the right bet?

**Pro.** v62's failure mode is plausibly a timeout, not a model-quality issue: 253ms × thousands of decisions × 25 swarm agents trivially blows the 8h wall. The throttle plan (n_sims=8, depth=3, ~120ms, cap 60/agent + 8/level) is the right surgical fix and the fresh slug removes the forge35 curse confounder. If v63 clears the queue, we get a clean read on whether a world-model fallback adds value on the 14 BFS-unreachable games.

**Con — and this is the dominant signal.** JEPA-XXS at 2.3M params int8 is not a world model with predictive power on novel ARC dynamics; it is a representation. It has no mechanism to discover the latent rule of a game it has never seen — that is what Rodionov's executable simulator and StochasticGoose's frame-change predictor explicitly do. Even if v63 clears, the upside is **incremental** (maybe +0.03 to +0.08 on noise floor), not structural. The 0.43 → 0.60 gap is paradigm-shaped, not budget-shaped. Continuing v63 burns our last week of pre-milestone submissions on a path that, at its best, lands us at 0.30-0.40 — still outside top-20.

**Verdict.** Ship v63 because it is already built and queued, **but treat it as a probe, not a strategy**. Do not iterate on v35+JEPA past v63. Pivot in parallel.

# Recommended next moves

Ranked by EV over the next 14 days:

1. **Re-attack StochasticGoose with a Kaggle-environment reproduction harness first.** Tufa is at 1.21 with this exact stack. The 1.56 local → 0.02 Kaggle collapse (forum thread 703990 + our own v55 result) is a *known, named, infrastructural* bug — lock-based serialization of concurrent games. Build a local Docker that mirrors the Kaggle sandbox exactly (per `feedback_kaggle_env_match.md`), reproduce the collapse, fix it, then run SG. **This is the single highest-EV move.** Budget: 3-4 days. If it works, +0.20 to +0.40 is on the table.
2. **Add visual-feature action priors to v10 graph explorer (Rudakov 2512.24156 method).** Training-free, hits 3rd on private LB pre-launch. We already have the graph; we are missing the visual feature scoring of untested actions. Budget: 2 days. Expected +0.05 to +0.10, low-risk submission.
3. **Build a minimal executable-world-model loop using Claude as the code generator** (Rodionov 2605.05138). One game at a time: agent observes, writes a Python sim, verifies against history, plans. Even a half-implementation that solves 2-3 extra games is +0.08 to +0.12 and opens a strategic lane no one in our peer cluster is in. Budget: 4-5 days, parallel to (1). Note Official LB sandbox is no-internet — generate the simulators offline against the public games and ship them as static modules.
4. **Ship v63 as planned, then stop iterating on v35+JEPA.** It is built. Submit it tonight. If it scores below 0.30, abandon the JEPA branch entirely. If 0.30-0.40, hold as safety net. Do not spend another build cycle on this lane.
5. **Add TTT to whichever of (1)/(2)/(3) ships first.** Even a thin TTT loop — 100 gradient steps on the current game's observations before planning — is what every winning ARC entry has done. Budget: 1-2 days bolt-on.

# Risks

- **Milestone #1 is June 30 (5 days).** Open-source requirement (CC0 / MIT-0) gates the prize. Verify license headers on every file we'd submit *before* the cut.
- **Reproducing the StochasticGoose Kaggle gap may take longer than 3 days.** If after 48h we cannot reproduce locally, fall back to (2) for guaranteed shippable progress. Do not let SG eat the whole week.
- **Cursed-slug pattern.** forge35 ERROR'd 3x in a row; forge62 may carry similar state. **Every new architecture goes on a fresh slug, every time.** Already proven (2026-06-24 fresh-slug test passed).
- **Tufa Labs and Tong Hui Kang will climb again before June 30.** The 1.21 → top-10 0.63 gap means top scorers have headroom; expect 1.30+ and 0.85+ this week. Our +0.20 climb won't move us into top-10 alone — it moves us into top-15 to top-20. Plan for two climbs, not one.
- **Watch-list.** (a) v63 outcome by tomorrow morning — if ERROR again, the JEPA lane is dead. (b) Whether Kaggle discussion thread 703990 names a concrete fix for the concurrency timeout — that single fact could unblock SG. (c) Any Tufa Labs commit or notebook publish; their stack is the reference. (d) Our submission quota usage — do not waste a slot on a v35 redraw "just to see".
- **Do not A/B prompts, do not single-knob tune.** Both are noise per `feedback_prompt_is_noise.md` and `feedback_simplicity_wins.md`. Every minute on those is a minute not spent on SG reproduction.
