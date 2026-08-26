# Daily brief — 2026-07-22

## §1a Result deep-dive

**Draw:** frozen-fork filler = **1.14** (00:07Z daemon fire, on schedule — the
audit-stub fix validated live for a second consecutive night). Band-typical
upper-half draw; frozen observed band 0.76–1.33 (now ≥1.39 including zoli800's
draw of the identical artifact, see §1b).

**Pre-registered expectation:** plain draw from the frozen distribution. **Met.**
No mechanism claim attaches; no kernel pull (filler runs are not evidence
artifacts).

**Ledger update:**
- Frozen control n=9 {0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14}:
  mean **0.992**, σ̂ **0.155**.
- Pooled (frozen + closed war arm) n=14: mean **0.972**, σ̂ **0.144**.

**Interpretation:** the 1.14 pulls the frozen mean back up to ~0.99; σ̂ stays
dead-center in the 0.13–0.17 model bracket. Naive-normal E[max over ~104
remaining windows] ≈ **1.41** — tantalizingly at the wall (1.44), but
P(single draw ≥ 1.44) ≈ **0.0006**/window under the same model, i.e. the wall
is reachable only via the extreme tail across many windows (and the model's
common-night correlation makes the naive E[max] optimistic). **Priced
conclusion unchanged:** filler is a lottery; credible ≥ +0.06–0.12 experiment
windows (sentinel W1/W2, EWM) remain better-priced than filler once their
gates clear.

**External calibration bonus (fork-diff, §1b):** zoli800's public 1.39 with the
byte-identical artifact confirms the true artifact distribution reaches 1.39+.
Our E[max] estimate is consistent with what another user of the same artifact
actually drew.

## §1b Discussions sweep (learnings/war_room/discussions_2026-07-22.md)

3 new posts + 2 Code-tab findings:
- **ADOPT → resolved NO-ACTION same-day:** zoli800 public notebook at **1.39**
  looked like a pin-diff candidate; full metadata+code diff
  (learnings/war_room/fork_diff_zoli800_2026-07-22.md) shows **byte-identical**
  config (same docker digest, same 3 dataset pins incl. vrfai Qwen3.6-27B FP8,
  code delta = one mojibake char in a print string). It is a lucky draw of OUR
  artifact. No re-fork (a fresh-slug variance clone remains a priced option,
  not queued). **Fork-wave risk is real:** anyone can clone to the same
  distribution; expect filler rank erosion.
- **ADAPT (ops, 1-min):** S. Brodehl reports RTX-6000-Pro accelerator not
  selectable (TPU only), host-silent → verify the accelerator selector before
  the next GPU build; added as explicit precondition in A17′ v2.
- IGNORE: #728210 beginner question; Scott Le Grand <100B-quant lament
  (validates 72B-AWQ ceiling choice); mbmmurad 0.86 notebook (superseded).
- **Watch-item unchanged:** 1.15x-vs-1.0x efficiency cap still host-unanswered;
  keep treating LB math as 1.0x completion-weighted.
- LB: KOJIMA #1 1.86 (active); gold cutoff ~1.47; wall 1.44 (ranks 19–22);
  Tufa Labs rank 18 at 1.45.

## §1c Research sweep (learnings/war_room/research_2026-07-22.md)

Not thin — the action is off-arXiv:
- **ADOPT — Schema harness** (Impossible Research, wide Jul 20–21):
  self-reported **98.98% RHAE, 25/25 public games, 183/183 levels** (Opus 4.8 +
  Fable 5), vs Rodionov 58.12% prior SOTA. Method: executable-program world
  model (inverse graphics → inverse dynamics), step function + goal predicate
  **certified by replay against full interaction history**, then BFS in-sim at
  zero interaction cost. Direct hit on our EWM step-0 abort problem:
  certification-against-history turns state aliasing into a certification
  failure → model revision/re-observation, never an abort. 50 trajectories
  released (HF schema-harness/arc-agi-3-schema-traces) — mining agent running
  (runs/schema_traces_mining/). Caveats: unverified, public-set-only.
- **ADAPT:** EnvProbe (arXiv:2606.31422) → budgeted probe selection on
  certification failure (EWM contract v1.1); its negative finding on verbalized
  self-uncertainty independently validates the sentinel's
  deterministic-observable design.
- **ADAPT:** RedHatAI/Qwen2.5-VL-72B-Instruct-quantized.w8a8 — named fallback
  SKU for A17 (≈100% vision-benchmark recovery, 1.87× vLLM speedup); upgrades
  the generic FP8 fallback pre-registration; verify 96GB fit before sealing.
- PARK: Agent-BRACE, Token Budgets. Ledger note: Schema's escalate-on-low-score
  (27B→72B per-game max) is a clean escalation-economics template.

## §2 Panel state — R16 verdicts synthesized (round16/_directives.md)

R16 (A14 sealing round): **0 ACCEPT / 5 MAJOR-REVISION / 0 fatal → A14 does NOT
seal; seals on R17.** Wedge post-mortem: verdicts were collected 07-21 08:32 but
the session died before synthesis; synthesis done today.

Rulings: Q1 arithmetic ratified, seal withheld (4 conditions); **Q2
SENTINEL_BUDGET=150 APPROVED-WITH-CONDITIONS** (attempt-unit conflation is the
core defect — 3 discharge items pre-push, in flight); Q3 (c)+Reki kill
ratified, no resurrection; Q4 EWM re-price ratified but audit-informed carrier
ruling NOT ratified (phase-augmentation = unregistered mechanism); Q5 banking
RESTRICTED to full-replay-only v1; **Q6 A17′ REJECTED as drafted** (4/5:
dead-code null branch, unquantified false-NO-GO, ρ mismeasured, cost figures);
Q7 W0 n=4 pooling approved w/ conditions; Q8 dream digest NOT-ADDRESSED
(re-table in R17).

R17 sealing checklist (9 items, all $0): composition sentence w/ per-branch
P(pass); held-out resolver validation (Wilson-LB ≥0.95, failures→UNRESOLVED);
engine-drift precondition + Q7 conditions; B+ rows pre-flagged + A16
replay-cost extension; A17′ re-filed; EWM measurement config sealed; sentinel
condition 4 + trigger-frequency table; hash commitment for the 0.99 bar;
sensitivity annexes.

## §3 Development state (agents in flight as of brief writing)

1. **sentinel-q2** — discharging the 3 pre-push conditions (attempt-unit
   analysis on ka59/re86/tu93; canary v3 with cross-attempt-waste counting;
   tokens-per-fire sentence). Push (2 available today) fires on its GO.
2. **resolver-holdout** — held-out resolver validation (the single item
   unblocking A14 seal, EWM carriers, banking scope).
3. **a17-repair** — A17′ v2 editing pass per Q6 defect list (incl. W8A8
   fallback SKU + accelerator-selector precondition).
4. **schema-mine** — Schema trajectory mining vs our aliased-game class.
5. R17 circulation assembled from 1–4 + this brief; panel launch after
   (panel_round.py now tree-kills timed-out reviewers — orphan fix from
   today's 13-process wedge cleanup).

**Infra:** wedge root-caused (proc.kill() orphaned kaos-run's claude children);
tree-kill fix live. ARCDailyIterate already has PT2H ExecutionTimeLimit but it
doesn't reach detached grandchildren — orphan-sweep at session start is now the
working mitigation (executed today, 13 killed).

## Open questions
1. Does the held-out validation shrink the RESOLVABLE set (→ smaller EWM
   carrier set / banking scope)?
2. Schema harness: does trace mining confirm the certified-model claim on our
   aliased games? If yes, EWM contract v1.1 pivots to certification-as-resync
   — R17 agenda.
3. Tonight's window: sentinel W1 seed-1 if Q2 conditions discharge + R17-
   compatible; else frozen-fork filler (queued, verified).
4. Fork-wave erosion of filler rank — monitor; fresh-slug variance clone is a
   priced fallback, not queued.
