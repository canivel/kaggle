# Latent-state audit protocol — quantifying hidden state behind the observable frame

Status: v1, 2026-07-20. Panel R15 mandate (5/5): state-aliasing is one root cause
behind (1) predict-metric recurrence acc 0.465 (`runs/predict_metric/report.md`),
(2) EWM step-0 plan aborts (`runs/ewm_dryrun/report.md`), (3) the N5 prune_trace
bug (`runs/war_eval_v1/prune_replay_diag.json`). This audit is a BLOCKING prereq
for EWM Stage-1 and any banking/replay build.

Implementation: `scripts/latent_state_audit.py` (stdlib-only, CPU, $0, offline).
Output: `runs/latent_state_audit/report.md` + `report.json`.

## 1. Data

- **Primary**: per-action event traces `runs/kernel_pulls/*/artifacts/*_events.jsonl`
  and `runs/phase1_ab/seed1/artifacts/*_events.jsonl` (auto-discovered). Each
  `type=action` event carries the full settled 64x64 `board`, `action_display`
  (fully qualifies ACTION1-5 direction and ACTION6 MOUSE(r,c)), `board_changed`,
  `level`, `score`. `type=initial` gives the pre-play frame; `type=analysis`
  frames are skipped (no action taken; predict_metric confirmed 0 digest drift).
- **Cross-reference**: `runs/ewm_dryrun/raw.json` per-game sim fidelity
  (step_acc, abort-step distribution) — the EWM-consumer view of the same
  aliasing.
- **Anchors**: N5 determinism audit (all 25 games frame-deterministic under
  full-history replay) — so ALL aliasing found here is *hidden state*, not
  engine stochasticity-from-reset. True per-transition stochasticity would show
  up as ALIASED-UNRESOLVED; N5 says any such finding on within-stream data must
  be treated as an unmodeled deterministic variable, not noise.

Analysis unit = versioned game id (e.g. `ls20-9607627b`); report row = 4-char
game id (worst verdict across versions if a game appears in several engine
versions — engine-version drift must never masquerade as hidden state).

## 2. Aliasing measurement (a)

A transition is `(s, a) -> s'` with `s` = blake2b-8 digest of the board the
agent acted on, `a` = `action_display`, `s'` = digest of the settled board.
RESET is an ordinary action. `t` = actions since last RESET (counts no-ops —
the N5 bug proved no-ops tick hidden state).

For every key with >= 2 visits, the empirical next-frame outcome distribution:

- **determinism** = sum(max outcome count per key) / sum(visits), over repeat
  keys, visit-weighted. 1.0 = frame is Markov on the observed support.
- **entropy** = visit-weighted mean Shannon entropy (bits) of the outcome
  distribution per key. > 0 = aliased.
- **aliased-key rate** = repeat keys with > 1 distinct outcome / repeat keys.
- Two scopes: **pooled** (keys shared across all streams of the versioned game
  — what banking/persistent models see) and **within-stream** (keys scoped per
  trace — what an in-run EWM sim sees). Within-stream aliasing is the stronger
  finding: the same session, same engine, same frame, same action, different
  outcome ⇒ hidden state moved.

## 3. Candidate hidden variables (b)

Augment the key with a candidate `h`: key = (s, a, h). Recompute determinism.
Candidates, ordered cheapest-first (the *minimal* augmentation reaching >= 99%
determinism wins):

| rank | candidate | h | class |
|---|---|---|---|
| 1 | level | pre-action level | observable-meta |
| 2 | score | pre-action score | observable-meta |
| 3 | meta | (level, score) | observable-meta |
| 4 | parity | t mod 2 | hidden phase |
| 5 | mod3 / mod4 / mod5 | t mod k | hidden phase |
| 8 | prev_bc | did the previous action change the board | hidden history |
| 9 | hist1 / hist2 / hist3 | last k action keys | hidden history |
| 12 | meta_parity | (level, score, t mod 2) | compound |
| 13 | meta_hist1 | (level, score, last action) | compound |
| — | tcount | exact t (diagnostic only, degenerate) | diagnostic |

**Support guard**: an augmentation is only eligible as a resolver if its
remaining repeat-visit mass >= max(10, 20% of base repeat visits). Otherwise it
merely shattered the keys (any injective function "resolves" everything at n=1)
and is reported as SUPPORT-COLLAPSED.

## 4. Per-game verdict (c)

- **CLEAN** — base determinism >= 0.99 (frame is Markov on observed support).
- **CLEAN-META** — resolved by observable metadata (level/score); the *full
  observation* is Markov even though the raw grid is not.
- **ALIASED-RESOLVABLE(h)** — a hidden candidate h reaches >= 0.99 with support.
- **ALIASED-UNRESOLVED** — no candidate resolves. Per N5 this means a hidden
  variable outside the candidate family (deep counter, object-internal state),
  not stochasticity — but it is operationally equivalent to stochastic for any
  frame-conditioned model.
- **LOW-SUPPORT** — < 20 repeat visits; no verdict earned either way.

## 5. Consumers — how the table answers them

**EWM Stage-1 (carrier selection + resync question)**
- Safe carriers = CLEAN / CLEAN-META games: a sim keyed on the frame (plus
  visible meta) can be faithful; mismatch-aborts there are sim bugs or engine-
  version drift, not aliasing.
- ALIASED-RESOLVABLE with a *phase* resolver (parity / mod-k): the sim drifts
  out of phase but reality stays deterministic ⇒ **resync-before-abort works**
  (re-read the settled frame, re-plan; or better, add the phase variable to the
  sim state). ALIASED-UNRESOLVED ⇒ resync does NOT restore predictability —
  abort-and-fallback is correct; do not carry EWM there.
- Cross-check column: ewm_dryrun step_acc + step-0 abort share should
  anti-correlate with determinism; step-0 aborts on games whose aliasing is
  phase-resolvable are exactly the "timer/hidden-counter phase misalignment"
  failure R15 named.

**Banking / replay**
- N5 already proved: full unpruned replay from RESET survives on all 25 games.
  The audit refines that to *partial/pruned* replay: a banked trajectory may be
  spliced or pruned ONLY in CLEAN / CLEAN-META games (frame Markov ⇒ a matching
  frame is a sufficient resync point). In ALIASED games — resolvable or not —
  banking must be **full-replay-only from RESET, zero pruning** (the exact
  prune_trace failure mode: dropped leading no-ops = dropped hidden-state
  mutations).

## 6. Selftests (must pass on every run)

1. **Hidden mod-3 counter**: synthetic game whose action only fires when a
   hidden counter (invisible in the frame) % 3 == 0. Audit must find base
   aliasing and recover `mod3` as the minimal resolver at >= 99%.
2. **Clean Markov walk** → CLEAN, zero entropy.
3. **Coin-flip transitions** → ALIASED-UNRESOLVED (no candidate, incl. history,
   may claim it).

## 7. Limitations (declared, not hidden)

- Determinism is measured on *observed support*; rarely-visited keys can hide
  aliasing (Wilson-style caution applies at low repeat counts — hence
  LOW-SUPPORT).
- Candidate family is finite; UNRESOLVED means "not resolvable by cheap
  counters/history windows", not "stochastic" (N5 forbids that reading).
- Streams come from agent policies, so key coverage is policy-biased; a CLEAN
  verdict is "no aliasing seen where the agent actually walked".
