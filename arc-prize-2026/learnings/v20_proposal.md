# v20 Proposal — Sim/Runtime Sync Verification

**Date:** 2026-05-01
**Predecessor:** v19 = 0.20 (variance band 0.19–0.34 on near-identical ashvin code).
**Single change:** First-step BFS verification gate — abandon BFS path when sim's predicted frame after action 1 does not match the live frame; fall through to CNN/WorldModel.

---

## Section 1 — DIAGNOSIS

Evidence from `runs/runs-20260430-180344-v19-perg2-ft09/run.log`:

- 18:03:54.671 — `BFS L0: 8 effective actions`
- 18:03:57.917 — `BFS L0: SOLVED (A*) in 4 actions (390 explored, 3.2s)`
- 18:04:04 → 18:04:31 — agent emits 69 ACTION6 calls; `levels completed 0` throughout.

The first 4 ACTION6 calls are the BFS-found solution; the rest are CNN-fallback random clicks after the BFS path was exhausted without winning. **BFS solved a different game than the one running.**

`ft09.py` has **no random/seed/time/np.rand calls** (grep confirmed), so the divergence is NOT RNG-based. The most likely root causes, ranked:

1. **Hidden initial-state divergence.** BFS instantiates `game_cls(); RESET; RESET` at line 645–647. The live engine receives `RESET` from the API host, which may have already advanced the game with internal warm-up frames the BFS replica never sees. ARC's engine returns multi-frame animations on RESET (`r0.frame[-1]` only takes the last frame); sim and live can land at different `frame[-1]` if animation length differs. Sim assumes determinism but the live host may emit one extra animation tick.
2. **Frame-vs-internal-state divergence.** ft09 has 23 sprites + tags `bsT`/`AcT`. Sprite enumeration order or `_available_actions` initialization can depend on dict iteration order across Python interpreter startups (e.g. PYTHONHASHSEED set differently on the host).
3. **Action coordinate mismatch is unlikely** — both paths use the same `(x,y)` ints in `ActionInput.data`.

The 5/11 BFS-solving games failing transfer (ft09, dc22, lp85, r11l, s5i5) all share short L0 BFS solutions (3–20 actions). When BFS path applied to live env produces no level advance, agent burns the rest of the budget on CNN random clicks. This is the dominant fixable bottleneck.

---

## Section 2 — V20 PROPOSAL

**Pick option (b): FIRST-STEP VERIFICATION.** Lowest implementation risk, directly addresses the diagnosed sync failure, free fallback to existing CNN+WorldModel path.

Option (a) online-BFS via `deepcopy(arc_env)` is rejected: `arc_env` is an HTTP API client — deepcopy yields a duplicate session that cannot independently step. Option (c) frame-driven sync is rejected: requires per-game introspection into `game.__dict__` to mutate it from a pixel frame, which is unsolved (the inverse problem).

### Mechanism

After BFS produces `s._bfs_solution`, the live agent applies action 1 and captures the live frame. BFS replays its own action 1 in sim and produces the predicted frame. If the two pixel arrays differ, declare the BFS path stale and clear it. Costs one live action (already spent regardless), one sim step (~50ms), one numpy compare.

### Code stub (insert in `choose_action` between BFS execution and CNN fallback)

```python
# ===== BFS SOLUTION EXECUTION =====
if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
    act_id, data = s._bfs_solution[s._bfs_step]
    sel = GameAction.from_id(act_id)
    s._last_action_data = {k: v for k, v in (data or {}).items() if k != 'game_id'}
    raw = s._raw(lf)

    # NEW: first-step sync gate
    if s._bfs_step == 1 and not getattr(s, '_bfs_verified', False):
        # raw here = live frame AFTER step 0 was sent on the previous turn
        sim_frame = s._bfs_predict_frame(s.cl, 1)  # replay step 0 in sim, return frame[-1]
        if sim_frame is None or not np.array_equal(raw, sim_frame):
            logger.warning(f"BFS sync FAIL @ L{s.cl}: live!=sim after step 0; abandoning path")
            s._bfs_solution = None
            s._bfs_step = 0
            # fall through to CNN fallback this turn
        else:
            s._bfs_verified = True

    if s._bfs_solution:  # still valid
        s._bfs_step += 1
        s.fhist.append(raw.copy()); s.pr = raw.copy(); s.la += 1
        return sel
```

Add helper on `MyAgent`:

```python
def _bfs_predict_frame(s, level_idx, n_steps):
    g = s._bfs.game_cls()
    g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    for pi in range(level_idx):
        for act_id, data in (s._bfs.solutions.get(pi) or []):
            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
            g.perform_action(ai, raw=True)
    last_r = None
    for act_id, data in s._bfs_solution[:n_steps]:
        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
        last_r = g.perform_action(ai, raw=True)
    return np.array(last_r.frame[-1], dtype=np.int64) if last_r and last_r.frame else None
```

Reset `s._bfs_verified = False` on every level change (next to existing `s._bfs_step = 0`).

---

## Section 3 — FALSIFIABLE PREDICTIONS

- **Expected:** Group B games (ft09, dc22, lp85, r11l, s5i5) flip from 0 levels-completed to non-zero in the local 19-game sweep, OR show CNN-fallback making progress (currently CNN never gets called because BFS keeps emitting stale path through to MAX_ACTIONS).
- **Disprove case:** if Group B remains at 0/0/0/0/0, the divergence is not at step 0 — frames match by chance (e.g. ACTION6 click on background pixel produces identical frame). Then we must check step 2/3 verification, OR the issue is the frame-extraction format itself (`frame[-1]` vs `get_pixels()`).
- **Group A regression risk:** if `_bfs_predict_frame` returns slightly-different frames on currently-working games (ls20, sk48, ar25, m0r0, cd82, cn04), v20 abandons working paths. Mitigation: log mismatch and percentage; require both Group A levels-completed unchanged and Group B>0 to call this a win.
- **Plan B if v20 fails on Group B:** the divergence is deeper than step 0. Pivot to (c) — write a per-game state-injector that mutates `game._available_actions` and `game._field_state` from the live frame to force-sync. Higher implementation cost (~300 lines) but the only remaining option.

---

## Section 4 — IMPLEMENTATION

**File:** `f:/kaggle/arc-prize-2026/notebooks/forge_agent/v20_agent.py` — clean `cp` of `v19_agent.py`, then 3 edits:

1. **Init flag** (in `MyAgent.__init__` near line where `_bfs_step` is initialized): `s._bfs_verified = False`.
2. **Reset on level change** (in `choose_action`, around line 1817 next to `s._bfs_step = 0`): add `s._bfs_verified = False`.
3. **Insert verification block** (replace lines 1890–1899, the "BFS SOLUTION EXECUTION" block, with the stub above).
4. **Add `_bfs_predict_frame` method** to `MyAgent` (anywhere in the class).

No changes to `BFSSolver`, `ForgeNet`, training, or CNN fallback. Total diff ≤ 60 lines.

**Local test plan:**
1. Copy `v19_agent.py` → `v20_agent.py`, apply edits.
2. Run on ft09 alone: `python local_eval.py --agent v20_agent --games ft09 --max-actions 400 --time 240`.
3. Inspect `run.log` for `BFS sync FAIL` line. If logged AND levels_completed > 0, success.
4. If ft09 advances past L0, expand to all 5 Group B games.
5. Then run a Group A game (ls20) to verify no regression.
6. Full 19-game sweep before any Kaggle submission.

**Estimated implementation time:** 30 min coding + 1 hr local validation per game (5 Group B + 1 Group A control) = ~7 hrs wall-clock with sweeps running serially. Submit only after sweep confirms no Group A regression.
