# PREREG — BENCH ARM (paired within-run A/B on the 13-graft stack)
**SEALED 2026-08-30, BEFORE THE DATA LANDS.** Kernel `canivel/arc3-tv28-bench`.
Notebook sha256 `7f822e8c06f3855cc5571a7fb2a1785347d829b8493a3b3081e7f742dda71f33`.

## 1. Vehicle
All 17 cells BYTE-IDENTICAL to `thtennant/arc3-duck-bench` (verified cell-by-cell); env
metadata byte-matched (RtxPro6000, same docker sha, same 2 datasets, same model_source).
Our own `arc3-tv28-fork` differs from that kernel in cells 12 and 14 ONLY; both were
transplanted verbatim. Preflight `trusted-fork` 0 FAIL / 1 WARN (WARN = unpushed, expected).

## 2. Design
Four tagged replicas of ONE game (`BENCH_GAME="m0r0"`, tags A0/B0/A1/B1), one offline
Save & Run, `bm.n_passes = 1`. Arm A = the full 13-graft stack. Arm B = `{"B": ["*"]}`
suppresses all graft TEXT, restoring the harness's stock world-model methods (the v12
floor prompt). Placebo control: every graft is INFORMATIONAL, so arm B is the same code
path, same wrappers, same engine calls — only the prompt text differs.

**COST: 0 submissions, 0 draws, 1 GPU Save & Run.** This kernel is NEVER queued and NEVER
submitted. The replica block is guarded on `if not TRUE_SUBMISSION:` (verified by assert
at build time), so it is provably inert in a real rerun.

## 3. What this instrument CAN and CANNOT do — SEALED
n=2 per arm on ONE game. Against a within-config spread of the 1.82/0.00 class
(tennant, byte-identical v22 runs, 2026-08-23), a 2v2 detects only a LARGE effect, and
**no inferential test is licensed at all** — the minimum one-sided Mann-Whitney p at 2v2
is 0.333. This is a **KILL instrument, not a promotion instrument.**

- **KILL READ (the only conclusion licensed):** if BOTH B replicas strictly exceed BOTH A
  replicas on levels cleared — a clean 2–0 separation in B's favour — read as *"the
  13-graft stack does not help and plausibly hurts on m0r0"* ⇒ stop investing in the
  graft stack; do NOT redraw the TV28 fork.
- **NO-VERDICT:** ANY overlap between arms ⇒ no conclusion. The instrument returns
  "cannot separate" and we stop spending on it. This is the expected outcome.
- **FORBIDDEN, sealed in advance:** A > B in ANY configuration MAY NOT be read as evidence
  the grafts help, MAY NOT promote the TV28 fork to a queue head, and MAY NOT be entered
  as a positive result. `feedback_screen_calibration_range`: 0 of 36 artifacts have ever
  produced a board draw above the local floor.
- Any result is a property of **m0r0 alone** and does not generalise to the 25-game field.

## 4. Certification (infra gates — checked BEFORE any outcome is read)
1. `[bench] game list:` printed with FOUR distinct external_game_ids carrying A0/B0/A1/B1.
2. `TAAF_GRAFTS FEATURES={...} API_VERSION=1` banner present AND containing the bench key.
3. `benchmark.json` carries 4 game-runs whose arm letters come from the artifact path.
4. **NEGATIVE CONTROL / INFRA DEATH:** if `[bench] replica build failed, using [:4]` appears,
   the arm is DEAD — the run played four DIFFERENT games, not four replicas, and NO
   outcome may be read from it under any reading.
5. Kernel log present and NON-EMPTY before scoring (the 08-27 partial-pull defect).

## 5. Known limitation carried forward, not fixed by this arm
`thtennant/taaf-kaggle-source-share-fork` is pinned by SLUG with no version (Kaggle kernel
`dataset_sources` has no version pin). The mounted bundle at build time is the 2026-08-29
12:51 republish: `bench.py` 11,430 B, `composite.py` 24,798 B (file census verified today).
A future republish silently changes this kernel's identity. Recorded, not mitigated.

---

# AMENDMENT 1 — 2026-08-30, WRITTEN BEFORE ANY BENCH DATA LANDED
**Trigger:** `feedback_audit_the_instrument` — audit the instrument before the data lands.
Measured `m0r0`'s dynamic range from our OWN retained archive after the m0r0 kernel was
pushed but ~9 h before it can return. This amendment only TIGHTENS the read; nothing is
loosened, and no bench outcome existed when it was written.

## A1.1 FINDING — `m0r0` IS THE WORST OF THE 25 GAMES FOR THIS PURPOSE
Two independent instruments, two different models, same answer:

| instrument | n | lc mean | lc sd | min | max | p(lc>0) |
|---|---|---|---|---|---|---|
| all-config archive, every m0r0 game-run we retain | 76 | 0.197 | 0.398 | 0 | **1** | 0.20 |
| within-config, `runs/tufa_example_run` 20 clone replicates (Qwen3.6) | 20 | 0.050 | **0.218** | 0 | 1 | **0.05** |

`m0r0` has **NEVER cleared 2 levels in 76 retained runs**, and **1 of 20 clone replicates**
cleared a single level. It ranks **#18 of 25 games** by lc sd on the all-config archive and
**LAST** on the within-config clone set. Our own 08-29 commit run of this very stack scored
it lc 1/6, final_score 0.475.

**Consequence:** a 2-v-2 on `m0r0` returns `0,0,0,0` by construction in the large majority
of runs. Under the measured p≈0.05–0.20 the sealed KILL read cannot fire often enough to
be informative, and the run's NO-VERDICT is predictable in advance. The author chose this
game; we inherited the choice without pricing it. **The m0r0 kernel is therefore
re-designated a REPLICATION of the author's exact rig and an infra shakedown of the bench
path — not a measurement of the graft stack.** Its sealed read stands unchanged, and its
expected outcome is NO-VERDICT.

## A1.2 SLOT 2 — `sb26` BENCH, the same rig with ONE token changed
Kernel `canivel/arc3-tv28-bench-sb26`, notebook sha256
`abb8c970cef8eef14bf50c0df6fe7e32b307f846cd92778bfe53d6e2cbd95826`. Verified: cell 14 is
the ONLY cell differing from the m0r0 bench, and the ONLY change is
`BENCH_GAME = "m0r0"` → `BENCH_GAME = "sb26"`. Guard re-asserted at build time.

**Why `sb26`** (within-config clone set, n=20): lc mean 1.100, **lc sd 0.300, min 1, max 2,
p(lc>0) = 1.00** — it is the only candidate that clears a level in *every* replicate, so the
outcome is graded rather than zero-inflated, and its low within-config noise is what a 2-v-2
KILL read actually needs. `ft09` has more range (sd 0.975, 0–3) but its noise manufactures
false 2–0 separations, which is the exact failure mode a KILL instrument must not have.
Independently: `sb26` carries **50.4% of the certified field floor's entire mean_score**
(strategy-0822, 77.778 of 154.3 points), so it is also the game that actually moves our number.

**CAVEAT, stated up front:** the 20-clone set is **Qwen3.6**; we run **Qwen3.8**
(weekend-prep). Per-game difficulty may shift with the model, so `sb26`'s p(lc>0)=1.00 is
NOT guaranteed on our stack. What IS robust is the negative result: `m0r0` is bottom-ranked
on BOTH instruments and BOTH models.

## A1.3 READ RULE for the sb26 arm — same discipline, one addition
Sections 3 and 4 above apply verbatim (KILL-only; A > B may NOT promote anything; any
`[bench] replica build failed` line = INFRA DEATH). **Addition:** because `sb26` is expected
to clear ≥1 level in every replicate, the primary statistic is `levels_completed` with
`final_score` as the tie-break, and a KILL requires **min(B0,B1) > max(A0,A1)** on
levels_completed. Still n=2/arm: no inferential test is licensed, the honest prior is
NO-VERDICT, and this remains a one-game result.
