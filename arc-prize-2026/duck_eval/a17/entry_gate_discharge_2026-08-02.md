# Boristown readiness-gate A/B — ENTRY-GATE DISCHARGE report — 2026-08-02

Discharges the two outstanding entry gates blocking the boristown readiness-gate
A/B prereg (`learnings/war_room/boristown_ab_prereg_2026-07-29_DRAFT.md`), namely
its fire-condition **§7.2** (gated-arm kernel exists + preflight not BLOCK + entry-gate
#1) which subsumes **BLOCKER 3** (entry-gate #1: 2-seed live-firing log + non-harm
screen) and **BLOCKER 2(b)** (entry-gate #2: `--max-diff-cells 1 --pin` preflight ALLOW).
This report is READ/VERIFY only — no kernel push, no submission, no queue change, $0 cloud.
kaggle CLI pinned to 2.0.0 (`uvx --from kaggle==2.0.0`) throughout; the 2.2.x CLI drops
kernel logs.

Version/proof mapping (per ITERATION_LOG 07-31 reconciliation): **v2 = seed-1, v3 = seed-2**
of slug `canivel/arc3-duck-gate-eval` (an unlogged v1 push existed 07-30; the 2-seed proof
is v2/v3). Arm B scored canary = `canivel/arc3-duck-gate` v1, build COMPLETE.

Pulls performed this session:
- `runs/tmp_pullback_gate_eval_s2v3/` — seed-2 (v3) eval output (fresh subdir; the earlier
  `runs/tmp_pullback_gate_eval_s2/` NOT clobbered).
- seed-2 `benchmark.json` + `arc3-duck-gate-eval.log` copied to canonical
  `runs/kernel_pulls/gate_eval_v2/` so the screen runs the same way seed-1 did.
- `runs/tmp_pullback_duckgate_v1post/` — arm B canary output; `-m` notebook to
  `runs/tmp_pullback_duckgate_v1post_nb/`.

---

## 1. Entry-gate #1 — 2-seed gate-eval markers (§6 of readiness note 2026-07-30)

Required on BOTH seeds: (M1) seed banner, (M2) `GATE armed` w/ `poll=5s timeout=180s`,
(M3) boris's `vLLM server ready`, (M4) `observed-firing vllm_ready_latency_s=<X> : GATE fired`
with X ≤ 180 s; plus GPU string RTX PRO 6000 (NC-12 / prereg NOTE 4, LOW).

### Seed-2 (v3) — `runs/kernel_pulls/gate_eval_v2/arc3-duck-gate-eval.log` — exact matched lines

- **M1 seed banner** (line 11):
  `A17-GATE-EVAL seed=2 mode=readiness-gate-ab-B-eval base=canivel/arc3-duck-gate(staged) : entry-gate live-firing (offline bench, NOT scored)`
- **M2 GATE armed** (line 655):
  `A17-GATE mode=readiness-gate-ab-B version=1 graft=boristown/agi-duck-harness-fast-eval#cell16 endpoint=http://127.0.0.1:1234/v1/models poll=5s timeout=180s : GATE armed`
- **M3 vLLM server ready** (line 656): `vLLM server ready`
  (line 641 also carries boris's own readiness dict for `vrfai/Qwen3.6-27B-FP8`)
- **M4 observed-firing / GATE fired** (line 657):
  `A17-GATE observed-firing vllm_ready_latency_s=0.0 : GATE fired` → **latency 0.0 s ≤ 180 s OK**
- **GPU string** (lines 279 / 283):
  `CUDA GPU check passed for rtx-pro-6000 x1: ['NVIDIA RTX PRO 6000 Blackwell Server Edition']`

### Seed-1 (v2) — `runs/kernel_pulls/gate_eval_v1/arc3-duck-gate-eval.log` — recorded 08-01, re-verified

- **M1** `A17-GATE-EVAL seed=1 mode=readiness-gate-ab-B-eval base=canivel/arc3-duck-gate(staged) : entry-gate live-firing (offline bench, NOT scored)`
- **M2** `A17-GATE mode=readiness-gate-ab-B version=1 graft=boristown/agi-duck-harness-fast-eval#cell16 endpoint=http://127.0.0.1:1234/v1/models poll=5s timeout=180s : GATE armed`
- **M3** `vLLM server ready`
- **M4** `A17-GATE observed-firing vllm_ready_latency_s=0.0 : GATE fired` → latency 0.0 s ≤ 180 s OK
- **GPU** `NVIDIA RTX PRO 6000 Blackwell Server Edition`

Both seeds' build status COMPLETE. **All §6 markers green on BOTH seeds, latency 0.0 s ≤ 180 s.**

---

## 2. Bench numbers (offline eval, 25 games, 1 pass)

| quantity | seed-1 (v2) | seed-2 (v3) |
|---|---|---|
| games / runs | 25 / 25 (won 0) | 25 / 25 (won 0) |
| raw mean score (summary.txt) | **1.43** | **1.94** |
| raw median score | 0.73 | 0.12 |

(Raw benchmark means are the harness's own summary figure; they are NOT the A/B decision
metric — the screen below uses paired Δlevels-completed vs the null10 baseline, per the
war/sentinel precedent. The offline eval is UNSCORED by construction — cell-15 offline branch,
dummy `submission.parquet`.)

---

## 3. Entry-gate #2 — non-harm SCREEN vs null10

Run identically to seed-1: `uv run python scripts/gate_eval_screen.py gate_eval_v2`
(paired Δlc vs the 10 `runs/null10/vanilla_seed*.json` baselines; RHAE scorer validated
max err 0.0e+00 over 1000 checks). Writes `runs/gate_eval_v2/screen_report.md`.

### Seed-2 screen result — NON-HARM: PASS

- mechanism fired: **True** (armed=True, fired=True, boris_ready=True, latency_s=0.0 ok=True,
  log=arc3-duck-gate-eval.log)
- Δlc not materially negative: **True** — harm-tail p_neg=**0.9194** ≥ α=0.05;
  worst game **lf52 Δlc −0.9** vs catastrophic cap **−1.0** (within cap, no game collapses >1 lvl)
- PRIMARY paired Δlc: mean **+0.152** (sd 0.537, **9W/7L**, pos-tail p=0.0933, harm-tail p=0.9194)
- Secondary Δlog1p(RHAE): mean −0.033 (p=0.6068)

### Both-seed screen summary (side by side)

| quantity | seed-1 (v2, gate_eval_v1) | seed-2 (v3, gate_eval_v2) |
|---|---|---|
| NON-HARM verdict | **PASS** | **PASS** |
| mechanism fired | True (lat 0.0 s ≤180) | True (lat 0.0 s ≤180) |
| PRIMARY Δlc mean | +0.112 | +0.152 |
| Δlc sd | 0.353 | 0.537 |
| W / L | 9W / 7L | 9W / 7L |
| pos-tail p | 0.0704 | 0.0933 |
| harm-tail p_neg | 0.9440 | 0.9194 |
| worst-game Δlc vs −1.0 cap | cn04 −0.5 (within) | lf52 −0.9 (within) |
| secondary Δlog1p mean (p) | −0.098 (0.8643) | −0.033 (0.6068) |

Both seeds: mechanism fires, Δlc mean positive, harm-tail p ≫ 0.05 (no significant harm),
worst-game within the −1.0 catastrophic cap. Non-harm holds on both seeds.

---

## 4. Entry-gate (arm B) preflight — single-diff-invariant ALLOW (BLOCKER 2(b))

Arm B canary `canivel/arc3-duck-gate` v1 build status = **COMPLETE** (verified this session).
Notebook pulled with `-m` to `runs/tmp_pullback_duckgate_v1post_nb/`.

**Pin verified before use** (`sha256sum`):
`runs/fork_diff_boristown/cells/boris_16_gatebody.txt` →
`37e3018108c7058ce3295dd27d1a3baa3548a311d28f3e99c43be131bd99078b` (582 bytes) —
**matches the required pin sha 37e30181…078b** (this is THE seal pin; the earlier
`boris_16_wait_body.txt` sha 9755ac54 is superseded per ITERATION_LOG 07-31).

Command (run against the PUSHED slug — no `--local-notebook` — so the T4 COMPLETE-status
leg is exercised, discharging the full fire-condition 2(b), which the staged §3 run in
`preflight_singlediff_ext_2026-07-30.md` had left as WARN/SKIPPED):

```
uv run python scripts/preflight.py \
  --mode trusted-fork \
  --kernel canivel/arc3-duck-gate \
  --upstream notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb \
  --max-diff-cells 1 \
  --pin runs/fork_diff_boristown/cells/boris_16_gatebody.txt \
  --pin-sha 37e3018108c7058ce3295dd27d1a3baa3548a311d28f3e99c43be131bd99078b
```

Verdict verbatim (exit code 0):

```
Preflight for canivel/arc3-duck-gate vlatest: ALLOW
  fails: 0, warns: 0
  [OK] T1: pulled canivel/arc3-duck-gate
  [OK] T2: staged local tufa-labs-duck-harness-june-30-milestone-winner.ipynb
  [OK] T3: audited single-cell graft OK: 1 inserted code cell(s) (<= 1), 0 deleted, 0 rewritten, 1 banner-only additive edit(s); each inserted cell contains pinned byte-span sha256=37e3018108c7058ce3295dd27d1a3baa3548a311d28f3e99c43be131bd99078b
  [OK] T4: latest build COMPLETE
```

JSON contract: `"n_fail": 0, "n_warn": 0, "verdict": "ALLOW"`. This is the operative
gate the prereg fire-condition **§7.2(b)** requires, now with the T4 COMPLETE leg
satisfied against the real pushed+COMPLETE slug (stronger than the staged §3 WARN).

---

## 5. Discharge verdicts

- **ENTRY-GATE #1 (2-seed gate-eval markers + non-harm screen) — DISCHARGED.**
  Cites prereg **BLOCKER 3** and fire-condition **§7.2** ("entry-gate #1 (2-seed
  live-firing log + non-harm screen vs runs/null10) discharged"), and readiness-note
  **§6** (four markers, both seeds, latency ≤180 s) + **§7** (non-harm PASS both seeds).
  Evidence: both seeds COMPLETE; M1–M4 + GPU string green on both logs; latency 0.0 s;
  non-harm screen PASS on both (harm-tail p 0.9440 / 0.9194, worst-game −0.5 / −0.9 within
  the −1.0 cap).

- **ENTRY-GATE #2 (arm B single-diff preflight ALLOW) — DISCHARGED.**
  Cites prereg **BLOCKER 2(b)** and fire-condition **§7.2** ("a `preflight.py` extension
  `--max-diff-cells 1 --pin <boris_16 sha>` … AND its build status is COMPLETE").
  Evidence: preflight verdict **ALLOW**, T1–T4 all OK (T4 = latest build COMPLETE),
  pin sha 37e3018108c7058ce3295dd27d1a3baa3548a311d28f3e99c43be131bd99078b verified,
  exit 0, n_fail 0 / n_warn 0.

Both blocking entry gates for the boristown readiness-gate A/B are discharged. Remaining
fire conditions are governance, NOT entry evidence: §7.1 (git-commit seal of the prereg)
and §7.3 (Sunday-panel R23 2026-08-02 ratification). This report supplies the evidence
those steps consume; it does not itself seal, ratify, queue, push, or submit.

---
*Prepared 2026-08-02, read/verify only. kaggle==2.0.0 CLI. No push, no submission, no queue
change, $0 cloud. Screen artifacts: `runs/gate_eval_v2/screen_report.md` + `screen_raw.json`.*
