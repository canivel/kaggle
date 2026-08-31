# ARC-AGI-3 DAILY BRIEF — 2026-08-31 (Monday; no panel, per the 07-27 cadence rule)

Collect (STEP 1a–1c) ran 06:00–06:27 and its full output is
`learnings/community/brief_2026-08-31.md` (2,651-row board archive, 195 discussion topics
paged to exhaustion, 3 new kernels pulled and read). This file is the merge + the day's
decisions; it does not repeat the community brief's tables.

---

## 1a. RESULT DEEP-DIVE — TV28 fork draw 2 = 1.36

**Not a gain, and not a loss either — it is two draws that bought no information.**

| config | n | mean | draws |
|---|---|---|---|
| TV28 fork (13 grafts live) | 2 | **1.4900** | 1.62, 1.36 |
| certified field floor | 8 | **1.5413** | — |

−0.05 against the floor. With sd ≈ 0.34 the SE on an n=2 mean is ≈ 0.24, so the gap is
**0.2σ**. The pre-registered expectation for draw 2 was explicitly *"make the config
readable"*, and the honest read is that at n=2 it still is not. This is exactly what the
panel's R29 FATAL rule anticipated (no kill below n=6/arm) and exactly what the stack's own
author reported (*"two byte-identical runs of v22 scored 1.82 and 0.00"*).

**Mechanism evidence pulled today** (`runs/bench_pull_0831/`, both bench kernels COMPLETE):

- `arc3-tv28-bench` (m0r0, 4 replicas): 4.76 / 0.00 / 0.00 / 0.00, mean 1.19. The
  instrument had already been REFUSED on dynamic range on 08-30 and this confirms it —
  m0r0 gives a three-zero read that cannot separate anything.
- `arc3-tv28-bench-sb26` (sb26, 4 replicas): arm A 2.78 / 4.36 (mean 3.57), arm B 2.78 /
  2.78 (mean 2.78). **n=2/arm. R29 bars a kill AND bars a promote. No verdict.**

**But the bench delivered something better than the verdict it was built for.** All four
m0r0 replicas ran the full 7920 s clock and generated **242,119 ± 2,100 tokens** — a **0.9%
coefficient of variation**. Generated tokens per game is very nearly a physical constant of
this rail. That makes throughput a *high-precision, n=1-certifiable* instrument, which is
what today's build is aimed at. It also re-confirms `feedback_decision_budget_binding` on
our own artifact for the second time: the clock binds, the action cap never does.

## 1b/1c. DISCUSSIONS + RESEARCH — one find dominates

Full adopt/adapt/ignore list is in the community brief's TOP 10. The headline:

**`romantamrazov/arc-real-agi-solution` (The AGI Boys, #22, 2.66 — the public ceiling) is a
public artifact whose own comments say the only thing it changed is FOUR vLLM SERVING
FLAGS**, on our exact wheelhouse (`driessmit1/arc3-vllm-h100-wheelhouse-v3`, vllm==0.19.0),
our exact model pin, our exact machine shape and docker image:
`--kv-cache-dtype fp8`, `--speculative-config {"method":"mtp","num_speculative_tokens":3}`,
`--async-scheduling`, and `--max-model-len 65536 → 262144`.

**Verified on our side today** [V]: our mounted bundle's vLLM argv ends at `--max-model-len
str(VLLM_MAX_MODEL_LEN)` with `VLLM_MAX_MODEL_LEN = 65536`, and none of the other three
flags appear anywhere. The gap is real and is ours.

→ **ADOPT — built and pushed today** (see §3). The attribution to 2.66 is **[V-doc] (the
author's own comment), never [V]**; that limit is written into the pre-registration so it
cannot be quietly upgraded.

Everything else: ignore or watch. Tong Hui Kang's public-game name table is a *public*-game
semantic prior and is exactly the overfit the private set punishes — explicitly NOT adopted.
The 21-team ≥+0.60 surge is substantially a released Kaggle submission backlog (topic
738216, resolved 02:26), not a field capability jump; median Δ/draw among gainers actually
*fell* 0.23 → 0.21.

## 1-legacy. BOARD

**#173 / 2,651 on an unchanged 2.05 — −20 ranks in 24 h**, gap to the #10 line widened
0.93 → 1.12 for a second straight day. Both control arms essentially flat (Tufa +0.04, Jack
Cole +0.00). cstl +1.52 → 7.51 on a single draw, artifact-dark.

---

## 2. INSTRUMENT CORRECTIONS MADE TODAY

1. **The "stale ledger" defect does not exist, for the second consecutive day.**
   The community brief again reported *"`runs/ledger.json` `latest_date` is 2026-08-20 and
   eleven draws have landed since"*. Re-derived from the Kaggle API today
   (`uv run python scripts/ledger.py`): **n=37, mean 0.9316, s 0.1771, bar 1.089, latest
   2026-08-20 — byte-identical.** The ledger is the *frozen-fork null pool* by construction
   (`MEMBERSHIP_TAGS = ("frozen-fork filler", "frozen-fork sigma draw")`); the eleven
   intervening draws were other configs and correctly do not belong to it. The 08-30
   session already corrected this claim and the correction did not propagate — **the brief
   generator and ITERATION_LOG are not connected**, so a refuted finding can be re-reported
   indefinitely. Worth fixing at the generator, not by hand each morning.
2. **Pull-back verification on Windows produces FALSE drift on any non-ASCII notebook.**
   The pushed serving kernel compared "different" in 5 of 11 cells; every difference was
   the Kaggle CLI's pull-side cp1252 mojibake (`—` → `â€"`). After correcting the mojibake
   all 11 cells are byte-identical. Same instrument class as the community brief's find #7
   (the CLI's renderer dropping topic bodies). Any future session that pull-back-verifies
   and sees "cells differ" must apply `.encode('cp1252').decode('utf-8')` before concluding
   drift.
3. **Preflight's default baseline BLOCKs this family.** `--baseline
   canivel/arc3-duck-war-eval` returns BLOCK on a healthy kernel (17-cell baseline vs our
   11-cell field-floor lineage). Against the correct baseline
   (`canivel/arc3-q38-field-eval`) it returns **ALLOW** with D4 confirming the differing
   cell set is exactly `[5]`. This is the third recorded instance of
   `feedback_audit_the_instrument`'s "a gate that blocks a whole healthy family".

---

## 3. TODAY'S BUILD — A24 SERVING ARM (pushed 08:40, slot 1 of 2)

`canivel/arc3-serving-mtp3` v1. The certified field-floor notebook with **exactly one
changed cell (5)**, adding the four flags atomically to the bundled setup here-doc's vLLM
argv, with a fallback ladder to the byte-identical floor command and a launch manifest that
records the argv the child process was *actually* started with.

- CPU smoke `scripts/smoke_serving_mtp3.py`: **18/18 PASS**, including a diff assertion that
  the armed command differs from the floor command in exactly the 7 intended added lines
  and 1 removed line, that the patched child script still parses, and that a drifted or
  unobservable bundle causes the arm to **refuse to fire** rather than half-arm.
- Preflight (correct baseline): **ALLOW**, differing cells `[5]`.
- Pull-back: metadata byte-identical, `model_sources` retained, all cells match.
- Pre-registration sealed **before** the push:
  `learnings/war_room/serving_arm_prereg_2026-08-31.md`.

**Why this arm and not another graft:** every mechanism finding we own says the binding
quantity is generated tokens under a fixed 7920 s clock, and speculative decoding is the
only lever that buys tokens without touching the policy. It is a *serving* change — the
analyzer context window stays 32768, gameplay/prompt/tool policy untouched — so unlike every
graft screened this month it is architecture-neutral and should transfer to the private set.

**Sealed primary read** (see the prereg for the full bands): `generated tokens/sec (job
wallclock)` from the 25-game × 1-pass × 7920 s offline eval, against an n=3 same-config null
of **266.28 ± 10.31** (263.61 / 277.66 / 257.58). FIRES > 297.21; REFUTED < 276.60.
**Score is recorded and explicitly not gated on.** A REFUTED read is a live, expected
outcome — speculative decoding pays least when the server is already saturated, and our
comparator is at ~25 concurrent streams.

---

## 4. OPEN QUESTIONS

1. **Does the throughput gain, if it fires, convert into score?** Fireability ≠ effect
   (`feedback_verify_treatment_can_fire`, proven on `untried` at 72.7% fire / 0 gain). Even
   a certified +30% tok/s leaves this open, and one public draw cannot close it.
2. **Is the vLLM server watchdog worth its own build slot?** The 2.66 kernel ships a 15 s
   `/v1/models` poll with 3-strike kill-and-respawn and a partial-score preserve. Deferred
   today on purpose: its respawn path needs a launch manifest we only start writing *today*,
   and shipping it in the same kernel would confound the throughput read. **Tomorrow's first
   build item** — and today's manifest is the prerequisite it needs.
3. **Should the TV28 lane be retired outright?** The bench cannot kill it at n=2/arm and
   resolving it by drawing costs four more nights for an arm already reading below the
   floor. Recommendation: park it, do not buy draw 3. If it is ever redrawn,
   `thtennant/taaf-kaggle-source-share-fork` must be pinned to an explicit VERSION first —
   it has now republished on two consecutive days, so draws 1 and 2 may not even be the same
   agent, which would void the 1.49 mean as well.
4. **Nothing on this board has yet been explained by an artifact we can read**, except
   today's. cstl (7.51), Lord Han Solo, Tong Hui Kang, Franzen, Son Pham, Kopiczko — all
   artifact-dark. The one time a top-25 artifact became readable, it was a serving change.
   That is one data point, not a theory.
