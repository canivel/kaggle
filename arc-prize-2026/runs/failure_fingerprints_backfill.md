# Failure-fingerprint backfill — campaign history validation

Generated 2026-08-30 by scripts/fingerprint_backfill.py. Store: `runs/failure_fingerprints.json` (23 incidents).

Two-tier design (Kimi-3 review cycle, corrected adopt #3): tier 1 = build-rail logs (rich: stage marker + normalized error; silent deaths keyed by last-progress marker + elapsed bucket); tier 2 = scored-rail submissions (coarse: slug, version, status class, score class, docker/machine). **Scored reruns are hidden executions — no logs exist, ever**; tier 2 never pretends otherwise.

## 1. Incident inventory

| id | date | tier | kernel | ver | status | score | fingerprint | confidence | source |
|---|---|---|---|---|---|---|---|---|---|
| inc-t2-001 | 2026-03-29 | 2 | canivel/arc3-cnn-frame-change-agent | 6 | COMPLETE | 0.00 | `5a2f649465c44d0a` | high | docs/competition_plan.md:8 |
| inc-t2-002 | 2026-04-24 | 2 | canivel/arc3-forge35 | - | COMPLETE | 0.00 | `d74bf4b027fa26e8` | low | memory/feedback_lock_deadlock.md |
| inc-t2-003 | 2026-05-26 | 2 | canivel/arc3-final | 26 | ERROR | none | `55dfdd5a95645722` | high | ITERATION_LOG.md |
| inc-t2-004 | 2026-06-01 | 2 | canivel/arc3-final | 30 | COMPLETE | null-band | `b4a09546d4d6b52c` | high | ITERATION_LOG.md |
| inc-t2-005 | 2026-06-08 | 2 | canivel/arc3-final | 36 | COMPLETE | null-band | `a84cac8feb056fc8` | high | ITERATION_LOG.md |
| inc-t2-006 | 2026-06-10 | 2 | canivel/arc3-final | 38 | COMPLETE | 0.00 | `bb2ea91b6f17537e` | high | memory/feedback_test_before_submit.md |
| inc-t2-007 | 2026-06-20 | 2 | canivel/arc3-forge62 | 4 | ERROR | none | `2a965c7a5e896833` | high | runs/submission_log.jsonl |
| inc-t2-008 | 2026-06-21 | 2 | canivel/arc3-forge35 | 1 | ERROR | none | `892bce56f212dfdb` | high | runs/submission_log.jsonl |
| inc-t2-009 | 2026-06-22 | 2 | canivel/arc3-forge35 | 1 | ERROR | none | `892bce56f212dfdb` | high | runs/submission_log.jsonl |
| inc-t2-010 | 2026-06-26 | 2 | canivel/arc3-jepa-v2 | 1 | ERROR | none | `b25263aca4783cca` | high | runs/submission_log.jsonl |
| inc-t2-011 | 2026-06-27 | 2 | canivel/arc3-execwm | 1 | ERROR | none | `61e5e32471003756` | high | runs/submission_log.jsonl |
| inc-t2-012 | 2026-06-28 | 2 | canivel/arc3-execwm-v2 | 1 | ERROR | none | `ddba324f535c18bf` | high | runs/submission_log.jsonl |
| inc-t1-001 | 2026-07-07 | 1 | canivel/arc3-pilot-eval | 1 | ERROR | none | `07d0f5248c48401d` | low | campaign |
| inc-t1-002 | 2026-07-07 | 1 | canivel/arc3-pilot-eval | 2 | ERROR | none | `07d0f5248c48401d` | low | campaign |
| inc-t1-004 | 2026-07-07 | 1 | canivel/arc3-duck-repro | 1 | ERROR | none | `fbfaf1cf21edfd8a` | medium | runs/submission_log.jsonl |
| inc-t1-003 | 2026-07-08 | 1 | canivel/arc3-pilot-eval | 3 | ERROR | none | `07d0f5248c48401d` | low | campaign |
| inc-t1-005 | 2026-07-25 | 1 | canivel/arc3-a17-72b-canary | 1 | ERROR | none | `fb1e96c3815797ad` | high | runs/kernel_pulls/a17_canary_v1/arc3-a17-72b-canary.log |
| inc-t1-006 | 2026-07-25 | 1 | canivel/arc3-a17-72b-canary | 2 | ERROR | none | `fb1e96c3815797ad` | high | runs/kernel_pulls/a17_canary_v2/arc3-a17-72b-canary.log |
| inc-t1-007 | 2026-08-14 | 1 | canivel/arc3-b122-boot-canary | 1 | ERROR | none | `b1cb06f2668b9b67` | high | runs/kernel_pulls/b122_v1/arc3-b122-boot-canary.log |
| inc-t1-008 | 2026-08-14 | 1 | canivel/arc3-lora-serve-canary | 1 | ERROR | none | `d666b385aca6f787` | high | runs/kernel_logs/lora_serve_canary_v1.log.json |
| inc-t1-009 | 2026-08-15 | 1 | canivel/arc3-q38-engine-eval | 1 | ERROR | none | `d4c8e135c7104f91` | high | runs/kernel_pulls/q38_v1/q38.log |
| inc-t1-010 | 2026-08-17 | 1 | - | - | ERROR | none | `4adc306b8ab3107d` | high | runs/kernel_pulls/q38low_v1/q38low.log |
| inc-t1-011 | 2026-08-18 | 1 | canivel/arc3-graft-floor-eval | - | ERROR | none | `474fe5420b4aa57e` | high | runs/kernel_pulls/graft_floor_v2/arc3-graft-floor-eval.log |

Tier-1 log scan: 57 retained build logs scanned, 7 contained failure signals (all retained pulls are COMPLETE eval builds — the scored-rail deaths above have no logs by construction, which is exactly why tier 2 exists).

## 2. Q1 — family collapse

23 incidents collapse into **19 distinct fingerprints** and **11 recurring families (n>=2)** (8 candidate-matchable, 3 report-only class families). Every family key of every incident:

| family | n | first | last | WARN active after death # | matchable pre-submission | incidents |
|---|---|---|---|---|---|---|
| `class:ERROR:none` | 7 | 2026-05-26 | 2026-06-28 | 2 | no (report-only) | inc-t2-003, inc-t2-007, inc-t2-008, inc-t2-009, inc-t2-010, inc-t2-011, inc-t2-012 |
| `provenance:scratch-built` | 5 | 2026-05-26 | 2026-06-28 | 2 | yes | inc-t2-003, inc-t2-007, inc-t2-010, inc-t2-011, inc-t2-012 |
| `slug:canivel/arc3-final` | 4 | 2026-05-26 | 2026-06-10 | 2 | yes | inc-t2-003, inc-t2-004, inc-t2-005, inc-t2-006 |
| `class:COMPLETE:0.00` | 3 | 2026-03-29 | 2026-06-10 | 3 | no (report-only) | inc-t2-001, inc-t2-002, inc-t2-006 |
| `slug:canivel/arc3-forge35` | 3 | 2026-04-24 | 2026-06-22 | 3 | yes | inc-t2-002, inc-t2-008, inc-t2-009 |
| `slug:canivel/arc3-pilot-eval` | 3 | 2026-07-07 | 2026-07-08 | - | yes | inc-t1-001, inc-t1-002, inc-t1-003 |
| `t1:07d0f5248c48401d` | 3 | 2026-07-07 | 2026-07-08 | - | yes | inc-t1-001, inc-t1-002, inc-t1-003 |
| `class:COMPLETE:null-band` | 2 | 2026-06-01 | 2026-06-08 | 2 | no (report-only) | inc-t2-004, inc-t2-005 |
| `slug:canivel/arc3-a17-72b-canary` | 2 | 2026-07-25 | 2026-07-25 | 2 | yes | inc-t1-005, inc-t1-006 |
| `t1:fb1e96c3815797ad` | 2 | 2026-07-25 | 2026-07-25 | 2 | yes | inc-t1-005, inc-t1-006 |
| `t1root:4f70850343550dae` | 2 | 2026-07-25 | 2026-07-25 | 2 | yes | inc-t1-005, inc-t1-006 |
| `slug:canivel/arc3-b122-boot-canary` | 1 | 2026-08-14 | 2026-08-14 | - | yes | inc-t1-007 |
| `slug:canivel/arc3-cnn-frame-change-agent` | 1 | 2026-03-29 | 2026-03-29 | - | yes | inc-t2-001 |
| `slug:canivel/arc3-duck-repro` | 1 | 2026-07-07 | 2026-07-07 | - | yes | inc-t1-004 |
| `slug:canivel/arc3-execwm` | 1 | 2026-06-27 | 2026-06-27 | - | yes | inc-t2-011 |
| `slug:canivel/arc3-execwm-v2` | 1 | 2026-06-28 | 2026-06-28 | - | yes | inc-t2-012 |
| `slug:canivel/arc3-forge62` | 1 | 2026-06-20 | 2026-06-20 | - | yes | inc-t2-007 |
| `slug:canivel/arc3-graft-floor-eval` | 1 | 2026-08-18 | 2026-08-18 | - | yes | inc-t1-011 |
| `slug:canivel/arc3-jepa-v2` | 1 | 2026-06-26 | 2026-06-26 | - | yes | inc-t2-010 |
| `slug:canivel/arc3-lora-serve-canary` | 1 | 2026-08-14 | 2026-08-14 | - | yes | inc-t1-008 |
| `slug:canivel/arc3-q38-engine-eval` | 1 | 2026-08-15 | 2026-08-15 | - | yes | inc-t1-009 |
| `t1:474fe5420b4aa57e` | 1 | 2026-08-18 | 2026-08-18 | - | yes | inc-t1-011 |
| `t1:4adc306b8ab3107d` | 1 | 2026-08-17 | 2026-08-17 | - | yes | inc-t1-010 |
| `t1:b1cb06f2668b9b67` | 1 | 2026-08-14 | 2026-08-14 | - | yes | inc-t1-007 |
| `t1:d4c8e135c7104f91` | 1 | 2026-08-15 | 2026-08-15 | - | yes | inc-t1-009 |
| `t1:d666b385aca6f787` | 1 | 2026-08-14 | 2026-08-14 | - | yes | inc-t1-008 |
| `t1:fbfaf1cf21edfd8a` | 1 | 2026-07-07 | 2026-07-07 | - | yes | inc-t1-004 |
| `t1root:3c9cffe6723129b8` | 1 | 2026-08-14 | 2026-08-14 | - | yes | inc-t1-007 |
| `t1root:46b9c105acd985ec` | 1 | 2026-08-15 | 2026-08-15 | - | yes | inc-t1-009 |
| `t1root:5c05453e9de08087` | 1 | 2026-08-14 | 2026-08-14 | - | yes | inc-t1-008 |
| `t1root:66b5ffe354c47bae` | 1 | 2026-08-17 | 2026-08-17 | - | yes | inc-t1-010 |

## 3. Q2 — where the recurrence WARN would have fired

Replay rule: before each submission/build, WARN if any candidate-matchable family (slug:, provenance:, t1:) of the candidate had **>=2 prior deaths** (low-confidence reconstructed incidents never count as evidence). Chronological result:

| incident | date | window | WARN active before it? | matching family (prior deaths) |
|---|---|---|---|---|
| inc-t2-001 (canivel/arc3-cnn-frame-change-agent v6) | 2026-03-29 | scored LB window | no | - |
| inc-t2-002 (canivel/arc3-forge35 v?) | 2026-04-24 | scored LB window | no | - |
| inc-t2-003 (canivel/arc3-final v26) | 2026-05-26 | scored LB window | no | - |
| inc-t2-004 (canivel/arc3-final v30) | 2026-06-01 | scored LB window | no | - |
| inc-t2-005 (canivel/arc3-final v36) | 2026-06-08 | scored LB window | **YES** | `slug:canivel/arc3-final` (2: inc-t2-003, inc-t2-004) |
| inc-t2-006 (canivel/arc3-final v38) | 2026-06-10 | scored LB window | **YES** | `slug:canivel/arc3-final` (3: inc-t2-003, inc-t2-004, inc-t2-005) |
| inc-t2-007 (canivel/arc3-forge62 v4) | 2026-06-20 | scored LB window | no | - |
| inc-t2-008 (canivel/arc3-forge35 v1) | 2026-06-21 | scored LB window | no | - |
| inc-t2-009 (canivel/arc3-forge35 v1) | 2026-06-22 | scored LB window | no | - |
| inc-t2-010 (canivel/arc3-jepa-v2 v1) | 2026-06-26 | scored LB window | **YES** | `provenance:scratch-built` (2: inc-t2-003, inc-t2-007) |
| inc-t2-011 (canivel/arc3-execwm v1) | 2026-06-27 | scored LB window | **YES** | `provenance:scratch-built` (3: inc-t2-003, inc-t2-007, inc-t2-010) |
| inc-t2-012 (canivel/arc3-execwm-v2 v1) | 2026-06-28 | scored LB window | **YES** | `provenance:scratch-built` (4: inc-t2-003, inc-t2-007, inc-t2-010, inc-t2-011) |
| inc-t1-001 (canivel/arc3-pilot-eval v1) | 2026-07-07 | build slot | no | - |
| inc-t1-002 (canivel/arc3-pilot-eval v2) | 2026-07-07 | build slot | no | - |
| inc-t1-004 (canivel/arc3-duck-repro v1) | 2026-07-07 | build slot | no | - |
| inc-t1-003 (canivel/arc3-pilot-eval v3) | 2026-07-08 | build slot | no | - |
| inc-t1-005 (canivel/arc3-a17-72b-canary v1) | 2026-07-25 | build slot | no | - |
| inc-t1-006 (canivel/arc3-a17-72b-canary v2) | 2026-07-25 | build slot | no | - |
| inc-t1-007 (canivel/arc3-b122-boot-canary v1) | 2026-08-14 | build slot | no | - |
| inc-t1-008 (canivel/arc3-lora-serve-canary v1) | 2026-08-14 | build slot | no | - |
| inc-t1-009 (canivel/arc3-q38-engine-eval v1) | 2026-08-15 | build slot | no | - |
| inc-t1-010 (? v?) | 2026-08-17 | build slot | no | - |
| inc-t1-011 (canivel/arc3-graft-floor-eval v?) | 2026-08-18 | build slot | no | - |

Ground-truth checks:

- **Structural-drift family** (`provenance:scratch-built`): 5 deaths (v45 05-26, v62 06-20, v63 06-26, v64 06-27, v65 06-28). WARN condition met at death #2 (v62) -> **3 subsequent deaths (v63, v64, v65) would have carried the WARN in advance.** The root cause was only found manually on 06-28, after death #5.
- **arc3-final slug family**: deaths v45 (ERROR), v30 (0.04), v36 (0.01), v38 (0.00). WARN condition met at death #2 (v30, 06-01) -> **2 subsequent deaths (v36 06-08, v38 06-10 missing-import 0.00) would have carried the WARN.**
- **arc3-forge35 slug ERRORs** (s12 06-21, s13 06-22): with the TIPS-deadlock attribution held at low confidence, only 1 high-confidence prior death existed before s13 -> **no WARN**; both windows burned unflagged. (Sensitivity: if the TIPS 0.00 is accepted as a forge35 death, s13 fires the WARN -> +1 flagged.) The 06-24 fresh-slug pivot is what the WARN would have recommended at death #2.
- **Pilot-eval IndexError family** (t1, v1-v3): identical normalized fingerprint all three times. Evidence held at low confidence (logs not retained), so the strict replay flags 0 of them; with the incidents taken at face value the WARN fires at death #2 and v3 (death #3) is flagged-in-advance.

## 4. Q3 — counterfactual: windows burned that would have carried a WARN

- **5 scored LB windows** were burned by deaths that a recurrence WARN would have flagged before submission: inc-t2-005 (canivel/arc3-final v36, 2026-06-08), inc-t2-006 (canivel/arc3-final v38, 2026-06-10), inc-t2-010 (canivel/arc3-jepa-v2 v1, 2026-06-26), inc-t2-011 (canivel/arc3-execwm v1, 2026-06-27), inc-t2-012 (canivel/arc3-execwm-v2 v1, 2026-06-28).
- **0 build slots** likewise (strict low-confidence rule; see pilot note above).
- Sensitivity (taking low-confidence attributions at face value): +1 scored window (forge35 s13) and +1 build slot (pilot-eval v3) -> **6 scored windows + 1 build slots** upper bound.
- The WARN is warn-only by design; the counterfactual claim is that these windows would have been submitted **with the prior incident references in hand** (e.g. "this family died v45+v62 already"), not that they would necessarily have been withheld. For the drift family that reference trail pointed at build_notebook.py 8 days and 3 burned windows before the manual root-cause hunt found it.

## 5. Limitations

- Scored-rail reruns are hidden: tier-2 fingerprints can never include stack traces; families are slug/provenance/class only.
- inc-t2-002 (TIPS deadlock) and inc-t1-001..003 (pilot IndexErrors) have no surviving local artifacts; they are recorded at low confidence and excluded from evidence counts.
- `kaggle competitions submissions` returned 403 during backfill (2026-07-18); scored-rail records come from runs/submission_log.jsonl + ITERATION_LOG.md.
- Kernel docker/machine metadata is read from the CURRENT kernel-metadata.json files; historical metadata drift is not reconstructed.

