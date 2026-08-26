# Fork diff: zoli800 1.39 vs canivel/arc3-duck-repro (frozen fork) — 2026-07-22

Pulled live via `kaggle kernels pull -m` (no push, no submit):

- Theirs: `zoli800/taaf-duck-harness-kaggle-share-resubmission-573a60` (LB 1.39, last run 2026-07-21)
- Ours: `canivel/arc3-duck-repro` v3 (last run 2026-07-06; band 0.76–1.33, mean ~0.97)
- Common upstream: `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`

## 1. Environment fields (kernel-metadata.json), side by side

| Field | zoli800 (1.39) | canivel/arc3-duck-repro v3 | Match |
|---|---|---|---|
| enable_gpu | true | true | YES |
| enable_tpu | false | false | YES |
| enable_internet | false | false | YES |
| machine_shape | NvidiaRtxPro6000 | NvidiaRtxPro6000 | YES |
| docker_image | gcr.io/kaggle-private-byod/python@sha256:57e612b4… | gcr.io/kaggle-private-byod/python@sha256:57e612b4… (same digest) | YES |
| competition_sources | arc-prize-2026-arc-agi-3 | arc-prize-2026-arc-agi-3 | YES |
| kernel_sources / model_sources | [] / [] | [] / [] | YES |
| is_private | false | true | n/a (not score-relevant) |

## 2. Dataset pins, side by side

| Dataset | zoli800 | ours | Last dataset update | Version drift? |
|---|---|---|---|---|
| jeroencottaar/taaf-kaggle-source-share (solver bundle) | pinned | pinned | 2026-06-12 | NO — both saved after 06-12, same latest version |
| driessmit1/arc3-vllm-h100-wheelhouse-v3 (wheels) | pinned | pinned | 2026-05-01 | NO |
| driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot (Qwen3.6-27B FP8) | pinned | pinned | 2026-05-17 | NO |

Our fork pins the **same vrfai Qwen3.6-27B FP8 snapshot** (not an older qwen3-27b-fp8 dataset), the same wheelhouse-v3, and the same source-share bundle. All three datasets were last updated **before** our v3 save (2026-07-06), so both kernels resolve identical dataset versions. There are no newer versions to "bump" to.

## 3. Notebook code diff

**Functionally byte-identical.** Full cell-by-cell diff:

- Markdown only: zoli replaced Cottaar's intro cell (Tufa Labs banner + writeup links) with a bare list of input links; ours keeps the original intro. Zero code effect.
- Code cells: identical except **one character** — an em-dash inside a `print()` string in the final diagnostics cell (ours has UTF-8 mojibake `â€”` from a past push encoding round-trip; zoli's has a clean `—`). Cosmetic; the line is a print literal in the diagnostics display cell.
- No deltas in: model snapshot paths, vLLM params, agent-loop params, timeouts, DATASET_SOURCES list, benchmark load, customization hook (both empty), run loop, teardown.
- Code sha256 (concatenated code cells): ours `b135bef5…`, zoli `767c3750…` — hash difference is entirely the one mojibake character.

## 4. Bottom line

**Zoli800's 1.39 is the same notebook, same env, same dataset versions, same model, drawn from the same stochastic distribution as our fork.** There is no config lever here. The 1.39 vs our 0.76–1.33 band is a lucky tail draw (his title — "Resubmission 573a60" — indicates he is doing exactly what we are: resubmitting the identical harness and sampling the seed lottery; 573a60 is likely a run/commit tag). 1.39 sits just above our observed max of 1.33 with n small; it is consistent with the same distribution, not evidence of a +0.06 systematic shift.

**Re-forking zoli's notebook or bumping pins is NOT a plausible systematic +0.06.** Expected value of a re-fork = another draw from the same ~0.97-mean band. Two marginal, non-config arguments for a re-fork anyway:

1. `feedback_fresh_kernel_slug` — a fresh slug is cheap insurance against hidden slug state on `arc3-duck-repro` (though that slug has been healthy: 0.76–1.33, no ERRORs).
2. Fixing the mojibake print is free hygiene, not a score lever.

If we choose to add a re-fork draw to the queue anyway (as a variance play, not an improvement play), the entry would be:

```json
{
  "kernel": "canivel/arc3-duck-zoli-refork",
  "version": 1,
  "file": "submission.parquet",
  "message": "re-fork of zoli800 1.39 resubmission (verified byte-identical code+env+pins to our frozen fork; pure variance draw on fresh slug)",
  "preflight_mode": "trusted-fork",
  "upstream": "zoli800/taaf-duck-harness-kaggle-share-resubmission-573a60"
}
```

But the honest recommendation is: **keep resubmitting the existing frozen fork** (or the fresh-slug clone) and spend effort elsewhere — the 1.39 confirms the duck-harness ceiling is reachable from our exact artifact; it does not reveal a better artifact.

*Analysis only — nothing pushed or submitted. Pulled copies in local temp (`zoli_pull/`, `ours_pull/`).*
