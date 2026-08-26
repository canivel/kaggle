# Rules Verification: Internet / External-API Access in Scored Submissions
**Competition:** `arc-prize-2026-arc-agi-3` (Kaggle)
**Date verified:** 2026-07-28 (live browser session, primary sources)

## VERDICT
**NO. Internet access is disabled in scored submissions — mechanically enforced by Kaggle — so external API model calls (OpenAI/Anthropic/Gemini/etc.) are impossible in this competition. Our offline-only strategy assumption is correct.**

---

## Evidence 1 — Kaggle Code Requirements (primary source, mechanical enforcement)
Source: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview (Code Requirements section), read live 2026-07-28.

> "Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:
> - CPU Notebook <= 9 hours run-time
> - GPU Notebook <= 9 hours run-time
> - **Internet access disabled**
> - Freely & publicly available external data is allowed, including pre-trained models
> - Submission file will be automatically generated."

This is not merely a policy — it is mechanical rejection (Method 4): the Submit button is **not active** for a kernel committed with internet enabled.

Additionally, the "Upgraded accelerators" section on the same page:

> "**No Internet** - To help ensure these machines are reserved for competition use, all RTX sessions must have internet disabled."

So on the RTX 6000 pool (which all serious submissions use), internet is disabled even at session level, not just at submit time.

## Evidence 2 — Kernel metadata: winner and our own submissions run offline with local weights
Milestone-1 winner's public kernel, pulled to `runs/winner_pulls/duck_public/kernel-metadata.json`:

```json
"id": "jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner",
"enable_internet": false,
"dataset_sources": [
  "driessmit1/arc3-vllm-h100-wheelhouse-v3",
  "jeroencottaar/taaf-kaggle-source-share",
  "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
],
"machine_shape": "NvidiaRtxPro6000"
```

The June 30 milestone **winner** runs `enable_internet: false` and attaches a full local HF snapshot of Qwen3.6-27B-FP8 plus a vLLM wheelhouse dataset (offline pip). If API calls were permitted, nobody would ship tens of GB of local weights and offline wheels. Our own submission kernels (e.g. `notebooks/duckwar/kernel-metadata.json`) are identical in this respect: `enable_internet: false`.

## Evidence 3 — Rules page (legal text)
Source: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules, Section 2.6 "EXTERNAL DATA AND TOOLS":

> "a. You may use data other than the Competition Data ("External Data") **to develop and test your Submissions**. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants... b. The use of external data and models is acceptable unless specifically prohibited by the Host... their use must be 'reasonably accessible to all' and of 'minimal cost'."

Note: this permits external data/models **during development** (e.g., open-weight checkpoints packaged as datasets). It does not override the Code Requirements' "Internet access disabled" condition at scoring time. Section 15 "INTERNET" is unrelated boilerplate (website liability).

## Evidence 4 — Discussion forum: host confirms the offline regime; no rule change
Searched discussion for "internet" and "API" — no thread announces any relaxation. Key thread, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/688481 ("Clarification on using local open-weight Qwen3.5 in ARC-AGI-3"):

- OP asks to confirm usage where "The model is run fully offline during Kaggle evaluation / No external API / internet access is used" and refers to "the standard no-internet rule".
- **Greg Kamradt (COMPETITION HOST)**: "Yes, this is fine to use. If you end up going this route, please consider making a bare bones template w/ Qwen3.5 model hooked up that others can build off of."
- A commenter's follow-up asking whether "GPT which require gpt api can be used" has **no host reply** endorsing API use.

The host's affirmative answer is specifically conditioned on the fully-offline, local-open-weight framing.

## Evidence 5 — The "Opus 5 30.2%" number is the ARC Prize ORG's own API eval, not this competition
Source: https://arcprize.org/leaderboard, read live 2026-07-28:

> "Claude Opus 5 (High) | Anthropic | 2026-07-24 | CoT | 97.5% | 88.3% | **30.2%** | $1.45 | **$20.7K** | ..."

That row sits on arcprize.org's frontier-model leaderboard — an eval the ARC Prize organization runs itself against commercial APIs, at ~$20.7K total V3 cost. The same page explicitly separates the regimes:

> "**Kaggle Systems** solutions showcase competition-grade submissions from the Kaggle challenge, operating under **strict computational constraints ($50 compute budget for 120 evaluation tasks)**. These represent purpose-built, efficient methods specifically designed for the ARC Prize."

Conflating the org's API-regime leaderboard with the Kaggle competition regime is the likely source of the challenge to our assumption. They are different evaluations with different rules and ~400x different compute budgets.

---

## Bottom line
- Scored Kaggle submissions: **internet off, no external API calls possible** — enforced by the platform (Submit button inactive otherwise) and by the RTX session policy.
- External *models* are allowed only as locally-attached open weights (datasets/models), which is exactly what the milestone winner and we do.
- Frontier-API numbers (e.g., Opus 5 at 30.2%) come from arcprize.org's own eval regime and have no bearing on what a Kaggle submission may do.
