# AIMO Progress Prize 3 - Competition Log

## Competition Info
- **Metric**: Exact-match accuracy on 50 public / 50 private IMO-level problems
- **Prize**: $2.2M+ | **Entry**: 2026-04-08 | **Final**: 2026-04-15
- **Limit**: 1 submission/day — ~12 total remaining as of Apr 3
- **Top score**: 46/50 | **Our best**: 39/50 (v34/Kaggle-v34) | **Target**: 44+

## Score History
| Version | Score | Config | Key Change |
|---------|-------|--------|------------|
| v13 | **1** | Wrong API (chat/completions) | Fatal: wrong protocol |
| v18 | **38** | Harmony protocol, temp=1.0, simple prompt | Fixed Harmony → +37 |
| v25 | **32** | huikang + mandatory verification + reliability voting | Huikang broke it |
| v34 (Kaggle v34) | **39** | 5-step prompt + T=1.0 + 43/50 vLLM params + follow-up | Param mismatch vs 44/50 |
| v35 (Kaggle v35, running) | pending | **EXACT 44/50 params** (ctx=65536, batch=256, gpu=0.96) + T=0.8 + follow-up | Fix param mismatch |

## Ready Notebooks (all syntax-validated)
| Local file | Push dir | Kaggle ver | Config |
|-----------|---------|-----------|--------|
| submission_v16_exact44.ipynb | push_v16/ | v35 (running) | exact 44/50 params + T=0.8 |
| submission_v17_verify.ipynb | push_v17/ | v36 | + binary verify cascade |
| submission_v18_twophase.ipynb | push_v18/ | v37 | + two-phase + domain hints (RISKY) |
| submission_v19_twophase_nodomain.ipynb | push_v19/ | v38 | + two-phase (no domain) + disagree ctx |
| submission_v20_eagle3.ipynb | push_v20/ | v39 | + Eagle-3 speculative decoding |

## Architecture
GPT-OSS-120B (or huikang fine-tuned) via vLLM subprocess → Harmony protocol → 
16 Jupyter kernels → 8-12 parallel attempts → entropy-weighted voting

## Key Learnings (stored in kaos DB: aimo3-learnings.db)

### [CRITICAL] Harmony protocol = +37 points
chat/completions → 1/50. Harmony completions with native tools → 38/50.

### [HIGH] 5-step structured prompt = +1 point (43→44)
UNDERSTAND→EXPLORE→PLAN→EXECUTE→VERIFY. Both 44/50 notebooks use this.

### [HIGH] Mandatory Python verification
"NEVER give \\boxed{} without code verification" — forces model to verify.

### [HIGH] Reliability-weighted voting
weight = (1/entropy) × (1/(1+errors)) × (1.2 if clean_code)

### [MEDIUM] Temperature: 0.5 works but so does 1.0
Both score 44/50. 0.8 is the middle ground for more diversity.

### [MEDIUM] jupyter_timeout=10 (was 6)
Complex sympy computations need more time.

### [MEDIUM] Docker image controls path mounting
Custom docker → model at /kaggle/input/gpt-oss-120b/
No docker → model at /kaggle/input/models/danielhanchen/

### [LOW] Complex entropy weighting hurts
Simple mean entropy beats position-weighted + variance + streak.

## Submission Plan (1/day limit, 12 remaining)
| Day | Kaggle ver | File | Expected | Key Change |
|-----|-----------|------|----------|------------|
| Apr 3 | v35 ✓ | push_v16/ | ~44 | Exact 44/50 params (param mismatch fix) |
| Apr 4 | v36 ready | push_v17/ | 44-45 | + Binary verify cascade (amanatar 44/50) |
| Apr 5 | v38 ready | push_v19/ | 44-46 | + Two-phase (no domain) + disagree ctx + retry |
| Apr 6 | v39 ready | push_v20/ | 44-47 | + Eagle-3 speculative decoding (+36% speed) |
| Apr 7-15 | v40+ | TBD | best | Repeat best config + novel ideas |

**Contingency:**
- v35 < 42: diagnose vLLM logs before burning more submissions
- Eagle-3 fails: fall back to push_v19 (v38 two-phase)
- v37 (push_v18) kept as backup but risky (domain hints = -3.7 pts per CRITICAL_REVERSAL_2)

## Key Bug Found (Apr 3)
v34 scored 39 (not 44) because we used "43/50 vLLM params" (ctx=81920, batch=64, gpu=0.99,
extra flags) with the 5-step prompt designed for the 44/50 setup (ctx=65536, batch=256, gpu=0.96).
The param mismatch likely caused slower inference and suboptimal batching.
v35 fixes all params to exact 44/50 values.
