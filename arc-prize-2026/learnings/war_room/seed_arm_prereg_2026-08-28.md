# SEED ARM — PRE-REGISTRATION (SEALED before push, 2026-08-28)

**Kernel:** `canivel/arc3-q38-seed-eval` (fresh slug, `feedback_fresh_kernel_slug`)
**Vehicle:** certified field floor `arc3-q38-field-eval`, byte-identical except cell 5.
**Change:** inject `'LOCAL_ANALYZER_SEED': '20260828'` into the TAAF setup env.
**Origin:** `yocybercode`/Thuitanium `thui-v1-1`, read off the public kernel and
reported in `learnings/community/brief_2026-08-28.md` (decision handoff 1).

## 1. WHAT THIS ARM IS — AND IS NOT

It is **NOT a score-improvement arm** and must never be read as one. It is a
**variance-decomposition instrument**. Our certified floor draws
1.59/1.58/1.63/1.16/1.92/1.14/1.26 -> **n=7, mean 1.4686, sd 0.2897**. That sd is
why nothing we have built is distinguishable at n=1 (lc MDE 11.1). Two competing
explanations for it, both on the record from 08-28:

- **SAMPLER variance** — the 27B's own stochastic decoding. Pinning the seed removes it.
- **SCHEDULER variance** — Spen, topic 732854 [V-doc]: *"depending on scheduling and
  timeouts of the game workloads, certain games get more time than others."* Pinning
  the seed does nothing to it.

Nothing we own can currently tell these apart. A seeded replicate pair can.

## 2. TREATMENT-CAN-FIRE — PROVEN BEFORE THE BUILD

`feedback_verify_treatment_can_fire`, and the 5-for-5 affordance class that
`scripts/affordance_audit.py` was written for on 08-28. Chain verified by direct
read of the pinned bundle source, **before** a slot was spent:

```
LOCAL_ANALYZER_SEED (setup env)
  -> tool_agent.py:159   _LOCAL_ANALYZER_SEED = _get_env_int("LOCAL_ANALYZER_SEED", -1)
  -> tool_agent.py:1536  build_chat_payload(seed=_LOCAL_ANALYZER_SEED)
  -> openai_compat.py    if provider == "vllm" and seed is not None and seed >= 0:
                             payload["seed"] = seed
```

Our provider is `vllm`. The default is **-1**, so **no seed reaches the wire today**;
a non-negative int puts it there. The key is **ABSENT** from
`setup_commands.json` (count 0, anchor count 1), so this is an INJECTION.

## 3. TEETH (adopted from the ladder, brief item 4)

The kernel fails **at setup, before the benchmark starts**, if:
its anchor is missing or duplicated; the bundle already sets a seed (the ABSENT
premise gone stale); injections != 1; or any of six untested analyzer variables
drifted (`TEMPERATURE 0.6`, `YIELD_SECONDS 60`, `MAX_OUTPUT 0`, `TOOL_STEPS 0`,
`TOP_P 0.95`, `TOP_K 20`, `ENABLE_THINKING true`). All six refusals are exercised
as negative controls in `duck_eval/seed/test_seed_graft.py` (18/0):
**a guard that has never refused may be one that cannot** (`feedback_guard_never_fired`).

## 4. THE SEALED READ

**Primary statistic:** sample sd of board score across seeded draws, ALL at seed
`20260828`, against the floor comparator sd **0.2897 (n=7)**. One-sided F-test,
alpha 0.05, H0 = "seed pinning does not reduce draw variance".

**Commitment, made BEFORE the first draw** (`feedback_seed_vs_own_config`; Scott
Le Grand's *"at least 4 before I remotely believe differences"*): **n=4 seeded
draws, or the arm is not started.** A 2-draw read is FORBIDDEN and no verdict may
be written before n=4. Draws need not be consecutive; the floor keeps the
queue on any night this arm is not ready.

**Declared MDE, honestly.** F(3,6) crit ~ 4.76, so n=4-vs-7 detects only a
variance ratio >= 4.76 — a **halving of sd (to <= 0.133)**. This arm can detect a
LARGE collapse and nothing subtler. A null result therefore means *"no large
collapse"*, **never** *"seed is inert"*.

**Branch on the result:**
- **COLLAPSE** (seeded sd <= 0.133): sampler variance was dominant. Seeded draws are a
  DIFFERENT config from the unseeded floor and are pooled separately under
  `project_arc_final_selection_rule`. Every future arm re-prices its required n downward.
- **NULL** (no significant reduction): scheduler variance dominates, Spen is right,
  and the field's n>=4 is a hard floor for us too. Seeded and unseeded floor draws
  are poolable as one config. This closes seed-pinning as a lever permanently.

**Score is NOT the read.** A high or low draw decides nothing here. If any seeded
draw exceeds our banked 1.92 that is a public-max artifact, not evidence
(`never gate on public-LB single draws`).

## 5. PRE-REGISTERED LIMITATION (stated now, not discovered later)

vLLM's per-request `seed` does **not** guarantee determinism under continuous
batching: 25 concurrent games share one server, and batch composition changes
numerics. So even under the SAMPLER hypothesis the collapse may be **attenuated**.
This weakens H1 only; it cannot manufacture a false collapse. If the read comes
back NULL, this limitation is a live alternative explanation and the verdict must
say so rather than claiming sampler variance was refuted.

## 6. KILL / STOP

No kill clause: this arm cannot "fail", it can only return NULL or COLLAPSE, and
both are informative. It is abandoned only if the TEETH refuse at setup (which
means the bundle moved and the arm's premise is void -> VOID, not REJECT).

## 7. EVIDENCE ON FILE AT SEAL TIME

- `scripts/local_gate.py --arm q38-seed` -> **PASS 55/0** (warn 2, skip 2)
- `scripts/local_gate.py --self-test` -> **PASS 13/0** after the N3 widening
- `scripts/local_gate.py --arm q38-field` -> **PASS 52/0** (no sibling regression)
- `duck_eval/seed/test_seed_graft.py` -> **18/0**, 6 of them negative controls
- N4 determinism: builder rebuilt twice -> byte-identical
- preflight structural (duck-harness family): 6 ok, 0 warn
