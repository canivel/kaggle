# A17 v7 seed-2 Concordance Note (MEASUREMENT ONLY)

- **Date:** 2026-07-30
- **Kernel:** `canivel/arc3-a17-72b-canary` v7, **seed=2** (confirmed: `A17_CANARY_SEED=2` in `a17_vllm_cmd.json`; kernel log `A17-CANARY seed=2 mode=throughput-canary-v6-dataset-weights`).
- **Artifact set:** `runs/kernel_pulls/a17_v7/`. Gate-look JSON: `runs/a17_v7_gate_look_2026-07-30.json`.
- **Scope:** Raw counts + concordance verdict only. Branch decision is governed arithmetic; no strategy content.

## Seed-2 measurement

| metric | v7 seed-2 | v6 seed-1 |
|---|---|---|
| SigmaN_72B (executed actions) | **5** | 5 |
| per-game N | vc33 4, sb26 1, ft09 0, lp85 0 | ft09 2, sb26 1, vc33 2, lp85 0 |
| games present / state | 4/4, all gave_up | 4/4, all gave_up |
| window_s (max drift) | 7920.0-7922.8 (0.036%) | 7920.0-7928.6 |
| LLM calls (meta blocks) | 1008 | 1044 |
| HTTP non-200 | 0 / 1011 | 0 / 1047 |
| finish_reason=stop / length | 1004 / **4** (all ft09) | 1044 / 0 |
| native tool-call parses | **0 / 1008** | 0 / 1044 |
| fenced-recovery parses | 7 | 8 |
| step_executed=True | 5 | 5 |
| actions_total freeze | =5 at t=720 s, held to t=7920 s | =5 at t=720 s, held to t=7920 s |
| turn_time_budget yields | 400 | ~412 |

## Mechanism evidence (concordance)

- **Engine healthy throughout:** 1011 requests all HTTP 200, gen_tps 14-98 tok/s, running_reqs=4, stall_s=0 to window end. Same as v6 — no stall, no deadlock.
- **Format non-compliance:** native hermes tool-call emission = **0/1008** (0.000), identical to v6's 0/1044. 1001/1008 responses carried neither a native call nor recoverable markup.
- **Degenerate byte-identical repetition:** late vc33 responses show `content_chars: 988` constant across turns (v6 showed the same signature at 919 chars per game) — model re-emits identical text under corrective re-prompt.
- **Fenced-recovery is the only action path:** all 5 executed actions came via the fenced fallback (7 recovered tool calls, ids `fenced-call-N`), all within the first **~96 s** (vc33 +15.7/+42.1/+60.8/+96.4 s, sb26 +20.7 s); infinite gap thereafter.
- **New-but-immaterial detail:** 4 `finish_reason=length` in ft09 (v6 had 0). This is 0.40% of calls and ft09 still executed 0 actions — not a truncation loop, does not change the mechanism.

## Concordance verdict

**CONCORDANT.** Two independent seeds (1 and 2), zero parameter/prompt/harness change between them (vLLM cmd + `taaf_setup_env.json` byte-identical, only `A17_CANARY_SEED` 1->2), both produce SigmaN=5 via the same deterministic format-non-compliance livelock: 0/N native tool calls, freeze at t=720 s, byte-identical re-emission, actions only in the first ~90-96 s via fenced recovery. The per-game trajectory shifted (ft09 2->0, vc33 2->4) exactly as expected under a seed change at temperature 0.6; the livelock **mechanism** is invariant, which is the claim under test — not byte-identical trajectories.

## What B2a formally closes

- **SigmaN = 5 < 138 -> branch B2a fires.** rho_action = 480/5 = 96.0 > 3.5 kill line.
- **72B route DEAD.** No third seed. No fix lane (tool-call format contract is sealed and not permitted to change between seeds; native-emission rate is a property of the Qwen2.5-VL-72B-AWQ x hermes-parser contract, reproduced across both seeds).
- **NC-10 DISCHARGED** (k=2 concordant). NC-4 stays discharged (second 1008-call parse study corroborates). NC-9 satisfied (sealed pre-observation). NC-12 unchanged (metadata-level parity note, does not gate B2a).
- **Priority reverts** to the boristown readiness-gate A/B lane (prereg draft commit 382b52e).
