# Research Sweep — 2026-07-27 (weekly-ish; priority window Jul 25–27)

Context read: `learnings/daily_brief_2026-07-26.md` + `learnings/war_room/research_2026-07-26.md`
(already-processed set honored: Schema+Kamradt+HF-traces, 2607.12227, TTHE 2607.08124, 2606.24842,
Duck Harness, 2607.20972, 2607.07196, 2607.08964, SWE-Replay 2601.22129, BAGEN 2606.00198,
aTTT 2607.03441, TTA-via-Env 2511.04847, AERA 2605.25931, Rodionov 2605.05138/2607.15439/OPINE-World
2607.01531, WorldEvolver 2606.30639, EvoAgentBench 2607.05202, OCM 2607.02846 [ADAPT since 07-19,
priced R15/R16], Qwen tool-format fix set [ADOPT'd 07-26]).

**Window reality check.** 7 searches run. No new in-window (Jul 24–27) arXiv paper in our fields that
is not already in-set. arXiv hits surfaced today are June-or-older tail items. AWQ/72B serving: zero
new datapoints (all hits 2025-era issues already known); canary v3 empirically cleared the serving
envelope anyway (34.3 tps, 0 stalls), so the field is self-answered.

## FINDINGS (new-to-set only)

| # | Item | Date | Gist | Verdict |
|---|------|------|------|---------|
| 1 | **Continual Harness** — Seth Karten (Prime Intellect), substack writeup, amplified by @arcprize (~Jul 11 tweet) | ~mid-Jul, never captured in our sweeps | Reset-free self-improving agentic harness on ARC-AGI-3: foundation model stores memories, writes reusable skills, deploys subagents, refines its own prompt during play; claims efficiency via test-time learning of an internal world model | **ADAPT-low (intel)** — converges with the OCM/Schema executable-skill-bank direction we already track; no frozen-fork action now; read the mechanics (skill bank + prompt self-refinement) as design input for post-A17 war-v4; note ARC Prize *officially amplified* it — watch for it appearing in competitors' harnesses |
| 2 | COMAP 2606.02372 — co-evolving world models + agent policies | Jun | Joint WM/policy co-evolution loop for LLM agents | IGNORE for ARC (training-time co-evolution, off frozen-fork regime); side-note only: overlaps KAOS CORAL meta-harness interest |
| 3 | Text World Models 2606.09032 ("Bridging the Agent-World Gap") | Jun | Text WMs as planning substrate for LLM agents | IGNORE — text-world regime, no visual/interactive-grid transfer beyond what OPINE/Rodionov cluster already covers |
| 4 | SkillMaster 2605.08693 / LiveClawBench 2604.13072 / AgentAtlas 2605.20530 / TTS-of-agents 2602.18998 | Feb–May | General-agent skill mastery & benchmark papers | IGNORE — general/coding-agent benchmark regime, not our frozen-fork ARC line (same class as EvoAgentBench exclusion) |

## Status updates (already-tracked, no re-report)

- **Schema harness:** no technical update, still self-reported / HF traces frozen at 50 / ZERO
  independent replication. New: social amplification spike (Digg tracking ~242.5K views; HN thread
  48935905; benchmark-trust debate). Intel-only; standing skepticism posture unchanged. Kamradt's
  read ("non-zero human intelligence baked in, unconfirmed") stands.
- **ARC-AGI-3 model leaderboard:** Opus 5 (High) 30.2% remains top (PARK unchanged); GPT-5.6 Sol
  13.33% public / 7.78% semi-private. No new frontier datapoint since 07-24.
- **Qwen2.5-VL tool-calling:** searches only re-surfaced the known fix set (custom chat template
  merging VL + tool formats; `--enable-auto-tool-choice --tool-call-parser hermes`; issue #1093).
  Nothing supersedes yesterday's ADOPT (xgrammar `tool_choice="required"`) or the v4 fenced-recovery
  adapter. No action.

## PLAN IMPACT

**None material — plan stands.** Zero new in-window papers touch the A17 lane, the sealed gates, or
the frozen-fork filler. The single new-to-set finding (Continual Harness) is directional
confirmation, not correction: a third independent line (after Schema and OCM/Rodionov) converging on
"executable/reusable skill memory + in-play world-model refinement" as the winning ARC-AGI-3 harness
shape — which is exactly the generalization-first direction already ratified, and which A17's 72B
capacity bet is meant to feed. Two watch items: (1) ARC Prize's amplification of Continual Harness
raises the odds competitors adopt reset-free skill banking — mild urgency signal for the A17→war-v4
timeline, no gate change; (2) Schema's virality without replication reinforces our C3 discipline of
not chasing self-reported public-set numbers. Today's real work remains R20 ratification + canary v4
push, unaffected by anything found here.

Sources: [schema-harness.github.io](https://schema-harness.github.io/) · [HN 48935905](https://news.ycombinator.com/item?id=48935905) · [Kamradt on Schema](https://x.com/GregKamradt/status/2077949388673151332) · [Continual Harness substack](https://sethkarten.substack.com/p/continual-harness-an-efficient-self) · [@arcprize amplification](https://x.com/arcprize/status/2072069184146833674) · [arXiv 2606.30639](https://arxiv.org/abs/2606.30639) · [arXiv 2607.02846](https://arxiv.org/html/2607.02846v1) · [arXiv 2606.02372](https://arxiv.org/pdf/2606.02372) · [arXiv 2606.09032](https://arxiv.org/pdf/2606.09032) · [Opus 5 30.2% leaderboard note](https://explainx.ai/blog/arc-agi-3-opus-5-leaderboard-july-2026) · [Qwen3-VL issue #1093](https://github.com/QwenLM/Qwen3-VL/issues/1093) · [vLLM issue #12988](https://github.com/vllm-project/vllm/issues/12988)
