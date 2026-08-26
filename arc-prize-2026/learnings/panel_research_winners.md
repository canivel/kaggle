# Panel Research: What Wins ARC-AGI-3 (Tufa Labs "Duck Harness", Milestone-1 winner, 1.21)

Sources: notebook `notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb`, source bundle `jeroencottaar/taaf-kaggle-source-share` (inspected at C:/Users/dcani/AppData/Local/Temp/taaf), writeup link kaggle discussion/717133 (JS-blocked; corroborated via Tufa X post: "Duck harness, built on Qwen 3.6 27B... 1.21%"), MLST interview exists.

## Architecture (the gold)

Two repos: `tufa-arc-agi-framework` (taaf = benchmark/deploy plumbing, `bm.solver` is a pickled `HarnessSolver`) + `ARC3-Inference` (the actual agent). Model: **vrfai/Qwen3.6-27B-FP8 on vLLM 0.19**, single RTX Pro 6000, max-model-len 65536, analyzer context 32768, prefix caching, reasoning-parser qwen3, thinking ON, temp 0.6 / top_p 0.95 / top_k 20.

**It is a coding agent, not an action-picker.** The LLM never emits game actions directly; it has exactly ONE tool: `python`. Actions are executed by calling `action([...])` *inside* sandboxed Python.

```
per game (25 games, up to 28 concurrent threads, one shared vLLM server):
  session = persistent chat history (last ~30 assistant turns, token-trimmed)
  while not (win | 7920s wall | stop):
      if GAME_OVER: auto-RESET (env keeps level: ONLY_RESET_LEVELS=true)
      user_msg = step/level, valid_actions, prev-sequence outcome summary,
                 + carried WORLD MODEL (World/Goal/Action model, Findings,
                   Open questions, Plan, Cross-level notes)
                 + current-grid PNG (4x upscale, multimodal)
      loop (unlimited tool steps, but YIELD control after 60s wall):
          model must first re-emit revised "World model:" labeled blocks
            -> parsed by regex, persisted across turns (the agent's memory)
          model calls python(code); sandbox exposes:
            current_frame.{ascii,segmentation,step,level,shape}
            previous_frame, history, transitions, last_action_result,
            valid_actions, action(actions)   # executes real env steps, refreshes state
          raw numeric grid deliberately HIDDEN; segmentation is primary view:
            4-connected same-color objects with id, color, position-invariant
            shape HASH (cross-frame object tracking), pixels, boundary,
            children (containment), adjacency_list
          each call: fresh interpreter, 30s limit, stdlib-only whitelist,
            output capped ~1024 tokens
      on level_completed / game_over / run_complete: stop acting, re-ground;
      world model wiped on level transition (cross_level_notes survive)
```

## Key design choices

1. **Segmentation-first representation.** Object graph (color, shape-hash, containment, adjacency) instead of raw grids; prompt forbids printing full boards. Shape hash gives free cross-frame object tracking. This is the biggest idea worth stealing.
2. **Code as the action interface.** Prompt explicitly tells the model to write BFS/flood-fill/beam-search when the goal is understood, and "discriminating probes" when it isn't. Batching many actions per `action()` call amortizes LLM tokens.
3. **Structured persistent world model** via labeled text blocks ("World model:", "Goal model:", "Action model:", "Plan:") regex-extracted from assistant prose and re-injected every turn — cheap memory without any retrieval infra.
4. **Anti-HUD engineering.** Multiple prompt paragraphs specifically about not mistaking timer/progress bars for gameplay ("DON'T DO THIS!") — evidence this was their dominant failure mode.
5. **Robust plumbing.** Tool-call recovery from `<tool_call>` markup in plain text, context-overflow retry with history trimming, auto-RESET on game over, per-request timeouts, teardown safety.
6. **Throughput over depth:** 28 threads share one vLLM server; prefix caching + shared system prompt make this viable. 7920s (2.2h) budget per game inside the ~12h Kaggle wall.

## Weaknesses (exploitable)

- **High variance / luck.** Cottaar himself: the cleaned notebook "hasn't had the same lucky result" as the 1.21 run. Single pass, temp 0.6, no voting/ensembling across attempts.
- **Amnesia.** World model wiped at every level transition; sandbox is stateless per call (utilities rewritten constantly); no cross-game or cross-run learning; history trimmed to ~30 turns so early-game evidence is lost in long games.
- **No systematic exploration.** No Go-Explore archive, no novelty/RND signal, no state dedup — exploration is whatever the 27B decides to probe. Random-walk games with sparse feedback will starve it.
- **Perception ceiling.** Segmentation is 4-connected same-color only: multi-color sprites, patterned/textured objects, and animations fragment into many nodes. Only the *current* frame is sent as image.
- **27B reasoning ceiling** at 32k analyzer context; long tool transcripts crowd out reasoning.
- **Everything after the milestone is forkware:** unmodified/near forks (thtennant, trex99, boristown etc.) cluster the LB at 1.28–1.56 (top: Mathurin Ache 1.56 vs duck's 1.21) — small deltas (model swaps like qwen3vl, prev-frame tweaks) already beat the original, so headroom above the harness is real and cheap.

## Compute profile per game

~2.2h wall; one analyzer turn = 60s of tool-loop (several 32k-context calls with thinking) then yield; python calls ≤30s; hundreds of env actions/turn possible via batched `action()`. Whole run: 25 games x 2.2h compressed into ~12h by 28-way concurrency on one GPU — i.e., ~token-bound, not env-bound.

## What we should do

Keep our BFS/MCTS strengths but adopt: (a) segmentation object-graph + shape-hash tracking as LLM context, (b) `action()`-inside-Python batching, (c) persistent labeled world model, (d) add what duck lacks: state-dedup exploration archive + multi-attempt voting to kill the variance they got lucky on.
