# Tufa Labs Milestone-1 Writeup — Reconciliation vs Campaign Plan v4
Source: Kaggle discussion 717133 (37 upvotes, 11 comments). Reviewed 2026-07-08.

# What the writeup reveals (that the code alone didn't)

1. **Their "World Model" is a prompt-carried text note, not code.** The model writes a `World model:` tag in its response; the harness copies it into the next user message until overwritten. No executable transition model anywhere in the winning solution. The 1.21% winner has zero program-synthesis WM.
2. **Score variance is worse than our plan assumed.** Their *best* submission scored as low as **0.77%** on Kaggle (official 1.21, one 1.30 retracted by Kaggle). Public train: **1.6002 ± 0.4475** over 25 games × 20 tries, per-game SD up to 0.4. Distribution is bimodal: some games >40% of levels consistently, others never clear level 1.
3. **Runtime budget: 9 hours for 110 games** (RTX Pro 6000, 96GB). Our plan's evidence base says "25 games × 2.2h in ~12h wall" — the scored environment is 110 games / 9h, ~5 min/game average. T_game assumptions need re-derivation from logs.
4. **They admit not using vLLM prefix caching optimally** — free throughput headroom for us.
5. **Mechanics revealed:** REPL fully resets between toolcalls (persistence only via pre-loaded harness variables: `history`, `transitions`, `current_frame.segmentation`, etc.); 30s exec / 4096-char output caps; 64k context cap, evict-to-32k by dropping oldest user message + subsequent assistant turns; UNDO deliberately removed (model undoes huge batches, wastes energy); image = 4× upscale, once per turn (16×16 Qwen patches); multi-frame/video injection **failed** for small models.
6. **Named blind spot:** no animation feedback → fails official games **sb26, tn36** where animation is crucial.
7. **Key negative result:** "hand-crafting specific tools for the model did not help, as it seems to hinder the creative abilities of the model." Main improvement drivers were better base models + multimodality. Nothing was systematically ablated.
8. **20-try × 25-game results directory is public** (`duck-harness/example-run` on GitHub, plus diagnostics.html) — a free variance dataset.

# Their stated weaknesses / future work (= differentiator shortlist)

- **Context compaction / curated memory across turns** (their #1 stated improvement).
- **Perception:** models can't reason over ASCII crops; need abstract descriptions; segmentation is "a first step."
- **Animation blindness** (sb26/tn36 named).
- **No systematic exploration, no dedup, no frontier management** — the words "explore," "dedup," "novelty" never appear. Entirely open lane.
- Heavy prompt-babysitting required (energy-bar misgoals, sprite hallucination, full-grid dumps diluting attention).
- Prefix caching unexploited; large run-to-run variance unaddressed.

# Contradictions / confirmations of our plan

- **Phase-2 kill: CONFIRMED, and recontextualized.** The winner needed no executable WM; their WM section describes a one-line memo mechanism duck already has. Our pilot's 0/10 Class-A is consistent with their observation that 27B-class models need extensive domain adaptation. Killing runtime synthesis loses nothing the winner had. The cheap substitute aligned with their own future-work list is **structured curated memory** (upgrade the `World model:` note into a schema'd, harness-validated memory block) — near-zero token cost, no synthesis.
- **Phase-1: CONFIRMED as the open differentiator.** Exploration/dedup/frontier is absent from their writeup and their future-work list — nobody in the winning lineage is doing it. Our plan's stated duck weakness ("no systematic exploration") is validated verbatim by omission.
- **One caution against Phase-1 as designed:** their tools-hinder-creativity finding argues `explore()` should run **harness-side** with results injected as curated context, not as another tool the LLM must elect to call. LLM self-routing (our A/B variant) is the risky arm.
- **P0 gate tension:** our 0.82 draw sits inside Tufa's own observed range for their best build (0.77–1.30). Gate ≥0.9-within-2-attempts implicitly assumed tighter variance; a second draw in the 0.8s would be indistinguishable from a faithful repro. Don't burn the 4-day bisect on what may be pure σ.

# Concrete adjustments to Phase-1 build

1. **Pull `example-run` (25 games × 20 tries) now**; compute per-game variance decomposition, tokens/turn, per-game solve bimodality. This pre-seeds the Phase-0b null and MDE, potentially cutting 30–60 A40-hours.
2. **Re-derive T_game from 110 games / 9h**, not 25/12h; recheck the explore 30% / act 25% allocation against ~5 min/game.
3. **explore() harness-side:** trigger scripted exploration in the harness, inject a ≤500-token curated summary (novel deduped states, frontier scores, changed-frame diffs) into the user message. Dedup archive lives as a pre-loaded REPL variable (matching their variable idiom), since the REPL resets per toolcall.
4. **Add frame-diff/animation summarization** targeting their named sb26/tn36 blindness: text summary of pixel deltas across the action's intermediate frames — attacks a confirmed gap without multi-image injection (which they showed fails).
5. **Enable vLLM prefix caching properly** (stable system prompt, append-only ordering, eviction boundaries at message granularity) — more turns per game inside the 9h budget.
6. **Keep:** UNDO excluded, 4× upscale single image, 64k/32k eviction scheme, segmentation tool (extend with Rudakov motion-salience tiers rather than replace).
7. **P0 interpretation:** if attempt 2 ≥0.9, proceed; if 0.77–0.9, treat as within Tufa's published variance band — log as repro-consistent, spend the bisect budget on Phase-1 instead.

# Comments worth acting on

- **Jeroen Cottaar (author):** frontier models were tried in the harness but only on a few games (cost); details in the tufalabs.ai/research/duck-harness blog post — worth reading for the frontier-vs-27B gap magnitude.
- **ola sadek (33rd):** claims a token-burning bug "in your output files while reading them." Unverified (downvoted) but cheap to check — audit the fork's file-read/output path for grid dumps exceeding the 4096-char cap.
- **Avinav Sahoo:** claims 9.76 on public train, "will submit soon." Unverified, downvoted, no rank shown — watch the LB, don't react.
- Rest are congratulations; no action.
