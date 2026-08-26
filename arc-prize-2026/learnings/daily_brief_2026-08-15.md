# Daily brief — 2026-08-15 (Sat)

**One-line state:** the board re-based and we did not; the principal ordered a STOP + RESTART on a
TTT/fine-tuning premise; **five independent agents came back and the premise did not survive** — but a
better-grounded version of the same alarm did.

---

## 1a — Result deep-dive

**Overnight filler 0.89** (`arc3-duck-repro` v3, COMPLETE). 43rd consecutive day.

Ledger re-derived via `scripts/ledger.py` (no hand edit): **n=32, mean 0.9353, s 0.1533**,
z = −0.30 (−0.62 vs the sealed control). Trailing-4 **0.910 → 0.860** — mechanical, the window
dropped a 1.09 and gained a 0.89. **Promotion bar of record: 1.0731** (was 1.0771).
Public max unchanged at **1.33, 28 days**.

**Watch-rule DID NOT FIRE, all three readings.** GUARD-2: 0.89 ≥ 0.80, sub-0.80 run 2 → 0.
Paired harm-pause as sealed: 0.9100 − 1.5·0.1992 = **0.6113**; we drew 0.89. Movement form
−0.050 against a −0.2299 trigger. The 08-14 pre-registered expectation ("if <0.80 escalation
fires") was met by not firing. 0.89 is now the record's **mode**, drawn 3×.

**The draw is a null. The board is not.** Overnight: gold/top-13 **1.58 → 1.62** (six-day freeze
broken), prize/top-5 **1.64 → 1.90 (+0.26)**, and we fell **#100 → #119 of 2331 on an unchanged
score**. Gaps widened: gold 0.25 → **0.29**, prize 0.31 → **0.57**.

**This kills the efficiency reframe arithmetically.** At its own estimated ceiling (μ 1.26–1.36),
P(clear prize line) drops **44–95% → 0.1–1.7%**. It was already refuted on capability grounds
08-12; it is now dead on both.

**Two premise corrections (mine, caught by the agent):** the "owed pulls" were not owed —
`effnote_v1` and `animation_v1` were pulled and scored days ago (NO-PROMOTE / KILL). And
**[3%,30%] is P1's M0 band, not theirs**; three distinct M0s exist, P1's delivered 3.68% PASS on
08-12. No ×1.10 reading was made.

---

## 1b — Community

Five new top-20 names, four above the *old* prize line. **cstl 2.70** (flat 3 days) ·
**Daniel Franzen 2.58 (NEW)** · **Sorokin 2.10 (NEW)** · Muroya 1.98 · **AbeLincoln1865 1.90 (NEW)** ·
KOJIMA 1.86 · **MLRush 1.75 (NEW)** · Andy liu 1.69, then a dense 1.61–1.65 band.
Two days ago exactly **one** team was above 1.86; now **six** are above 1.75.

2 new topics + 1 staff comment. **No flag category tripped** — no rules/scoring change, nothing on
private-LB mechanics.

Full board archived for the first time: `runs/lb_daily/lb_full_2026-08-15.csv` (2331 rows).

---

## 1c — Research sweep

**16 items · 0 ADOPT · 2 ADAPT · 14 IGNORE.** (`sweep_research_2026-08-15.md`)

- **ADAPT — Harness-IF (2608.11727):** Against-Prior Accuracy — score compliance only on rules
  opposing the model's unprompted default. All 12 frontier models 3.6–7.4 pts worse on
  against-prior rules. **Our P1 endpoint (M0) is exactly the raw rate this overstates.** Bonus:
  precedence does *not* follow prompt depth — tool/skill descriptions rank **last**, which is where
  the 08-12 pull-side helper arm proposes to put its instruction. *Cheapest test:* difference
  archived P1 logs vs null10 at matched `(game, board_hash)`. CPU, ~1h, data on disk.
- **ADAPT — Guardrail survival under self-summarization (2608.11392):** "a presence check is not a
  safety check." Scoped narrowly — our harness truncates, it does not summarize.
- **Nothing published since 08-12 touches the forgetting question.**

### ★ The literature audit — every published ARC-AGI-3 headline is non-comparable

Chased **MAP (2605.13037)** after it surfaced with zero hits in our entire record. The agent then
corrected its own headline, which is the right outcome:

> "22 out of 25" is **not a solve count** — it is *"consistent improvements over ReAct in 22 out of
> 25 games"*, a win-rate over a baseline scoring near zero. Summing its own tables:
> **MAP ≈ 41 levels vs ReAct ≈ 4, of 183 public levels. We clear 17.**

| System | Model | Headline | What it counts | Comparable |
|---|---|---|---|---|
| MAP 2605.13037 | Opus 4.6 | 22/25 | beat-ReAct rate; ~41 levels | **NO** |
| OPINE-World 2607.01531 | unnamed | 20/25, 78.4 | genuine solves | **NO** (exec substrate CLOSED) |
| Tycho 2607.28287 | GPT-5.6 Sol / Opus 5 | 100.0 RHAE, 183/183 | genuine saturation | **NO** |
| 2607.15439 | GPT-5.6-sol | ~99% RHAE | genuine saturation | **NO** |
| NULL 2608.04066 | — | 0/52 runs | genuine null | only pre-registered row |

**No published result exists for a small local model on the private set.** That absence argues
*against* reading this literature as evidence we underperform.

**Miss cause — not our search terms.** `panel_research_literature.md` (frozen 07-06) asserted "no
other papers surfaced," and **every later window-filtered sweep inherited that assertion without
re-checking it.** A stale instrument treated as ground truth.

---

## 2 — THE RESTART: the order's premise did not survive

The principal's order (08-15): *"THE NAME THAT MATTERS: Daniel Franzen at 2.58 … His arrival at
2.58 in one submission is strong evidence the top regime is TTT/fine-tuning, not harness
engineering."* Four agents examined it independently.

### Verdict: DOES-NOT-TRANSFER

The ARChitect recipe won **ARC-AGI-1**, a *grid-transduction* task: TTT fine-tunes on each puzzle's
demonstration pairs, then augmented inference + candidate selection pick among grids.
**ARC-AGI-3 has no demonstration pairs and no single-grid output.** By the paper's own ablation the
non-transferring components supply **two-thirds of the gain** — TTT +14.5 of +46.5, candidate
generation +17.0, aug-scoring +11.5, DFS +3.5. Only the offline fine-tune transfers, **onto a lane
we already own** (stages 1 and 1c DONE, costed 7–10 GPU-h inside the free allowance, 2 slots
already spent on stage S). The lane is not "un-executed."

### The evidence base: 0 DISCLOSED / 1 INFERRED / 7 UNKNOWN

No entrant in the top eight has disclosed anything about their ARC-AGI-3 method. `dfranzen` has one
discussion post ever (about ARC 2025), no 2026 kernel, dataset, model, repo or paper. The TTT
attribution is **INFERENCE FROM AUTHORSHIP, NOT DISCLOSED METHOD** — and we are reasoning from his
**2024** recipe while his **2025** work is LLaDA masked diffusion.

**Three factual corrections to the order:**
1. Franzen has **41 submissions**, not one.
2. He moved **1.24 → 2.58 across 3 submissions**, after **38 submissions never exceeding 1.24 —
   below our 1.33.**
3. "Nikita Sorokin = NVIDIA/Huawei researcher" is unverified name-coincidence (search returned
   *Ivan* Sorokin). Muroya is Expert rank 950, not Grandmaster.

### ★ Pedigree does not predict this board

> **Jack Cole (`jcole75`, MindsAI, TTT originator, ARC-2025 3rd): 1.59, #22, 95 subs.**
> **Tufa Labs (his 2025 teammates; authors of the duck harness we fork): 1.62, #15, 107 subs.**

If TTT were the regime carrying teams past 2.5, its inventor and our harness's authors would not be
parked one notch above us. **Franzen is one point, not a regime.**

Also: **nobody has published TTT on ARC-AGI-3.** All 16 arXiv papers are frozen-weight; DreamTeam
states verbatim *"Fine-tuning is too slow for an online loop."* The one gradient method that won
anything here (StochasticGoose, Preview 1st) is an online CNN — **our own v35 lineage, which scored
0.18.**

### What DOES survive — adopt this instead

**On our own ledger process (n=32, μ 0.9353, σ 0.1533), the top jumps are +4.8σ to +8.7σ in a single
draw.** The >1.70 band is therefore **genuinely not draw count**. This is method-agnostic, says
nothing about TTT, and is the defensible form of the alarm. A rescore was ruled out (7+ teams
byte-identical across snapshots; our own draws 0.70 → 0.89 through it).

Whether the five top movers share a cause: **NOT ESTABLISHED.**

---

## 3 — Qwen3.8-27B: NOT-WORTH-IT (but both stated blockers were false)

**Q1 UNSUPPORTED CORRELATION.** Zero top-20 entrants have said they used it. The thread's
participants rank 82nd/148th/307th/1521st; 82nd actively refutes it for #1. The only substantive
claim is 148th's "2x on the **local** 25 dataset" — no LB number, no seeds, no artifact. Combined
downloads across all three Kaggle FP8 mirrors: **61**.

**Both blockers refuted — do not cite them again.** The model we serve *today* already declares
`Qwen3_5ForConditionalGeneration` with a vision config; `text_config` diff 3.6→3.8 = **zero
differing keys**; vLLM 0.19.0 registers the arch. Official FP8 exists (30.89 GB) plus three Kaggle
mirrors in our snapshot shape. **The real blocker nobody named:** quantization backend changes
per-tensor-static → block-FP8; DeepGEMM provably off on SM120; CUTLASS path **NOT ESTABLISHED**;
worst case Triton fallback — slowest. **Action-count efficiency is our binding constraint, so a
better brain can LOSE score.**

### ★★ The finding of the day, and it is free

> Both checkpoints declare `max_position_embeddings: **262144**`.
> **The 31,744-token budget is OURS, not the model's.**

---

## 4 — OPEN DISAGREEMENT BETWEEN AGENTS (do not resolve by fiat)

Two agents took **opposite positions on the forgetting root cause** and both showed work:

- **Qwen scout:** the agent forgets — 66% of turns emit no world-model update; new weights cannot
  make it query `transitions`; and the context ceiling is self-imposed (above).
- **ARChitect agent:** forgetting is **REFUTED, not secondary** — mech-C delivered on **96.3%** of
  turns and behaviour did not move; EFFNOTE 93.8%, same result. *Delivery is not the binding
  constraint; the information arrives and is ignored.*

These are reconcilable only one way: **the agent receives the state and fails to use it.** If true,
raising the context budget is necessary-but-insufficient and would show as another 96%-delivery
null. The ARChitect agent flags its own tension: "what is left is the weights" contradicts the
08-13 census (52 teams, 0.00–1.62, **same brain**), so the LoRA lane is the *last untried* lane,
**not an evidence-backed one.**

**→ Panel item #1 for 08-16.** This is the campaign's central question and it is now genuinely open.

---

## 5 — Actions taken today

- **All lanes remain STOOD DOWN** per the principal order. LoRA canary v2 **NOT pushed** — §11.4's
  ledger re-confirm caught that the 08-14 authorization I was working from was superseded by the
  stand-down entered later. Artifact is otherwise **GO**: smoke 75/0, scorer selftest 35/0, AST gate
  182 loaded / 0 unresolved (re-injecting the real v1 bug is caught), preflight ALLOW 0 fails 0
  warns, D4 = [2,6,8,14] exact, adapters byte-exact 41,962,184 B, 4/4 datasets, env byte-matched.
  **08-15 slot 2 is free and unspent.**
- **Push guards ported to the LoRA lane** — the only guarded push script in the repo was hardcoded
  to the b122 kernel *and* to `date == 2026-08-14` (lane CLOSED); v1 was pushed ad-hoc, unguarded.
  `duck_eval/lora/lora_push_v2.sh` now carries `--confirm-push`, the idempotence check, a
  mechanised ledger re-confirm, and `--dry-run`.
- **Full-LB probe in build** — daily full-board archive with `SubmissionCount` + differ, reading
  **Tufa Labs and Jack Cole as the control arm**. Zero spend, resolves in 24–72h.
- **`ARCMorningCheck` incident** — RCA + reconciliation logged (`incident_morningcheck_2026-08-15.md`).

---

## 6 — Open questions for the 08-16 panel

1. **Is forgetting refuted or is delivery-without-use the real mechanism?** (§4 — the central one.)
2. **Does the 31,744 self-imposed context ceiling get raised, and what would a null look like?**
3. Adopt the **+4.8σ..+8.7σ** framing in place of the TTT premise? The alarm is real; the diagnosis was not.
4. Audit-gate design: should it exempt `preflight_mode: trusted-fork`? A late boot costs a filler draw
   and the streak for want of a heading. **Not weakening a safety gate unilaterally.**
5. Sweep cadence 1c → every 2–3 days (evidence: ~60% of today's effort re-confirmed yesterday's dedup;
   arXiv announced **zero** in-scope agent papers Aug 14–15). Reallocate to the 66%-no-world-model-update
   finding, which has no open lane.
6. Paper registry with a comparability flag, reconciled row-count each sweep (fixes the 07-06 freeze).
7. **Free dated falsification event: Milestone #2 (09-30) forces Franzen and cstl to open-source to
   claim $10K/$25K.** Six weeks out. Plan around it.
8. `ARCDailyIterate` still carries `MultipleInstancesPolicy=IgnoreNew` — needs a lock file, not
   `Parallel` (a Parallel policy there could double-push).

**Tonight:** frozen-fork filler, queue armed (1 pending), audit stub present, daemon fires 18:37.
**No slot spend today.**
