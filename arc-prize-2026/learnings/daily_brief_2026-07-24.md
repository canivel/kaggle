# Daily brief — 2026-07-24

## §1a Result deep-dive

### Exploration draw 1/12: sentinel arm = 0.71 — campaign low; HARM-PAUSE executed

**Draw:** `canivel/arc3-duck-sentinel` v1 (vanilla duck + (f) continuation
hygiene-default + budget sentinel v2 @ SENTINEL_BUDGET=150; NO warpack, NO
ledger) scored **0.71** at the 00:07Z fire — the lowest scored draw of the
campaign (previous low: war draw #5 = 0.76; frozen-fork low 0.82).

**Pre-registered expectation vs outcome:** entry was via 2-seed canary PASS +
non-harm screen (pooled Δlc ≈ −0.05/game, direction flipped across seeds).
The sealed A21/C2 rule — draw < 0.80 pauses the arm — **triggered; the pause
was executed pre-loop** by the early interactive session. This is the rule
operating exactly as designed: no inference is claimed from n=1, but exposure
stops immediately.

**Statistics (validated this morning):**
- vs frozen control n=10 (mean 0.975, σ̂ 0.156): **z = −1.70**, one-sided
  p ≈ 0.044.
- vs pooled control n=15 (mean 0.962, σ̂ 0.144): z = −1.75, p ≈ 0.041.
- A single draw at p ≈ 0.04 is suggestive, not conclusive — but it is the
  THIRD independent negative signal for this composition: eval-rail
  Δlog1p(RHAE) was negative on BOTH certified seeds (s1 −0.315 p=0.997; s2
  −0.166 p=0.90). Three aligned negatives across two rails = the prior for
  the sentinel composition is now poor.

**Mechanism evidence:** none pullable from the scored rerun (logs hidden);
the composition's mechanism story is already sealed from the eval rail (W1:
mechanism PASS / behavior "fires, doesn't pay" — warnings changed nothing,
21/22 fired games kept grinding). The 0.71 is consistent with the sealed
story: the sentinel adds FACT lines that at best do nothing and at worst
perturb the policy slightly negative, and under the completion-weighted
scorer an efficiency observable has no depth channel to pay through.

**Cost accounting:** one exploration window of 12 (A21); priced ~0.001–0.002
E[max]-equiv in max currency. 11 windows remain. LB best UNCHANGED at 1.33.

**Disposition question for panel (Q1 below):** the calibrated W2 rule
(sentinel_w2_preregistration.md amendment: z-rule vs legal control w0_s1 =
1.731; two-seed KILL α ≈ 0.02) is denominated on the EVAL rail — the scored
0.71 does not formally enter it. Is the $0 W2 eval build still worth a push
slot to close the line cleanly, or does the harm-pause + sealed W1 evidence
suffice to shelve the sentinel as "certified telemetry, no lift channel"
without W2?

### Ledger

Frozen control unchanged (0.71 was a different composition — it does NOT
enter): n=10 mean 0.975 σ̂ 0.156, band 0.82–1.33. Pooled n=15 mean 0.962 σ̂
0.144.

### LB context

Field compression continues: **20 teams ≥ 1.45** this morning (the old 1.44
wall is fully submerged; yesterday gold ≈ 1.47). Top: KOJIMA 1.86, Tecnod8.AI
1.61, then a dense 1.45–1.60 band. We hold 1.33 (~#45) and continue to erode
without a depth-channel gain. This is the strategic backdrop for the A17
screen and the depth-budget lane: efficiency channels price ≈ 0; a single
frontier depth event prices +0.19 to +0.29 rail (d4_provisional_reprice).

## §1b Discussions sweep (learnings/war_room/discussions_2026-07-24.md)

- **ADAPT (ops-critical, TODAY) — #684625 Scott Le Grand comment:** vLLM
  silently hangs on RTX Pro 6000 with the duck notebook after 15–20 min at
  ≥8–25 concurrent sessions (reproduced off-Kaggle). Directly relevant to
  today's A17 canary (same GPU + vLLM stack). Mitigation posture: the canary
  runs 4-game concurrency (below the reported ≥8 threshold), and our
  certified 27B 25-game runs completed clean at 2h12m — but the canary gains
  a log-heartbeat observable so a silent hang is diagnosable post-run; A24
  watchdog wording already covers the class. Watch thread for root cause.
- **ADOPT (upgraded) — boristown `duck-harness-fast-eval` public 1.47 gold,
  144 upvotes.** The top public artifact is 1.47, not 1.39. Defensive
  fork-diff target for our frozen fork upgrades accordingly — potential
  low-risk +0.14 (fork-never-build discipline; byte-matched metadata
  mandatory; fork-diff analysis is $0 and precedes any decision). New Q5.
- **WATCH:** unscored "Deep Reasoning Agent (179/183 levels)" code-tab entry
  updated 1h ago — hype-smelling; becomes ADOPT-diff only if it scores
  >1.47. Re-check tomorrow.
- **IGNORE:** #728210 (beginner), trailing drive-by comments on 07-23
  threads, #697720 grumbles (dropdown-check ops-ADAPT already carried).
- **LB (live top-49):** gold cutoff ≈ **1.49** (top 13); wall band 1.44–1.60
  dense; **we slid #45 → #49 overnight at 1.33** (~4 ranks/day bleed), now
  tied with Jack Cole. Two low-entry high scores (1.60@6, 1.49@5) exceed all
  public artifacts → real methods, not clones; the public 1.47 is itself
  seeding the 1.44–1.47 band. No arc_agi version-bump news.

## §1c Research sweep (learnings/war_room/research_2026-07-24.md)

- **Zero ADOPT, three ADAPT (all wording/design, no build), two PARK.**
- **ADAPT (low) — arXiv 2607.20972** cue-anchored working memory (Jul 23):
  agents made 0 voluntary memory ops in 114 turns; harness-injected
  deterministic cues worked with zero false alarms. Application: the
  depth-budget lane harness re-injects EWM contract + certificate state on
  deterministic cues (level change / revise-loop entry), never
  agent-volitional.
- **ADAPT (low-med) — arXiv 2607.07196** world-model admissibility ladder:
  action-following robustness ranks ABOVE visual fidelity; one line into EWM
  v1.1 certification wording (admissibility-before-verdict).
- **PARK — 2607.08964** Long-Horizon-Terminal-Bench (dense partial credit,
  best model 15.2%): external support for depth ≫ efficiency. **PARK —
  2601.22129** SWE-Replay banked-prefix replay (+3.8% at −17.4% cost):
  cleanest external statement of banking, but blocked by our latent-state
  audit (branching presumes faithful state restoration; state-aliasing
  breaks it).
- **Schema replication watch: still ZERO** (HF traces frozen at 50
  trajectories; no reproduction anywhere). OPINE-World follow-ups: none.
- **A17 serving:** no direct 72B-VL-AWQ-on-96GB datapoint anywhere;
  vllm-blackwell-guide (lastloop-ai) confirms quantized vLLM serves clean on
  RTX PRO 6000 sm_120 at 27B/35B — **toolchain half de-risked; the
  weights+vision-tower+KV fit half is settled empirically by today's
  canary** (as A23 anticipated: envelope is self-certifying).

## §2 Today's plan

1. **A17 canary push (slot 1, ratified; all sign-offs sealed via scope-v2 §9).**
   4-game full-window Qwen2.5-VL-72B-AWQ canary: measures ρ_action
   denominator (27B numerator frozen 480 actions/7920s), serve-config smoke
   (hermes parser, no thinking flags, MM path), envelope check. Build agent
   running (starts from proven eval builder; loud-fail policy if 72B fails to
   serve — a silent 27B fallback would poison ρ_action). §7 preconditions
   walked in order before push; interpretation discipline per C3
   (a17_error_model): the screen is a ≥+4-to-+5-level capability-existence
   detector; NO-GO at modest lift is the designed outcome.
2. **Panel R19 (routine, reduced bench)** on this brief. Then implement
   directives.
3. **Slot 2 contested — panel rules:** W2 sentinel confirmatory null vs
   tr87 fresh-stream depth-lane confirmation (held from the OBJ-H
   ratification: "rides a later quota slot after the panel sees this
   ratification" — the panel is seeing it today).
4. Queue: frozen-fork filler head (verified pending); daemon 18:37/20:07.

## Open questions (for R19)

1. **Sentinel disposition after harm-pause:** shelve on sealed W1 evidence +
   harm-pause (no W2, save the push), or spend a slot on the $0 W2
   confirmatory null to close the calibrated two-seed KILL cleanly? Note the
   arm cannot re-enter LB regardless without a new A21 entry case.
2. **Slot-2 allocation today:** tr87 fresh-stream confirmation (depth lane,
   OBJ-H ratified GO, kill-switch = held-out null10) vs W2 vs hold. The
   depth lane is the only non-A17 depth-targeting line; concentration risk
   says fund it.
3. **Exploration budget pacing:** 1/12 spent on a paused arm. Should the
   next exploration draw wait for the A17 screen outcome (GO → war-v4
   composition is the obvious next arm) or is there a cheaper
   depth-composition candidate first?
4. **A17 canary contingency:** if the canary shows serve-fit failure at
   32768 (envelope NO-GO per A23 self-certification), does the FP8 fallback
   TBD-(i) search start today or wait for the formal AWQ NO-GO?
5. **boristown 1.47 public fork (NEW, from §1b):** the top public artifact
   now scores 1.47 — above our 1.33 best and at the gold cutoff. A
   byte-matched fork of a public 1.47 is the exact move that produced our
   frozen duck (ashvin 0.42 era) and is low-risk under fork-never-build.
   Questions for panel: (i) does adopting it violate any sealed discipline
   (it is a PUBLIC artifact — generalization-first memory says avoid
   luck-chasing, but this is a floor-raise, not a variance lottery)? (ii)
   fork-diff first ($0, today) then queue as filler-replacement? (iii) how
   does it interact with the exploration-budget accounting — is a public
   fork a "filler" (no window cost) or an arm?
