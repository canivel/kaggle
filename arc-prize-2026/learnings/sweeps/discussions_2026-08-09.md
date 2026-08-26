# ARC-AGI-3 Discussion Sweep — 2026-08-09

Window: activity since `discussions_2026-08-07.md` (which recorded **zero** new topics; newest topic on the
board at that time was **732974**, posted 08-05, and the sweep moved to every-other-day → next = today).
Plan context at sweep time: A22 compaction lane **DEAD** (3 strikes; harm = eviction itself, monotonic in
eviction pressure); successor-lane candidates (a) state-externalisation w/ Tycho artifact schema,
(b) additive typed memory, (c) banking/replay revival; zero cloud budget; fork-never-build; ledger n=26
mean 0.9365 s 0.1540; our LB 1.33. **R24 full panel is today.**

---

## 1. What was actually observed (method + honesty notes)

- **Tool:** chrome-devtools MCP, live reads. No fallback to WebSearch/WebFetch was needed for the forum.
- **URL:** `https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion`
- **Sort — honest caveat:** the Kaggle SPA **discards the `sort` query parameter**. Both
  `?sort=recently-created` and `?sort=recently-published` were rewritten by the app to
  `?sort=undefined`, i.e. **the feed rendered is the default ordering, not an explicit recency sort.**
  Two compensating checks were run instead:
  1. the **highest topic ID** on the board is **733865** (top of page 1, "19h ago"), and Kaggle topic IDs
     increase monotonically — so nothing newer than 733865 exists in this competition;
  2. **page 2 was loaded and enumerated** (20 topics, all "12d ago" or older) to confirm no recent topic had
     been pushed off page 1 by the default ranking.
- **Topics visible:** page 1 = **7 pinned + 13 other topics**; pagination shows **9 pages** total.
- **Newest post dates on the board:** 733865 = **19h ago (2026-08-08)**; 733697 = **2d ago (2026-08-07)**;
  next-newest is 732974 (08-05, already processed 08-07). So the window contains **exactly 2 new topics**.
- **Both new topics were opened and read in full**; for 733865 the linked GitHub repo and its 10-page PDF
  were also downloaded and read (the Kaggle post body is a bare link).
- **No new host announcements.** Banner still "3 MONTHS TO GO". No pinned topic changed.

### Carried monitors — status
- **borro1980 732932 "Paper Track team-up"** — last comment **4d ago (08-05)**, i.e. **no new activity**;
  still −4 votes; **none of his 5 named merge targets (Nkosi Ndwandwe #12 1.58, Yuchen20 #13 1.58,
  anngle #15 1.56, Nilesh Sarkar #26 1.47, vansher) has visibly replied.** Cross-post 718572 also last
  touched 08-05. **Monitor holds, unchanged.**
- **Reki 732854 "What are your agents scoring on the 25 public games?"** — last comment **5d ago**, still
  no new replies. Community per-game baselines are **still not accumulating**. Watch-item holds.

### Leaderboard (live, same session, 08-09)
Top-5 prize cutoff **1.61**; **gold/top-13 cutoff 1.58** (Yuchen20 #13) — confirms the 1.56→1.58 move noted
in the task context. KOJIMA #1 **1.86** (63 entries). Page-1 floor **1.40** (#49 Mustang Liu). Our frozen
fork **1.33** remains below #49. Tufa Labs #35 at 1.45 (100 entries).

---

## 2. New posts since 08-07

| # | Topic | Author | Date | One-line content |
|---|---|---|---|---|
| 1 | [733865 — "RPS ARC-AGI 3 Solutions Technical Report"](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/733865) | Jason Feng (**177th**) | 2026-08-08 (paper dated 08-09) | Bare link to `github.com/iamjasonfeng/RPS-ARC-AGI-3`; a 10-page technical report on **three solutions built on the Tufa Labs Duck harness with Qwen3.6-27B** — Gorilla-1.1 (LoRA/DPO curriculum), Sandwich (6-proposal advisory tournament), **Tiger (within-level working memory + persistent cross-level memory + surprise proposer)**; attached to his paper-track submission. **Explicitly presents zero quantitative results.** |
| 2 | [733697 — "Solved (sort of): 'system error' persisted across 7 straight submissions… fix was a brand-new kernel"](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/733697) | Antoine Matemane Mahirwe (1611th) | 2026-08-07 | Full elimination log: 7 consecutive generic `system error` submissions on one iterated kernel, including a rebuild of the exact code from the only successful run; **fix was pushing to a brand-new kernel slug** (worked first try, 0.14). Also reports **ERROR submissions do not count against the 1/day limit**. 0 comments. |

---

## 3. Verdicts

### 733865 — RPS Technical Report (Jason Feng) → **ADAPT (design source), MONITOR (paper track)**

**Why this is the most relevant forum item in weeks:** it is the **only** published ARC-AGI-3 harness work
that runs on **our exact substrate** — Tufa Duck + Qwen3.6-27B — rather than on a frontier API model. Every
external design source in the R24 proposal (Schema/Opus 4.8, Tycho/Opus 4.8+5, Prime Agent/Opus 5) sits
above the model-scale line the proposal itself flags as lane (a)'s single biggest risk (§3(a) "no weak-model
ablation exists in any of the three"). This one does not.

Specific items worth taking:

1. **Tiger's MRPS is an independent instance of our lane-(b)/P3 "un-wipe" — with a refinement we had not
   specified.** Feng separates **within-level working memory (WLMRPS)** from **persistent cross-level memory
   (MRPS)**, explicitly so that "a model can revise its current-level working theory without automatically
   rewriting the cross-level policy." Our P3 as drafted (R24 §3(b)) is a single change: stop the wipe in
   `_update_summarized_knowledge_from_step_summary` and re-inject. Feng's split says the *policy* should be
   consolidated **at level-1 clear** and thereafter only refined/corrected/excepted — i.e. a two-timescale
   memory, not one durable blob. **ADAPT into the P3 arm design** (still one arm, one change: the change is
   "durable cross-level policy", with the within-level channel left at baseline).
2. **Memory kept "natural language, compact through prompting rather than symbolic truncation."** This is a
   third-party, same-harness echo of the A22 mechanism finding and of arXiv:2608.01326's generation-vs-
   selection result. **Supports** the R24 ordering constraint (§2.2) that context-shrinking work stays
   sequenced behind externalisation. No change to the constraint, but it is a cheap corroborating cite.
3. **Installed via "notebook-level runtime hooks rather than a modified Tufa source bundle."** Independent
   validation of our warpack monkeypatch discipline and of fork-never-build. **ADOPT as corroboration only**
   — no action.
4. **Sandwich's consultation channel is a working prior for our L4 "consult gate."** An *intercepted Python
   consultation request* returns advisory text instead of an environment action; **max one consultation per
   action turn**; proposers have no tools and cannot act. That is exactly the separation L4 needs, already
   demonstrated inside the Duck's single-`python`-tool shape. **ADAPT (design only; L4 is not authorised
   this week).**
5. **The de-rating is severe and must ride with the citation.** Feng is **177th** — below our 1.33. His §6
   and §8 state plainly: **no quantitative comparison, no ablations, four mechanisms combined at once in
   Tiger**, one-submission-per-day prevented repeated runs. So this is **design evidence with zero efficacy
   evidence** — strictly weaker than Tycho/Schema, and it must be handled under the same standing
   provenance rule (R24 §5.3.2). **Do not let it be quoted as validation that cross-level memory works.**
   Its actual value is as a *feasibility existence proof at 27B*, not as a result.
6. **Governance datum for R24 §5.3.1.** Feng discloses in §10/§11 that Sandwich and Tiger were
   "co-developed by me and Codex" and that Codex assisted Gorilla-1.1's training. A competitor publicly
   disclosing LLM-assisted authoring in the paper track is a **direct precedent for the proposed ruling**
   that agent-assisted code authoring is in-bounds when disclosed as authoring provenance. **ADOPT as a
   precedent cite in the §5.3.1 ruling.**
7. **MONITOR (paper track).** He has attached this to an existing paper-track submission and says he will
   keep submitting the three notebooks as the daily limit permits. Kaggle displays each notebook's best
   score automatically, so **the efficacy numbers he currently lacks will become publicly readable on his
   notebooks over the coming days.** That is a free, zero-cost external read on whether cross-level memory
   on a 27B Duck moves anything — worth a standing check in the daily brief.

**Not adopted:** Gorilla-1.1's LoRA/DPO curriculum (training-gated, and the weights route is DEAD for us);
the Surprise Proposer (20% of Explore turns × 5/level of extra inference calls — Feng's own §8.4 flags the
runtime risk, and it burns the exact deliberation budget the quadratic action penalty punishes).

### 733697 — fresh-kernel fix (Antoine Matemane Mahirwe) → **ADOPT (operational), no lane impact**

1. **Third-party confirmation of `feedback_fresh_kernel_slug` / `feedback_aimo3_fresh_slug`.** An
   independent competitor reproduced the exact phenomenon we have logged twice: an iterated slug reaches a
   bad state where *every* scored rerun fails generically, **including a faithful rebuild of the last known-
   good code**, and a brand-new slug with the same code succeeds immediately. Our memory entry moves from
   "our observation" to **externally corroborated**. **ADOPT** — and it strengthens the standing rule that
   any ERROR streak on a lane arm should be re-run on a fresh slug **before** the arm is judged, which is
   directly relevant to K2 (`PATCH FAILED` / VOID) in the R24 §6.3 kill rules: a VOID should not be charged
   to a mechanism if the slug is the suspect.
2. **New, undocumented submission-limit fact: `SubmissionStatus.ERROR` submissions appear NOT to count
   against the 1/day limit** — he made 7 failures + 1 success on one calendar day and was only blocked
   (immediate 400) after the success. **ADOPT (operational, low confidence: n=1, self-reported, no host
   confirmation).** If true it materially changes the risk of a scored-draw attempt on a fresh artifact:
   a failed push does not burn the day. **Do not act on it as fact until we observe it ourselves**; flag
   for the daily-submit daemon owner, do not change `submission_queue.json` policy on one report.
3. **Two harness details worth a look, not a lane:** (i) the framework's Swarm hardcodes `record=True`,
   writing full per-step session JSONL (~1 GB on a heavy game); (ii) the base `Agent` class **never prunes
   `self.frames`**, retaining full per-step history for the whole episode, multiplied across concurrent
   game threads. **MONITOR** — this is a *disk/RAM* pressure class, and it lands squarely on P1's safety
   canaries (R24 §6.2: live-child count ≤ 16, zero orphans, zero `RLIMIT_CPU` kills). A persistent
   namespace held per `_HarnessGameSession` adds a *third* long-lived allocation on top of these two.
   Worth checking our warpack's disposition of `record=` and frame retention **before** the P1 push, at
   zero cost.
4. **Kernel version history is retrievable only via the website, not the API/CLI.** Minor operational note;
   our repo already keeps its own byte-audited dataset artifacts, so no exposure.
5. **IGNORE for lane value.** Nothing about compaction, memory mechanism, or scoring.

---

## 4. Does this change the R24 panel input?

**No change to the lane ranking or to anything sealed. Two additions to the panel packet, one of them
non-trivial.**

- **Lane (a) unchanged as the recommendation.** Nothing in this window bears on state-externalisation,
  programmatic world models, Tycho's schema, the L0 sim re-verification, or the P1 persistent-namespace
  screen. The 08-08 proposal's §4 recommendation stands as written.
- **Lane (b) gains its first same-substrate feasibility datum — and a design refinement.** Feng's Tiger is
  the only known implementation of durable cross-level memory on the Tufa Duck at **Qwen3.6-27B**. It does
  **not** raise lane (b)'s rank (it carries no efficacy evidence at all, from a 177th-place entrant, with
  four mechanisms confounded — a confound of exactly the M3 shape the proposal is built to avoid). What it
  does is (i) **retire the "is this even implementable at 27B via runtime hooks" question** for the P3 arm,
  and (ii) **suggest P3 be specified as a two-timescale memory** — consolidate a cross-level policy at
  level-1 clear, refine-not-rewrite thereafter — rather than a flat un-wipe. Recommend the panel fold this
  into §3(b)/S4 as a design note.
- **§5.3.1 gains a precedent.** A competitor's public paper-track artifact discloses LLM-assisted authoring
  as provenance. Cite it in the proposed ruling.
- **A22 closure unchallenged.** Zero forum discussion of compaction, eviction, or context management in the
  window. Feng's independent choice of *prompt-level compactness over symbolic truncation* is weak
  corroboration of the generation-over-selection finding, nothing more.
- **Operational rail, not a lane:** confirm the fresh-slug rule and the `record=`/`self.frames` retention
  check as pre-P1 hygiene. Both are free.
- **Monitors carried forward:** (a) Feng's three notebooks — Kaggle will surface their best scores as he
  resubmits; this is the cheapest external read available on cross-level memory at 27B; (b) borro1980's
  merge solicitation — still zero uptake from any of the five 1.47–1.58 targets, thread at −4;
  (c) 732854 replies — still none.
- **Cadence:** the window produced 2 new topics, one of them substantive. Recommend **staying on
  every-other-day (next: 2026-08-11)** rather than reverting to daily.
