# Top-of-board evidence audit — 2026-08-15

**Scope:** the eight named entrants above 1.65 on the 08-15 board. Read-only. No spend, no GPU, no
push, no submission. One Kaggle write: none (leaderboard/profile reads only, already-archived CSVs).

**Method:** Kaggle public profiles (rendered via chrome-devtools MCP, signed out), `kaggle kernels
list --user` / `--competition`, our own `runs/lb_daily/*.csv` archives, and web search. Every method
attribution below carries an evidence class and the classes are never blurred:

- **DISCLOSED** — the entrant publicly said so, and the artifact/text is linkable.
- **INFERRED** — derived from authorship, history, timing or arithmetic. Not a statement by them.
- **UNKNOWN** — no public trace of the method. This is the expected answer and it is a complete one.

---

## §0 — Headline of the audit

1. **Evidence-class tally: 0 DISCLOSED / 1 INFERRED / 7 UNKNOWN.** Not one of the eight has
   disclosed a single word about their ARC-AGI-3 method.
2. **Zero of the eight has any public artifact in this competition.** No public notebook, no dataset,
   no model, no forum post, no writeup, by any of the ten individuals across the eight teams
   (`kaggle kernels list --competition arc-prize-2026-arc-agi-3` returns 100+ public notebooks;
   **none is authored by any of them**). Franzen's two public kernels are his 2024 and 2025
   solutions to *different* competitions.
3. **The "public notebook was forked" hypothesis is REFUTED with data.** The highest score claimed
   by any public notebook in this competition is **1.21** (`caoyupeng/1-21-from-great-team-tufa-labs`);
   the next is 1.17. Nothing published can seed 1.75–2.58. No new high-claim notebook exists.
4. **The "platform rescore / scoring artifact" hypothesis is REFUTED with data.** cstl flat at 2.70
   for 3 days, KOJIMA flat at 1.86 for ≥22 days, Andy liu flat at 1.69 since 08-03, our own draw
   byte-identical at 1.33. A rescore does not leave incumbents untouched.
5. **★ The Qwen3.8 timing argument rests on a field that cannot carry it.** The leaderboard's
   `LastSubmissionDate` is the date of the team's **most recent** submission, while `Score` is their
   **best**. The two need not be the same submission — our own `runs/lb_ground_truth.md` says so
   explicitly. **"Last submission after 15:00Z" therefore does not establish that the scoring
   submission was after 15:00Z.** It is also near-vacuous as a discriminator: 85% of teams ≥1.60
   have a post-release last-submission, but so do 37% at 1.30–1.45 and 17% at 1.00–1.30 — the
   gradient measures *engagement*, not the release.
6. **★★ A large, real, quantifiable distortion IS present and it is boring: best-of-N inflation.**
   The public score is a *maximum* over submissions. Calibrated on our own byte-frozen artifact
   (n=32, s=0.1533), the correction reproduces our own displayed 1.33 → per-draw 0.94 against a
   measured mean of 0.9353. Applying it re-orders the top of the board substantially (§3).
7. **★★★ The "credentialed ARC pedigree ⇒ top of this board" inference is REFUTED by counter-example.**
   **Jack Cole (`jcole75`) — MindsAI, the originator of test-time training for ARC, ARC Prize 2025
   3rd place — is at 1.59, rank #22, with 95 submissions.** His 2025 teammates **Tufa Labs are at
   1.62, #15, with 107 submissions**. Franzen high is one data point; the two other most
   TTT-credentialed entries on this board are mid-pack.

---

## §1 — The table

Score = public LB 2026-08-15 ~14:17Z. "First seen" is bounded by our own top-20 archives
(`runs/lb_daily/`), whose per-day visibility floor is given in §2 — absence from an archive means
*score below that day's floor*, not absence from the competition.

| Entrant | Score | Subs | First seen at this score | Background | Method | Evidence class | Source |
|---|---|---|---|---|---|---|---|
| **cstl** (`tehnar`,`gatamaz`) | 2.70 | 25 | 2.70 banked **08-12 20:02Z** (in 08-13 archive); 2.52 on 08-11 18:25Z; **1.59 from 08-04→08-09** | `tehnar` = "Tehnar", SWE, Amsterdam NL, 11y, pedigree *competitive agents in simulation* (Lux AI S3 116/701, ICPC Finals 2016). `gatamaz` = "TG", San Francisco, 9y, **1 competition ever**. Both: zero ARC artifacts | **UNKNOWN** — untraced 4 days running. `tehnar`'s only public notebook is an 11-yr-old Theano CNN; `gatamaz` has none. No forum post, no dataset, no model, no repo, no paper | **UNKNOWN** | Kaggle profiles `/tehnar`, `/gatamaz`; `runs/lb_daily/lb_2026-08-0{6..13}.csv` |
| **Daniel Franzen** (`dfranzen`) | 2.58 | **41** | **< 1.56 as of 08-14 12:31Z**; 2.58 by 08-15 14:17Z. Last sub 08-14 21:37Z. **NOT "one submission" — 41 total, and the scoring one is undated** | **The most credentialed ARC entrant on the board, VERIFIED from profile: ARC Prize 2024 = 1/1427 (grand prize), ARC Prize 2025 = 2/1455.** Deep Learning Researcher, Univ. of Mainz; PhD on equivariant NNs. Mutual-follow with Jan Disselhoff → *the ARChitects*. Also AIMO2 293/2212, AIMO3 1286/4138, Jane Street 804/3757 | **Prior methods DISCLOSED, current method UNKNOWN.** DISCLOSED for **ARC-AGI-1/2**: (a) 2024 — Mistral-NeMo-Minitron-8B fine-tune + TTT + augmented inference + candidate selection (`dfranzen/arc-prize-2024-solution-by-the-architects`, 141 votes; model `wb55L_nemomini_fulleval`); (b) 2025 — **"Recursive Masked Diffusion", LLaDA-8B fine-tuned**, writeup "The ARChitects' Solution", report at `lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects`; models `LladaMix1400k-*-4bit`. **For ARC-AGI-3 he has published NOTHING**: no kernel, no dataset, no model, no forum post, no writeup | **DISCLOSED (2024/25, different benchmark)** + **UNKNOWN (AGI-3)**; "he is doing TTT here" is **INFERRED** | Kaggle `/dfranzen{,/models,/datasets,/competitions,/writeups,/discussion}`; ARC Prize 2024 tech report arXiv:2412.04604 |
| **Nikita Sorokin** (`nikitasorokin`) | 2.10 | **6** | < 1.56 as of 08-14 12:31Z. Last sub 08-14 19:30Z | Joined 4 yr ago. **No tier, no medals, no bio, no occupation, no location, 1 follower** (follows Jeremy Howard). Only other competition: "Journey to Springfield" (community) at **5031/5251**. **2.10 on 6 submissions total** | **UNKNOWN** — zero public artifacts of any kind | **UNKNOWN** | Kaggle `/nikitasorokin{,/competitions}` |
| ⚠ *identity trap* | | | | **Two same-surname researchers surfaced in search and NEITHER is verified as this account: (a) "Nikita Sorokin", NLP researcher, Huawei/Skoltech; (b) *Ivan* Sorokin, NVIDIA KGMoN / team NVARC, who won ARC Prize 2025 on ARC-AGI-2 with synthetic data + TTT.** (b) is a **different first name**. Do not let either harden | — | **UNKNOWN — name coincidence, explicitly not attributed** | WebSearch; refuted against the Kaggle profile, which shows no research affiliation |
| **Yusaku Muroya** (`ymuroya47`) | 1.98 | **71** | < 1.56 as of 08-14 12:31Z. Last sub 08-15 02:36Z | **Competitions Expert, rank 950/212,348** (2 gold-equivalent + 2 medals), Senior RF Engineer, **Murata Manufacturing, Kyoto**. 49 competitions, 13 public notebooks — a broad-domain grinder (CAFA-6, PhysioNet ECG, Santa 2025, ROGII). **None of his 13 notebooks is ARC-AGI-3** | **UNKNOWN** — no public artifact here; his 10 forum posts are all admin/logistics questions in *other* competitions | **UNKNOWN** | Kaggle `/ymuroya47{,/discussion}`; `kernels list --user ymuroya47` |
| ⚠ *search error corrected* | | | | WebSearch asserted he is a "legendary Japanese Kaggle Grandmaster" at "Rank 13". **FALSE.** He is a Competitions **Expert** at rank 950. Recorded as an instance of search-summary fabrication | — | — | Kaggle profile, direct |
| **AbeLincoln1865** (`abelincoln1865`) | 1.90 | **7** | < 1.56 as of 08-14 12:31Z. Last sub 08-15 00:22Z | Pseudonymous. Joined ~1 yr ago. **No tier, no medals, no bio, no location, no followers, zero completed competitions.** Active in 3 (RSNA Knee, ARC-AGI-3, Kaggriculture). **1.90 on 7 submissions** | **UNKNOWN** — zero public artifacts. External search for the handle returns only an unrelated fandom-wiki user | **UNKNOWN** | Kaggle `/abelincoln1865{,/competitions}`; WebSearch (nil) |
| **YUTO KOJIMA** (`kojimatech`) | 1.86 | 69 | **1.86 unchanged in every archive 08-06 → 08-15**, and #1 since ≥07-24 per `lb_ground_truth.md`. **Predates the event by ≥3 weeks** | Joined **7 months ago**. No tier, no medals, no bio, no location. 2 active competitions (RSNA Knee, ARC-AGI-3), zero completed. 24 followers (people watching the ex-leader) | **UNKNOWN** — zero public artifacts. Resubmits near-daily at 00:00Z with no score change | **UNKNOWN** | Kaggle `/kojimatech{,/competitions}`; `lb_2026-08-0{6..15}.csv` |
| **MLRush** (`mlrush`) | 1.75 | 49 | < 1.56 as of 08-14 12:31Z. Last sub 08-15 00:01Z | **Competitions Master**, rank 14,041 (highest ever 180). "Research Scientist at VOID MAIN LAB", Hong Kong, joined **13 years ago**. **All 6 completed competitions are 2013–2014 CTR/ranking/recsys** (ICDM 2013 Expedia 5/336, Influencers 2/132, Criteo, Yelp, dunnhumby); his only writeup is dated Nov 2013 and his only 2 forum posts are 13 yr old. **Returned from a ~12-year hiatus.** Zero ARC / LLM / agent history. Linked handle `scmyyan` → GitHub with 5 repos (Perl data utils, sofia-ml fork, RealChar) — **no ARC/reasoning work** | **UNKNOWN** | **UNKNOWN** | Kaggle `/mlrush{,/competitions,/writeups,/discussion}`; github.com/scmyyan |
| **Andy liu** (`codinggodandyliu`, `ichuqinggaove`) | 1.69 | **7** | **1.69 in every archive since 08-06; last submission 2026-08-03 12:09Z.** **Static for 12 days — banked 11 days BEFORE the event.** Not part of the 08-15 wave at all | 2-person team. `codinggodandyliu` joined 8 months ago, one completed comp (BirdCLEF+ 2026, 2735/4094). `ichuqinggaove` ("Chuqing Gao") **joined 24 days ago, last seen 9 days ago**. No tier, no bio, no artifacts either side | **UNKNOWN** | **UNKNOWN** | Kaggle `/codinggodandyliu{,/competitions}`, `/ichuqinggaove`; `lb_2026-08-0{6..15}.csv` |

**Tally: DISCLOSED 0 (for ARC-AGI-3) · INFERRED 1 (Franzen, and only as a hypothesis about method
transfer) · UNKNOWN 7.**

---

## §2 — What our own archives can and cannot date

Our daily archives stored only the **top-20**, so each day has a *visibility floor*: a team below it
is invisible, and we can only bound their score, not read it.

| Archive | Pulled (local) | Rows | Floor |
|---|---|---|---|
| 08-06 | 06:01 | 16 | 1.54 |
| 08-07 → 08-11 | 06:01–06:02 | 20 | 1.49–1.50 |
| 08-12 | 06:00 | 20 | 1.54 |
| 08-13 | 06:01 | 20 | 1.56 |
| **08-14** | **08:31 (= 12:31Z)** | 20 | **1.56** |
| 08-15 | 10:16 (= 14:16Z) | 20 | 1.60 |

**The tightest bound available on the five new names is therefore: score < 1.56 at 2026-08-14
12:31Z, ≥ their current score at 2026-08-15 14:17Z.** That is a **25.8-hour** window. The
Qwen3.8-27B release (08-14 15:00Z) sits **2.5 hours into it**, so ~90% of the window is
post-release. **This is weak-to-moderate temporal support, not the "every one of them is after the
release" that a `LastSubmissionDate` reading suggests** — and `LastSubmissionDate` does not date the
scoring submission at all (§0.5). We cannot exclude that any of these scores was banked between
12:31Z and 15:00Z on 08-14.

Two names are outside the window entirely and must not be counted in the wave:
**YUTO KOJIMA (1.86, unchanged ≥22 days)** and **Andy liu (1.69, last submission 08-03)**.
cstl (2.70 on 08-12 20:02Z, ~43 h pre-release) is likewise outside it. **So "six above 1.75" is
three pre-existing scores plus five genuinely new ones, not six simultaneous arrivals.**

---

## §3 — The boring hypothesis that is actually TRUE: best-of-N inflation

The public score is `max` over a team's submissions. Our own record is a controlled measurement of
how big that distortion is: **n=32 draws of a byte-identical artifact, mean 0.9353, s 0.1533**, and
our displayed score is **1.33 = mean + 2.57 s**. Inverting `E[max of n]` (Blom) recovers our own
per-draw mean to two decimals — the correction is calibrated on our own data, not assumed.

| Team | LB (max) | Subs | E[max]/s | **Implied per-draw mean** | Rank by LB → **by per-draw** |
|---|---|---|---|---|---|
| cstl | 2.70 | 25 | 1.96 | **2.40** | 1 → **1** |
| Daniel Franzen | 2.58 | 41 | 2.17 | **2.25** | 2 → **2** |
| Nikita Sorokin | 2.10 | **6** | 1.28 | **1.90** | 3 → **3** |
| AbeLincoln1865 | 1.90 | **7** | 1.36 | **1.69** | 5 → **4** |
| Yusaku Muroya | 1.98 | 71 | 2.38 | **1.62** | 4 → **5** |
| YUTO KOJIMA | 1.86 | 69 | 2.36 | **1.50** | 6 → **6** |
| Andy liu | 1.69 | **7** | 1.36 | **1.48** | 8 → **7** |
| MLRush | 1.75 | 49 | 2.24 | **1.41** | 7 → **8** |
| Tufa Labs | 1.62 | 107 | 2.52 | **1.23** | 15 → — |
| Jack Cole (MindsAI) | 1.59 | 95 | 2.48 | **1.21** | 22 → — |
| **Canivel (us)** | **1.33** | 111 | 2.54 | **0.94** *(measured: 0.9353)* | — |

Board-wide this is not a small effect: **r(log1p(subs), score) = 0.561** across all 2331 teams;
median submissions are **44.5** for teams ≥1.60, **25** for 1.30–1.59, **3** for the rest.

**Caveat, stated so it is not lost:** this assumes each team's per-submission spread resembles ours
(CV ≈ 0.16). A near-deterministic agent has σ≈0 and no inflation at all; a noisier one has more.
The corrected column is therefore **INFERRED**, is a *lower* bound on capability for a
low-variance team, and should be read as a re-ordering hypothesis, not a measurement.

**What it does establish regardless of σ:** the leaderboard's top-8 ordering is **not** an ordering
of agent capability, and **Sorokin (6 subs) and AbeLincoln1865 (7 subs) are the two most
capability-dense entries after cstl and Franzen** — they reached 2.10 and 1.90 with essentially no
draws. Whatever they are running, it works on the first few tries. **Those are the two most
interesting teams on this board and they are exactly the two with the least public trace.**

**What it does NOT do:** it does not explain the wave. Adding one draw to a team with 50 raises
`E[max]` by ~0.003. The five new names moved **+0.19 to +1.02**. Those are real step improvements,
not order statistics.

---

## §4 — Question A: is there a COMMON regime?

**Ruling: (ii) a plausible but unevidenced story — with two specific components separated out.**

**What is genuinely evidenced (class: measurement, from our own archives):** five teams made real
step improvements of +0.19 to +1.02 inside a single 25.8-hour window. Simultaneity at that scale is
not routine daily churn and is not an order-statistic artifact. **The event is real.**

**What is NOT evidenced:** that they share a method. There is **no** disclosure, no shared artifact,
no forum trace, no notebook lineage, and no repo, from any of the eight. The inference "the top
regime is TTT/fine-tuning" rests on **one** entrant's *prior-competition* authorship, and the
transfer is across benchmarks — Franzen's disclosed work is on **static grid puzzles**
(ARC-AGI-1/2), while ARC-AGI-3 is an **interactive agentic** benchmark. That is a real gap, not a
technicality.

Boring hypotheses, adjudicated explicitly:

| Hypothesis | Verdict | Basis |
|---|---|---|
| A public notebook was forked | **REFUTED** | Best public claim in this competition is 1.21; no new high notebook exists |
| Scoring / rerun artifact | **REFUTED** | cstl, KOJIMA, Andy liu and our own draw all flat and byte-identical through the event |
| Competition-endgame acceleration | **REFUTED** | Deadline is 2026-11-02; ~3 months remain |
| Best-of-N / submission-count ratchet | **TRUE and large — but explains drift, not the jumps** | §3; +0.003 per extra draw at n≈50 vs realized +0.19…+1.02 |
| A new model release lifting many boats | **PLAUSIBLE, UNPROVEN** | ~90% of the 25.8 h window is post-release, but the window opens 2.5 h *pre*-release and `LastSubmissionDate` cannot date the scoring submission. **cstl (2.70, −43 h) is a hard counter-example** |
| Several strong teams independently maturing | **PLAUSIBLE, UNPROVEN** | Requires a 36-h coincidence, which is the whole reason the release hypothesis is attractive |
| Pedigree / TTT lane maturing | **REFUTED as a general rule** | **Jack Cole (MindsAI, TTT originator, ARC-2025 3rd) = 1.59 #22; Tufa Labs (his 2025 teammates, authors of the duck harness) = 1.62 #15.** Franzen is one point, not a regime |

**The honest sentence:** *six of eight are UNKNOWN, one is UNKNOWN-for-this-benchmark with a
DISCLOSED history on a different one, and one (Andy liu) was never part of the event.* We are
pattern-matching a real burst of scores onto an unobserved cause. The burst is data; the cause is a
story. **Two of the campaign's boring hypotheses are now dead by measurement, one is confirmed as a
large distortion of the ranking, and the model-release hypothesis survives only as untested.**

**Registered before any pivot:** the claim "the top regime is TTT/fine-tuning" is **INFERRED from a
single entrant's 2024/25 authorship**, is contradicted by the position of the two other most
TTT-credentialed teams on this same board, and must not be logged as anything stronger.
The ITERATION_LOG line *"His arrival at 2.58 in one submission"* is **factually wrong**: he has
**41** submissions, and the leaderboard does not reveal which one scored.

---

## §5 — Question B: highest-information no-spend probes, ranked

**#1 — Archive the FULL leaderboard daily (with `SubmissionCount`) and diff it. Read Tufa Labs and
Jack Cole as the control arm.** *Cost: one `kaggle competitions leaderboard -d` call per day, zero
spend, already-proven route.* This is the highest information-per-cost by a wide margin because it
converts the single decisive ambiguity into a measurement within 24–72 h:

- Per-team **Δscore and Δsubmissions** separates a real step from an extra draw — the exact
  confound §3 shows is board-wide and currently unquantified per team.
- **The discriminating readout: does the 1.55–1.65 duck band lift?** That band is a known shared
  public artifact and **it is where we live**. If a drop-in engine swap is the mechanism, the band
  lifts broadly within ~72 h. If only the same five teams hold their gains, the mechanism is
  team-specific and no model swap will buy it.
- **Tufa Labs (107 subs, 1.62) and Jack Cole (95 subs, 1.59) are the natural control.** They wrote
  the harness and the TTT literature respectively, they submit constantly, and they know exactly how
  to swap an engine. **If they do not move in 72 h, the commodity-engine story is weak.**
- Bonus: it repairs the instrument defect already logged — rank history before today is
  unreconstructible because only top-20 was archived, which is *why* §2's bounds are 25.8 h wide
  instead of ~6 h.

**#2 — Publish the best-of-N correction as a standing lens on the board.** *Cost: zero, the data is
already in hand (§3).* It changes what we are chasing: the gold line's per-draw mean is ≈1.2, not
1.62, and our per-draw mean is 0.94 — so the honest deficit to gold is ≈0.28 in per-draw terms while
the honest deficit to cstl is ≈1.46. It also flags **Sorokin and AbeLincoln1865 as the highest
capability-per-draw entries on the board**, which the raw ranking hides.

**#3 — Read the ARChitects' ARC-2025 technical report** (`lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects`)
and Franzen's public Kaggle models (`wb55L_nemomini_fulleval`, the two `LladaMix…4bit` masked-diffusion
checkpoints). *Cost: zero.* It is the **only** DISCLOSED method belonging to anyone in the top eight.
It will not tell us what he is doing on AGI-3, but it bounds what he would plausibly port — and it
tells us whether the 2025 recipe (LLaDA masked diffusion, not the 2024 autoregressive TTT recipe the
campaign has in mind) is even transferable to an interactive benchmark. **Note the campaign is
currently reasoning about his 2024 method; his most recent disclosed method is a different one.**

**#4 — Ask on the forum.** Low yield (nobody in the top eight has ever posted here), and it
telegraphs our interest. Not recommended.

**#5 — Further cstl attribution.** Exhausted four days running. Zero trace, two handles, no repo, no
paper, no post. Stop spending on it.

---

## §6 — Instrument notes generated by this audit

1. **`LastSubmissionDate` ≠ date of the scoring submission.** Any timing claim built on it is
   unsound. This is now the second instrument on the leaderboard path found to mislead
   (after the non-contiguous `--show --page-token` pagination).
2. **`SubmissionCount` was in the full leaderboard download all along and was never used.** It is
   the single most informative unused column we hold: it makes best-of-N inflation measurable.
3. **WebSearch summaries fabricated two verifiable facts in this session** — Muroya's tier/rank
   ("legendary Grandmaster, Rank 13"; actually Expert, rank 950) and a Sorokin identity (returned
   *Ivan* Sorokin of NVIDIA NVARC for a query about *Nikita*). Both were caught by going to the
   primary source. **Do not let a search summary become an evidence class.**
4. **Top-20-only archiving cost us the ability to date this event to better than 25.8 h.** Fixed by
   probe #1.
