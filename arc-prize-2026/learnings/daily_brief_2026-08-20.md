# DAILY BRIEF — 2026-08-20 (iterate session)

## 0. STATE AT SESSION START
- Overnight draw **0.41** (frozen-fork filler, COMPLETE) — new record minimum, z = −3.44 vs the prior band.
- Board figure **UNCHANGED at 1.33** (public score is a MAX over submissions; banked 07-18, now 33 days stale).
- **Both push slots already SPENT** by the graft lane before this session opened (06:45 q38-field v1, 07:15
  graft-floor v5). `runs/lane_locks.json` owns both lanes. **This session pushes nothing** — correct per the
  lane-ownership ruling; today is monitor / certify / instrument work.
- Both kernels **RUNNING** at session start; ETA ≈ 14:45 and 15:15 EDT — inside the 18:00 queue-head gate.
- `kaos bench rejections` → **empty**. The registry has no rejected-mechanism rows yet, so the CONSUME half of
  the publish+consume loop currently returns nothing. Not a failure; it means we are still the only publisher.

---

## 1a. RESULT DEEP-DIVE — the 0.41, and why the rule written to read it is the wrong ruler

Re-derived from `runs/ledger.json` (n=37, mean 0.9316, s 0.1771). **Three findings, all against our own prior.**

**(i) The PARTIAL-RUN DEATH / mixture story has no support in the data.**
Filliben normality correlation on all 37 draws = **0.9872** vs the n=37, α=.05 critical value **0.9665** →
**normality is NOT rejected.** Draws below 0.65: **observed 1, expected under a pure Gaussian 1.06.** A
death-mixture firing at any appreciable rate would pile up *more* mass in the low tail than we observe; there
is none. The mixture was adopted last night as a "working hypothesis" and has been repeated since as though it
were a mechanism. **It is an UNTESTED STORY and must be labelled as one in every entry that repeats it.**

**(ii) z = −3.44 overstates the surprise ~36×.** That z is the probability of a *pre-designated* draw landing
at 0.41 (2.90e−04). But 0.41 is the **minimum of 37 draws**. Multiplicity-corrected:
P(min of 37 ≤ 0.41) = **0.0107** — a **~1-in-94 campaign event**, not a 1-in-3400 one. Surprising; not evidence
of a broken rail.

**(iii) ★ THE ACTIONABLE ONE — the sealed watch rule is under-powered and was adopted as if decisive.**
The rule: *"a second consecutive draw < 0.65 BREAKS stationarity and reopens structural investigation."*
Against the current band, **P(draw < 0.65) = 0.0559**. So the rule **fires by chance about one night in 18**;
its false-alarm rate is **5.6%**, on a single draw, for a decision that costs days of campaign time.
Calibration: T=0.65 → 5.59% (loose) · 0.60 → 3.06% · 0.55 → 1.56% · 0.50 → 0.74% · 0.45 → 0.33%.

**Sealed pre-registered read for tonight** (`learnings/war_room/watchrule_calibration_audit_2026-08-20.md`,
written BEFORE the 00:07 draw, artifact bytes of the original rule untouched):
1. **≥ 0.65** → reversion; 0.41 stands as a one-off extreme tail draw of a stationary band; do not reopen.
2. **0.60–0.65** → the sealed rule technically fires but at α=5.6% is **NOT decisive**; record WATCH-CONTINUES,
   n=38, re-derive; spend no slot on structural work.
3. **< 0.60** → **DECISIVE (α ≤ 3.1%)**; stationarity broken low; the rail, not the arm, is first suspect.

Ledger must be re-derived by `scripts/ledger.py` before any prereg reads it. The bar moves nightly. Never cache.

**What would actually test the mixture:** per-game scores from a rerun. The public LB gives one aggregate number
and **cannot** show a partial run. We have no such artifact and no way to get one, so the hypothesis is not just
untested — on present instruments it is **untestable**. Say so rather than carrying it as a mechanism.

---

## 1a-bis. ★★ THE DAY'S PRIMARY DELIVERABLE — a broken instrument caught BEFORE the data landed

`feedback_audit_the_instrument`: audit the gate before the data lands. Both scorers on disk were checked against
today's in-flight field arm. **Neither can read it, and both fail toward INFRA DEATH:**

| scorer | verdict on a *healthy* field arm | why |
|---|---|---|
| `duck_eval/graft/graft_score.py` | **INFRA DEATH** | requires `[goalkeep] armed` + `[hudmask] armed`; the field arm is a byte-faithful FOYSAL rebase with **grafts 0 by construction**. Also asserts served model `vrfai/Qwen3.6-27B-FP8`; this arm serves `Qwen/Qwen3.8-27B-FP8`. Two independent counts. |
| `duck_eval/q38/q38_score.py` | **INFRA DEATH** | sealed on a different question (REFUTE-2×/CONFIRM-2× bands, not the K3″ HARM/NULL/SIGNAL lines this arm was sealed on); also trips its own window-drift assert. |

**This is not theoretical — it was executed.** A realistic fixture (25 games, lc_total 30, clean FOYSAL log) was
run through all three scorers: the two on disk return INFRA DEATH; the new sealed scorer returns **SIGNAL**.
Had 14:45 arrived and either been reached for, **a healthy arm would have been voided by the instrument rather
than by the evidence — the 08-12 failure mode, fifth instance, and once again silently in our favour.**

**Closed:** `duck_eval/q38/q38field_score.py`, sealed ~09:15 EDT **before** the kernel reached COMPLETE. Every
constant transcribed from the prereg (no new number invented); HARM/SIGNAL derived as ±C(3)·σ, never typed.
Selftest **OK, 21/21**, and every certification gate is **negative-controlled** per `feedback_guard_never_fired`
— wrong served model, pinned `reasoning_effort`, graft marker present, clickmap armed, stock fallback,
ModuleNotFoundError, wrong game count each individually **proven able to refuse**, and the good path proven **not**
refused. The 08-19 JSON-array log-decode fix is **imported** from `graft_score.py`, not copied, so it cannot
silently diverge.

**Design note that made a shared scorer impossible:** the graft arm is defined by graft markers being
**PRESENT**; the field arm by their being **ABSENT**. Same principle, mirrored — one scorer cannot hold both.
`--certify-only` returns **before** lc/score are computed, so the 18:00 queue-head call cannot be contaminated
by having seen the numbers.

---

## 1b. DISCUSSIONS SWEEP — nothing new

Forum enumerated via `kaggle==2.2.2 competitions topics list` (browser route blocked: the chrome-devtools
profile is held by another process; WebFetch returns only a JS shell — the CLI is the reliable route and should
be the default). **Newest topic is 735662, 2026-08-17 13:03.** No topic on 08-18, 08-19 or 08-20 — confirmed by
max-id check across both pages. **NOTHING NEW SINCE YESTERDAY'S SWEEP → no adopt/adapt/ignore calls today.**
Re-verified: the forum still discloses **nothing** about banking/transfer/grafts, and nothing about cstl.

---

## 1c. RESEARCH SWEEP — one direct hit on the standing open question

**★ BeliefMem — "Belief Memory: Agent Memory Under Partial Observability"** (arXiv **2605.05583**, v1 2026-05-07)
→ **ADAPT (park for Sunday's panel; no slot today).**

Its stated failure mode is our open question named almost verbatim: existing memory *"store[s] each observation
as a single deterministic conclusion … the agent acts on the stored conclusion, never revisits alternatives, and
reinforces the conclusion over time."* Mechanism: retain **multiple candidate conclusions with probabilities**,
updated by **Noisy-OR**, all surfaced together at retrieval so alternatives stay visible.

*Why it matters to us:* it is a **third, distinct** mechanism for "the agent FORGOT". We already have two —
(a) blanket-wipe on level transition, (b) **734843 delivery failure** (97.64% of content goes to the hidden
channel and is never captured; source-proven 08-18, KAOS exp_id 18). BeliefMem says that even with perfect
capture, a **committed** conclusion crowds out revision — which on ARC-AGI-3 looks exactly like "tried action X
once, concluded it does nothing, never retried".

*Why ADAPT and not ADOPT:* evaluated on **LoCoMo/ALFWorld, not ARC-AGI-3**; no extractable effect size (the
abstract reports "best average performance" with no number); and it is a memory-**store redesign**, i.e. expensive.
Under `feedback_arc_zero_budget` and with the 734843 fix already staged and cheaper, it does not earn a slot today.

*Note on our own framing:* the brief's "forgetting REFUTED or DELIVERY-WITHOUT-USE?" question is now partly
**settled in favour of delivery FAILURE** — 08-18 proved the harness never captures the update. BeliefMem is the
best remaining candidate for the residue (mech-C delivered 96.3% and still saw no behaviour change).

Other memory papers surfaced (AgentOCR 2601.04786; Revisitable Memory 2509.23040; Addressable Recall Compaction
2607.25066; Multi-Layered Memory 2603.29194) → **IGNORE**: all target context-window *overflow*. Our window is
31,744 tokens against a model ceiling of 262,144 — **we are not overflowing**, so compression buys us nothing.
Thin week otherwise; not padded.

---

## 2. OPEN QUESTIONS INTO TOMORROW
1. **Tonight's draw** — read strictly against the sealed 3-branch rule above, not against 0.65 alone.
2. **18:00 queue-head call** — `q38field_score.py --certify-only` on the pulled `arc3-q38-field-eval` v1.
   Certified ⇒ queue head as the A21 exploration draw (the board-verified 2.23 carries the decision, not our
   lc/score bands). Not certified by 18:00 ⇒ filler one more night.
3. **KAOS spawn remains unexecutable for tool-using work** — both defects reproduced again today on BOTH routes
   (`fable-panel` text-only/1-turn; `claude-sonnet` killed by the hardcoded 60s first-token watchdog). Sweeps were
   done inline. This is now a **two-day, four-attempt** confirmation; it is a KAOS bug to file, not a workflow to
   keep retrying.
4. **BeliefMem** on Sunday's panel agenda as the candidate mechanism for the forgetting residue.
