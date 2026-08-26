# ARC-AGI-3 Discussion Sweep — 2026-08-11 (STEP 1b; closes the 08-10/08-11 forum gap)

**Window:** activity since `discussions_2026-08-09.md`, whose stated frontier was **topic 733865**
(posted 2026-08-08 17:22). Everything with a topic id > 733865, plus every *comment* posted on or
after 2026-08-08, is in scope.

**Coverage gap being closed:** `learnings/war_room/intel_sweep_2026-08-11.md` recorded plainly that
"the Kaggle discussion feed was NOT read directly today … Forum coverage therefore stands at 08-09 …
**08-10 and 08-11 forum activity is unswept.**" That gap is now **CLOSED** — see METHOD.

---

## 0. METHOD — what worked, what failed, and why

### FAILED — chrome-devtools MCP (same failure as 08-10 and 08-11 morning)

Both `mcp__chrome-devtools__list_pages` and `mcp__chrome-devtools__new_page` (the latter tried with
`isolatedContext: "sweep0811"`, which does **not** help — the isolation is per browser-context, not
per user-data-dir) returned verbatim:

```
The browser is already running for C:\Users\dcani\.cache\chrome-devtools-mcp\chrome-profile.
Use --isolated to run multiple browser instances.
Cause: ... Use a different `userDataDir` or stop the running browser first.
```

Diagnosed rather than merely retried. A live Chrome (**pid 4272**, renderers 1660/51568/36848) holds
that profile and was launched with **`--remote-debugging-pipe`**, *not* `--remote-debugging-port`;
there is consequently **no `DevToolsActivePort` file in the profile and no TCP endpoint to attach
to** — an out-of-band CDP attach is impossible by construction, not merely inconvenient. Killing the
process was declined (it belongs to another live MCP session). **Route stays broken until the MCP
server is started with `--isolated` or a distinct `userDataDir`.** No login was attempted; no
credentials were entered anywhere in this sweep.

### FAILED — raw HTML fetch of the discussion page

`Invoke-WebRequest` on
`https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion?sort=recent-comments`
returns **HTTP 200 but only 5,550 bytes** — the SPA shell. Zero occurrences of `Kaggle.State.push`,
`__NEXT_DATA__`, `topicId`, `forumTopic` or even `"title"`. There is no server-rendered payload to
scrape. (WebFetch was therefore not spent on the same URL; it converts the same shell to markdown.)

### FAILED — Kaggle CLI 2.0.0 (the version pinned in our runbooks)

`uvx --from kaggle==2.0.0 kaggle forums topics list …` →
`kaggle: error: argument command: invalid choice: 'forums'`. The 2.0.0 command set is
`{competitions, datasets, kernels, models, files, config, auth}` — **there is no forum surface at
all in that version.** This is almost certainly what produced the "403" folklore in earlier notes;
in any case 2.0.0 cannot reach the forum.

### ✅ WORKED — **Kaggle CLI 2.2.2 has first-class forum commands.** This is the fix.

`uvx --from kaggle==2.2.2 kaggle --help` exposes `forums`, and — decisively —
**`competitions topics {list, show}`** and **`competitions topic-messages`**, none of which exist in
2.0.0. Authenticated with our existing `kaggle.json`; read-only; zero spend.

```bash
export PYTHONIOENCODING=utf-8   # required: without it, topics with non-ASCII author
export PYTHONUTF8=1             # names die with 'charmap' codec errors (hit on 734046, 734233)

uvx --from kaggle==2.2.2 kaggle competitions topics list arc-prize-2026-arc-agi-3 -s new -v
uvx --from kaggle==2.2.2 kaggle competitions topics list arc-prize-2026-arc-agi-3 -s new -v -p 2
uvx --from kaggle==2.2.2 kaggle competitions topics list arc-prize-2026-arc-agi-3 -s recent -v
uvx --from kaggle==2.2.2 kaggle competitions topics show          arc-prize-2026-arc-agi-3 <id> --page-size 50
uvx --from kaggle==2.2.2 kaggle competitions topic-messages       arc-prize-2026-arc-agi-3 <id> -s new -n 20 -v
```

**Coverage obtained:** pages 1–2 of `-s new` (40 topics, back to 2026-07-13) — this brackets the
window with ~2 weeks of margin; page 1 of `-s recent` for comment activity on older topics; **full
body text of all 5 new topics**; **full message threads of 6 candidate older topics**. `topics list`
returns real `postDate` values, so recency ordering is verified from data rather than inferred from
the SPA's ranking — a strict improvement over the 08-09 browser sweep, which had to work around the
SPA silently rewriting `?sort=` to `?sort=undefined`.

**Known limitation of the route:** the API returns **`authorName` blank for individual messages**
(it is populated for topics). Comment authors below are identified from the message body (@-mentions,
self-reference) and are marked as such. Vote counts and dates are exact.

**➡ ACTION FOR THE RUNBOOK: pin `kaggle==2.2.2` for all forum reads.** The chrome-devtools route is
no longer load-bearing for the forum leg. This closes the named gap for 08-10 and 08-11 and removes
the browser-profile single point of failure that cost us two days.

**Not used:** WebSearch/WebFetch were unnecessary once the CLI route landed, and would have been
strictly worse evidence (secondary, undated, unciteable to a post id).

---

## 1. NEW TOPICS SINCE 2026-08-08 — 5 found

Ordered newest first. All quotes are from the post bodies retrieved via
`kaggle competitions topics show`. Canonical URL form:
`https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/<id>`

---

### 1.1 — **734369 · "Write Up: Taaf Anim Agent" · Jakob Brüggen · 2026-08-11 07:55Z · 2 votes, 0 comments** — **ADOPT (as a correction to the arm we sealed and pushed four hours later today)**

<https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734369>

**This is the single most decision-relevant item retrieved.** It is the author's own write-up of the
**exact feature we built, sealed and pushed today** (`animation_prereg_2026-08-11.md`), from the
Helmut AGI member (#8 @ 1.61) whose public bundle was sweep finding #1. The morning sweep read his
*code*; this post is his *results*, and it materially changes what that finding means.

**Substance — he publishes a NULL, and a harm mechanism, in his own words:**

> "…the reason our public-set A/B did not show a significant improvement (**+1.4 % mean score,
> p = 0.92 over 6 games × 4 passes**), even though the private set went well."

> "**Tokens are the real currency, not actions.** Every run in both arms hits the 132-minute
> wallclock cap. Nothing ends because of an action limit and nothing ends in a win. … Our animation
> arm went from **384 to 449 tokens per action (+17 %)**, and the effect was mechanical: in every
> single game, more tokens per action meant fewer actions… On one game we lost 54 % of our actions
> and gained 129 % token cost per action."

> "It used the tool unprompted in 21 out of 24 runs… But of **181 calls across all games, only 2
> landed on a genuinely informative animation.** Out of 96 large animation events in the data, it
> inspected **2.1 %**."

> "measuring which games those are with a random-walk script **does not predict what happens under a
> real agent**. We were off by a factor of ten on one game in both directions."

**And he retracts finding #2 of this morning's sweep himself:**

> "[the hard no-op guard] is small, and honestly **our own measurements of it are weaker than we
> first thought**. It looked like a solid ~12–20 % reduction in actions used, but we later realised
> that our runs are bound by the wallclock limit, not by the action budget… a blocked action still
> costs a full LLM turn. **So take that number with a grain of salt — we did.**"

**Why the verdict is still ADOPT and not ABORT — the distinction is load-bearing and it is ours.**
His arm is **all three stages behind one flag**: metadata + an `animation()` retrieval tool + a
proactive hint. Our sealed arm is **stage 1 only** — `animation_prereg_2026-08-11.md` §2.1 states
verbatim: "*No retrieval tool, no per-frame timeline, no proactive hint — those are the competitor's
stages 2 and 3 and are **explicitly OUT of this arm***", with a ~45-token fixed-schema scalar dict
emitted **only on animated actions**. His +17 %/action is dominated by exactly the two stages we
excluded (181 tool calls returning diff timelines, 178 of them uninformative). **His null is
therefore not a test of our arm** — but it *is* the strongest possible external validation of our
pre-registered kill-switch **K-A3 (`animation_tokens_est / total_tokens < 1 %`)** and it promotes
**M2 (tokens/action, tokens/lc, wall-clock/action)** from "descriptive" to *the* metric that decides
whether this arm can ever pay. He also states the mechanism we must now assume by default:
**under a wall-clock cap, every added token is paid for in moves.** Our own R24 finding that
"wall clock not actions binds" is now independently confirmed by a 1.61 team.

**Three further corrections to the record, free:**
- He names **`sp80`** as "the clearest case" (ACTION5 pouring phase, 22 frames, "624 pixels that only
  ever exist mid-animation"). **Our own audit classes sp80 as type-2 with 0 INVISIBLE actions.**
  Both can be true — his is a within-response transient-pixel count, ours is cross-action board
  aliasing — but it is a second instance of the taxonomy divergence our prereg §1.1 already flagged
  for `sb26`, and it is evidence our narrower INVISIBLE metric is the right one to steer by.
- He says **13 of 25** public games produce multi-frame responses; our LM-free audit measured
  **17 of 25** over 11,104 actions. Ours is the larger and better-instrumented measurement; no change.
- His "random-walk script does not predict what happens under a real agent, off by a factor of ten"
  is an independent restatement of our prereg's own **HONEST CAVEAT** (probe A 51/1,400 vs probe B
  350/1,200 INVISIBLE). Two independent measurements now say the same thing. **Our refusal to claim
  the probe-B rate was correct.**

**Action:** do **not** unseal or amend the prereg (that would be exactly the post-hoc reordering the
seal exists to prevent). File this as a **pre-result external prior**, append it to the arm's
interpretation section, and treat a K-A3 breach as fatal rather than advisory. Also: his stage-2/3
result is a **pre-registered negative for any future arm** that proposes the retrieval tool or the
proactive hint — those two are now carrying a published null from the team that invented them.
**And the hard no-op guard (sweep ADOPT #2) drops from ADOPT to WATCH:** its only quantitative
support was the ~12–20 % figure its own author has now withdrawn.

---

### 1.2 — **734233 · "Five consecutive 'Kaggle Error' submissions — anyone else?" · مشعل العتيبي · 2026-08-10 16:26Z · 0 votes, 0 comments** — **ADAPT (operational; feeds the submit daemon's failure playbook)**

<https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734233>

Five submissions on 2026-08-10 (ids 55394684 → 55408519) all failed with error type **"Kaggle Error"**
while earlier submissions from the *same notebook and account* completed and scored (0.23 on 08-07,
0.22 on 08-08, 0.00 on 08-09). "The 'Save & Run All' commit finishes successfully with no exception…
The failure only happens during the competition rerun." Config: GPU on, internet disabled,
NvidiaRtxPro6000, one model input `google/gemma-4/Transformers/gemma-4-31b-it/1`.

**Verdict ADAPT.** No host reply yet, so this is not yet a confirmed platform incident — but it is a
**second independent report of the same slug-rot pattern** as 733697 (08-07), where the fix was a
brand-new kernel slug with byte-identical code. That is our own
`feedback_fresh_kernel_slug` memory, reported twice by strangers in four days. Relevance is direct
and immediate: **we pushed an animation eval seed today**, and our daily submit daemon has exactly
one shot per day. **Concrete rule for the daemon: on the second consecutive "Kaggle Error" on one
slug, rotate to a fresh slug rather than debugging the code.** Note his own count is also useful —
he burned 5 same-day submissions on errors, consistent with 733697's finding that ERROR submissions
do not count against the 1/day cap.

---

### 1.3 — **734092 · "Chimpanzee-1.1: An RPS-Trained Model for ARC-AGI-3" · Jason Feng · 2026-08-09 20:11Z · 1 vote, 0 comments** — **IGNORE (already under monitor; body is a bare link)**

<https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734092>

Entire post body: `https://github.com/iamjasonfeng/Chimpanzee-1.1-Paper`. Same author as the RPS
technical report (733865, already processed on 08-09) and the open-sourced Sandwich/Gorilla notebooks
(732823). **IGNORE for the lane:** RPS-trained models are training-gated and off our zero-budget rail
(`feedback_arc_zero_budget`), and the **Jason Feng standing monitor already exists** from the 08-09
sweep — this is that monitor firing, not a new lane. No score claim is made in the post.

---

### 1.4 — **734054 · "Agent.MAX_ACTIONS defaults to 80, the loop takes 81, and the README never mentions it" · maximo lorenzo y losada · 2026-08-09 17:04Z · 1 vote, 0 comments** — **IGNORE (confirmed non-applicable to us; already dispositioned)**

<https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734054>

The forum companion to the notebook `maximolorenzoylosada/your-agent-stops-after-81-actions`, which
`intel_sweep_2026-08-11.md` §8 already swept and dispositioned. Reference `Agent.MAX_ACTIONS = 80`,
guard is `<=` on a zero-based counter, so the body runs 81 times; not documented in the 137-line
README; the harness's own Playback agent sets `MAX_ACTIONS = 1_000_000`. He asks the organisers to
confirm raising it is in bounds — **no host reply as of this sweep.** Also flags the API rename
`frame.score → frame.levels_completed`, `frame.win_score → frame.win_levels`.

**IGNORE for us, verified not assumed:** our solver is the duck line —
`duck_eval/taaf_bundle/.../taaf/kaggle_random.py` sets `DEFAULT_MAX_ACTIONS_PER_GAME = None` and our
shipped `preamble.txt` records `max_actions_per_game=None`. **Unbounded, confirmed.** Retained only
as LB-process-model colour: a large share of the sub-1.2 band are reference-agent forks with a
structural ~8.7 % ceiling.

---

### 1.5 — **734046 · "(Question to competition owners) — Continuity of task similarity of arc-agi-2 (cbebaa4b) and arc-agi-3 (cn04)" · Doruk Doğrular · 2026-08-09 16:39Z · 0 votes, 0 comments** — **IGNORE (unanswered question; and adopting it would violate our own priority rule)**

<https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734046>

Full body: cn04 and ARC-AGI-2 task `cbebaa4b` are "very similar. Nearly the same logic" — should
ARC-AGI-2 be treated as the primitive set for ARC-AGI-3? **No host answer.**

**IGNORE.** It is an unanswered question, not a finding, and the strategy it proposes — mining
ARC-AGI-2 tasks as a per-game prior — is a `feedback_arc_generalization_first` violation of the same
shape as `jinbowang1`'s human-induced rules (intel sweep §9): it optimises the 25 public games
against a private set that has more games. Worth **one line in the monitor list** only if a host
replies, since a host answer would be a rules statement rather than a competitor opinion.

---

## 2. NEW COMMENTS ON OLDER TOPICS SINCE 2026-08-08 — 2 found

Checked message threads on the six topics with any recent activity (732854, 732932, 732974, 732823,
732706, 733697). Only two messages postdate 2026-08-08.

### 2.1 — **732854 "What are your agents scoring on the 25 public games?" (Reki) — new comment 3511388, 2026-08-11 03:31Z, 0 votes** — **IGNORE**

Full text: **"Got 2.8 locally so far"**. Author name not returned by the API (see METHOD limitation).
No setup, no per-game table, no model, no action budget — none of the five fields the thread itself
asked for. **Uninterpretable**: "2.8" is not on the LB scale (the board leader is 1.86) and is
presumably a private local metric. **The standing watch-item from 08-09 — "community per-game
baselines are still not accumulating" — is CONFIRMED, not resolved.** Two comments in seven days,
one of them an image and one an unqualified number. Recommend **retiring** this monitor; the thread
is not going to produce a community baseline.

### 2.2 — **732932 "Paper Track team-up" (borro1980) — new comment 3511030, 2026-08-10 07:19Z, 0 votes** — **IGNORE (but the LB read inside it is worth one line)**

A targeted merge pitch to **@foysalemonshanto and @jakobbrggen**, replacing the 08-05 mass ping (−2
votes) that went unanswered. His observation: "the leaderboard shows **1.61 three times: rank 5, 6
and 7**. Your two teams posted the same score as the team inside the prize band; what separates you
from it is **submission timing, not performance**. Leaderboard prizes stop at top 5." He offers a
finished Paper-Track methods paper against their score, "even split by default", merger deadline
**Oct 26**, self-estimated odds "one in three or four for a top-three place".

**IGNORE as an offer** — we are not merging, and his own score is 0.79. **Two facts worth carrying:**
(i) **merger deadline Oct 26** confirmed from a competitor's post, which bounds the consolidation
meta-signal the 08-05 sweep opened; (ii) **still zero replies** across both his threads and 8 named
targets over 6 days — the merge market at the top of this board is thin, which weakly de-rates
"consolidation" as an explanatory model for the recent top-10 entries (Helmut AGI notwithstanding,
that one was a completed merger, not a solicitation).

---

## 3. DOES THIS CHANGE THE PLAN?

**One item does, and it lands on the thing we shipped today.** Topic 734369 is the author's own
write-up of the animation feature, and it publishes a **null public-set A/B (+1.4 % mean, p = 0.92,
6 games × 4 passes)** plus a specific harm mechanism: **+17 % tokens per action, paid for one-for-one
in moves under the 132-minute wall-clock cap**, in every game measured. It also **retracts** the
~12–20 % action-reduction claim for the hard no-op guard, on the grounds that a blocked action still
costs a full LLM turn under a time cap. That retraction comes from the author of the number, and it
is the only quantitative support finding #2 ever had.

**What this does NOT do is falsify our arm.** Ours is stage 1 only — a ~45-token fixed-schema summary
emitted solely on animated actions, with the retrieval tool and the proactive hint pre-registered as
explicitly OUT (§2.1). His token inflation lives almost entirely in the stages we excluded: 181 tool
calls, of which by his count **2** were informative. Read correctly, his post is (a) an external
pre-result validation of our K-A3 token bound, (b) a published negative that should keep stages 2
and 3 permanently out unless someone solves the *timing* of tool calls rather than their existence,
and (c) a warning that **M2 — tokens/action and tokens/lc — is now the metric that decides this arm**,
not M0. The honest summary: **the mechanism evidence stands and got stronger; the efficacy prior just
got materially worse, from the only team that has measured it.** The prereg stays sealed — this is
external prior, filed pre-result, not an amendment.

**Sequencing changes, small but real:** *animation stage-1 first* is unchanged; *hard no-op guard
second* is **demoted from ADOPT to WATCH** pending a number that survives the wall-clock correction.
Nothing here touches lane (a) state-externalisation, A22 compaction (still open, still unworked), or
the closed S1 exec-sims verdict. The two operational items — fresh-slug rotation on repeated
"Kaggle Error" (734233 + 733697), and the confirmed-unbounded action budget (734054) — cost nothing
and are already consistent with our memory.

**No host activity in the window.** No new pinned topic, no organiser reply to either open question
(734054's "is raising MAX_ACTIONS in bounds?", 734046's "is ARC-AGI-2 the primitive set?"), and
`arcprize.org` remains unchanged since 2026-07-06 per this morning's sweep. **Nothing in this sweep
argues for a strategy change; one item argues for lowered expectations on the arm already in flight,
and one retires a monitor that was never going to pay.**

---

## 4. STATUS

**COVERAGE GAP: CLOSED.** Forum coverage now runs through **2026-08-11 07:55Z** (topic 734369, the
highest topic id on the board; Kaggle topic ids increase monotonically). The 08-10 and 08-11 window
named as unswept in `intel_sweep_2026-08-11.md` has been read in full — 5 new topics and 2 new
comments, all bodies retrieved and quoted, all traceable to a post id and URL.

**Route to keep:** `uvx --from kaggle==2.2.2 kaggle competitions topics list|show` +
`topic-messages`, with `PYTHONIOENCODING=utf-8`. **Route to stop depending on:** chrome-devtools MCP
for the forum leg — it has now failed three sessions running on a profile lock that cannot be worked
around from inside the tool.
