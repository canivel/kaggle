# OBJ-I — Schema fixed-resolver verification (LOCAL, $0, C6-legal)

**Directive:** R17 OBJ-I / prog-synthesis R3. **Filed:** 2026-07-23.
**Verdict headline: tr87 RE-ENTERS the certified-resolver set; wa30 and ka59 do NOT.**

A *fixed* external hypothesis — Schema-harness's backtest-certified hidden-phase
law for each aliased game, extracted verbatim from the released world_model_v5.py
snapshots BEFORE touching our data — is verified against ALL of our streams per
game with **no fitting and no selection budget**. Legitimate under C6 because the
hypothesis is externally sourced and fully specified before measurement; a fixed
hypothesis consumes no selection budget, so the certificate is the **pooled
augmented determinism over all streams**, no train/test split required.

**Certificate bar (identical to sealed holdout_report.json):** augmented
determinism >= 0.99 AND Wilson 95% lower bound >= 0.95.

**Machinery:** the sealed audit transition extraction, alias_stats (determinism)
and wilson_lb are imported VERBATIM from scripts/latent_state_audit.py; we supply
only the fixed key. Streams reproduce the holdout_report.json
benchmark-engine-version set (drift versions, the post-report sentinel_eval_v1
stream, and the *_tool_sentinel_events.jsonl sidecar excluded from the primary
run; sentinel_eval_v1 re-added in an appendix, where it only strengthens the fixed
laws). Reproducer: runs/schema_fixed_resolver/verify_fixed_resolvers.py -> results.json.

---

## Law extraction (exact file + line; no improvisation)

Source root:
kaggle-data/schema_traces/claude_fable_opus/claude-opus-4-8_max_<game>_100.0/world_model_v5.py
(backtest-certified; Schema mining report confirms engine versions hash-identical
to our audited set, so these are laws for our exact game class).

- **tr87 — floor(n/2)** — world_model_v5.py:19: "* row 63 = budget bar:
  floor(n_actions / 2) cells of colour 4, filling from the right." and :262-265:
  "budget bar: one colour-4 cell per K actions. K = 2 on levels 0-4; level 5 bar
  is slower ... st['K'] = 2 if (lv is None or lv < 5) else 4". -> hidden sub-tick
  phase = n mod K(level), K=2 lv0-4, K=4 lv5; n=actions-since-level-start, all
  actions tick. **Fixed key = n % K.**

- **wa30 — mod-rate** — world_model_v5.py:98-111 (_BAR table) and :114-118
  (_bar_params): filled = (mult*n + off) // D with a PER-LEVEL (D,off,mult) table
  {0:(3,2,1),1:(1,0,1),2:(3,1,2),3:(3,1,2),4:(2,0,1),5:(7,0,6),6:(2,1,1),
  7:(7,3,3),8:(1,0,1)} (default (2,1,1)). -> hidden sub-tick phase = n mod D(level);
  all actions tick (box-push game, bar is a pure move counter). **Fixed key =
  n % D(level)** (our streams sit at harness level 1 = Schema level 0 -> D=3).

- **ka59 — parity-inverted** — world_model_v5.py:31: "BAR (row 63): zeros =
  round(64*n/budget); the budget is PER-LEVEL." / :51-52 _bar_for(n,b) =
  (2*64*n+b)//(2*b) (= round(64n/b)); FUSE semantics :20-22: "it shrinks one unit
  per MOVE (clicks do NOT tick it) and WRAPS to full at 0." -> hidden phase = move
  parity with clicks (ACTION6) excluded. **Fixed key = move_count % 2**,
  move_count = non-click, non-RESET actions since level start.

All three laws extracted cleanly and unambiguously; no improvisation required, so
no game was excluded for engine-version/law ambiguity.

---

## Results (primary: holdout_report.json stream set, fixed law, no split)

| game | law | streams | base det | fixed-aug det | visits | Wilson LB | bar (0.99 / 0.95) | verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| **tr87** | floor(n/2)=n%K, K=2 | 8 | 0.9103 | **1.0000** | 143 | **0.9738** | **PASS / PASS** | **RE-ENTERS CERTIFIED-RESOLVER SET** |
| wa30 | mod-rate=n%D, D=3 (lv0) | 8 | 0.7388 | 0.9057 | 106 | 0.8350 | FAIL / FAIL | STAYS ALIASED-UNRESOLVED |
| ka59 | parity-inverted=move%2 | 7 | 0.7407 | 0.9464 | 56 | 0.8539 | FAIL / FAIL | STAYS ALIASED-UNRESOLVED |

**Appendix (same laws + post-report sentinel_eval_v1 stream; more support only
helps a fixed law):** tr87 -> det 1.0000, LB **0.9792** (181 visits, still
PASS/PASS); wa30 -> det 0.9040, LB 0.8516 (FAIL); ka59 -> det 0.9524, LB 0.8839
(FAIL). Adding support does not rescue wa30/ka59 and only firms tr87.

---

## tr87 — the re-entry, in detail

Under the fixed Schema floor(n/2) law, tr87 pooled augmented determinism over all
8 streams is a perfect **143/143 = 1.000000**, Wilson LB **0.9738 >= 0.95**. It
clears BOTH prongs of the sealed certificate bar.

This is the exact bar the in-sample 4/4 held-out split FAILED in the sealed section-1
record: "tr87 parity: held-out det 1.000 but Wilson LB 0.927 < 0.95 on 49 visits."
The failure was **support starvation from the split**, not wrong physics — the
section-11 Schema annex already anticipated this ("not certifiable from 7-8 streams,
not no phase mechanism exists"). As a fixed external hypothesis the split is
illegitimate (no selection to protect against), so the full pooled support (143
visits vs the split 49) is available and the Wilson LB clears 0.95 with margin.
**tr87 struck EWM channel is no longer hostage to a free test: the fixed law
certifies it.**

Law/key equivalence: at Schema level 0 (our display level 1, where all tr87 streams
sit), K=2, so floor(n/2) => phase = n % 2 = parity — the same mechanism class the
in-sample audit named, now confirmed by the external law and certified on full
support.

## wa30 and ka59 — why the fixed law does NOT certify (honest, per-game)

Both fail the bar, and the reason is instructive and is exactly the C6 point — the
literal external law is not the same key the in-sample selection picked:

- **wa30** — literal law n % 3 (Schema D=3 at level 0): det **0.9057**, LB 0.835 ->
  FAIL. The in-sample audit reported mod4 (det 1.000) — but mod4 is NOT the wa30
  level-0 law (Schema says D=3). Diagnostic confirms: n%4 (the in-sample pick)
  gives det 1.000 / LB 0.957, while the fixed law n%3 gives det 0.906 on the same
  streams. The in-sample selection found an over-fit modulus that happened to
  shatter our support cleanly; the honest external law does not certify on our
  data. (Plausibly our wa30 streams mix level-0 D=3 with a few post-level-up frames
  or irregular off-skips the fixed single-D key cannot absorb; Schema model
  self-corrects off online, which a fixed key by construction cannot.)

- **ka59** — literal law move % 2 (clicks excluded, per the Schema docstring): det
  **0.9464**, LB 0.854 -> FAIL. The in-sample audit reported parity (t%2, clicks
  INCLUDED) at det 1.000. Diagnostic confirms click-inclusive n%2 gives det 1.000
  but Wilson LB only **0.930 < 0.95** (51 visits) — so even the click-inclusive
  variant fails the certificate on support, and the literal click-EXCLUDING law
  fails on determinism too. ka59 fails under both readings. (The click-exclusion
  the Schema law prescribes actively lowers determinism on our streams — a real,
  reportable discrepancy, not a bug: our ka59 traces carry only 15 clicks, too few
  to resolve which reading our engine version follows.)

Reporting both readings per the directive rather than improvising: neither wa30 nor
ka59 re-enters the certified set under the fixed external law.

---

## Consumer consequence (provisional; no seal moves)

- **tr87** re-enters the certified-resolver set (ALIASED-RESOLVABLE, hidden-phase
  floor(n/2) class) on a fixed, C6-legal certificate — a candidate to re-admit its
  struck section-7 EWM channel and its banking key via the sealed re-entry path.
  This is a fixed-hypothesis certificate, NOT the in-sample selection the section-1
  audit banned; it should be filed as a sealed re-entry input, not a silent
  un-strike. (Its EWM value is separately gated by D4: a phase-resolved tr87 is a
  DEPTH enabler, and tr87 frontier is L1 — a new-clear worth its small L1 weight —
  so re-entry is a correctness win more than a score win.)
- **wa30, ka59** remain ALIASED-UNRESOLVED; the sealed section-1 verdicts stand.
  The external laws exist and are certified on Schema support, but do not certify on
  OUR support under the honest fixed key — exactly the "support-starvation, not
  wrong-physics" reading the section-11 annex sealed.
- Nothing here moves a sealed threshold; it is a $0 re-entry-path input.
