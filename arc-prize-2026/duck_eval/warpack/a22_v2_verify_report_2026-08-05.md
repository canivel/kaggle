# A22 compaction v2 — build-verify report — 2026-08-05

**Scope:** post-hoc verification of the v2 `compaction_patch.py` written 2026-08-04
(build session terminated before any verification ran). Sealed authority:
`learnings/war_room/a22_compaction_v2_prereg_2026-08-04.md` (§2 mechanism spec,
§3 canary, §6 process constraints). NO Kaggle pushes performed in this session
(§6: the orchestrator verifies and pushes). NO submission, NO queue change, $0.

**File under audit:** `duck_eval/warpack/_kaggle_dataset/compaction_patch.py`
- sha256 as found (pre-audit): `9fd114b873bc535f92ee15ee8ed54f842463121b27d74b2c0b56077e0b225d36`
- sha256 after in-compliance edits (final, the push candidate):
  `5d8579ad0960312629c4804a27e99a905e6ffec601673b81a6b26e13ace1804f`

---

## 1. Spec-compliance table (prereg §2, line refs = final file)

| §2 point | verdict | evidence (line refs) |
|---|---|---|
| §2 VERSION → v2; banner `compaction v2: ACTIVE` | COMPLIANT | L96 `VERSION = "v2"`; apply() banner L856-867 (smoke-verified string) |
| §2 same flag `COMPACTION=1` / kill `COMPACTION_DISABLE=1` / kill beats flag | COMPLIANT | L125-130, L844-848 (kill checked first) |
| §2 vanilla-fallback-on-any-failure | COMPLIANT | apply() blanket except L870-872; every wrapper try/except falls back to the stock method (trim wrapper → `_orig_trim` L805-809); smoke V10 poisons the store and gets stock behavior |
| §2 zero LLM calls in eviction path | COMPLIANT | no requests/urllib/httpx/openai import anywhere (smoke V11 source scan) |
| §2 NO locks anywhere (forge_v35) | COMPLIANT | zero `threading`/`Lock` occurrences; state on the ToolAgent instance (`_get_store` L571-577, `__slots__` store L362-401) |
| §2 no game-id logic | COMPLIANT | `_store_key` L157-168 is a filename-stem label only; no game branching |
| §2 `ledger_core.py` NOT modified | COMPLIANT | mtime 2026-07-16 12:48 (pre-dates v1); sha256 `671e883ce054…17bff3` byte-identical to canonical twin `duck_eval/ledger/ledger_core.py`; untracked dir so no git delta possible; all v2 hygiene applied inside compaction_patch at ingest/render time (L174-205, L502-552) |
| §2.1 pins: system prompt / most-recent scientist-note carrier (≥2 labels) / most-recent reasoning block / preserve_recent tail | COMPLIANT | system structurally outside history (L755-757); `_is_scientist_note` ≥2 label match L247-254 (`_SCI_LABEL_RE` L242-244 carries all 7 labels); `_has_reasoning` L257-261; pins = max index L296-302; preserve tail via `guard = n - preserve_recent` L290-293 |
| §2.1 eviction order 1-5, oldest-first within class | COMPLIANT | `_select_evictable_block` L285-356: class1 stale episodes (skip newest, L325-331); class2 older user blocks, never the current-frame carrier (`bi == last_user` skip L333-338), head user block evicts as the full turn-cycle span so the head-invariant never eats uncaptured messages (L305-315); class3 non-pinned assistant text L339-345; class4 newest episode L346-350; class5 last resort — pins yield (`pinned()` not consulted) L351-355. Smoke V3 walks the exact 1→2→3→4→5 sequence deterministically |
| §2.1 final user block never an eviction candidate (classes 1-4) | COMPLIANT | class2 `bi == last_user` skip; classes bounded by `hi < guard` |
| §2.1 both capture points remain capture points; head-invariant removals captured | COMPLIANT | token-trim capture in the region loop L779-781; head-invariant drops captured+classed L786-789; persistence-cap capture L736-744; vanilla `_drop_oldest_history_block` wrapped for the `_force_reduce_messages`/fallback path L701-718 |
| §2.2 digest renders ONLY refuted + gated FACTs + meta tail; NO ACTIVE/CONFIRMED lines | COMPLIANT | `render_digest` L502-560: refuted L508-509, gated facts L510-511, EVICTED/ACTION-EFFECTS/PROGRESS meta L544-559; no hypothesis-status lines other than REFUTED |
| §2.2 FACT hygiene gate (hedge prefixes actually/wait/maybe/i think, case-insensitive; must end '.'/'!' after stripping trailing quotes/brackets; truncations + questions fail) | COMPLIANT | `_HEDGE_PREFIX_RE` L174-175, `_fact_hygiene_ok` L179-189; smoke V1 13 accept/reject cases incl. punctuation-prefixed hedge and mid-sentence truncation |
| §2.2 refuted NEVER elided; no "+N more"; REFUTED>FACT>meta priority; newest-first; overflow drops OLDEST lines silently | COMPLIANT (after edit E1) | one `emit()` per record, no count line anywhere; refuted `reversed()` newest-first L530; **E1** made FACTs newest-first too (L537) and stops each record loop at first overflow so the drop is exactly the oldest contiguous tail (L526-540). Smoke V6: 30-refuted overflow renders a contiguous newest tail, zero elision |
| §2.2 header softened + non-quotable; no "do NOT re-verify" | COMPLIANT | `_DIGEST_HEADER` L105-110: "treat as prior, not proof; re-testing is allowed; internal memo -- do not quote or restate" |
| §2.2 anti-self-ingestion strip before extraction | COMPLIANT | `_DIGEST_ECHO_RE` L195-198 (FACT F\d+: / REFUTED\|ACTIVE\|CONFIRMED H\d+ / marker), applied in `ingest_message` L441-443; own digest never ingested (marker guard L426-427; stale digests stripped pre-trim L761). Smoke V2 end-to-end: echoed FACT never re-seeds the ledger |
| §2.2 empty gate ⇒ inject NOTHING (no header-only digests) | COMPLIANT | `render_digest` returns "" on no-content or no-surviving-records L505-513; injection only when digest non-empty L802-804. Smoke V7: record-free eviction injects nothing, digest_tokens=0 |
| §2.2 reserve only when earned (default 1000, env `COMPACTION_RESERVE_TOKENS`; pre-trim non-empty test; records first appearing during a no-reserve trim deferred to next trim) | COMPLIANT | `_reserve_tokens` L139-145; `reserve_applied = (not stuck) and bool(store.render_digest())` computed PRE-eviction L764-768; digest rendered post-trim only if reserve_applied L792. Smoke V5: first (earning) trim injects nothing + `reserve_applied=0`; next trim injects. (Note: L145 clamps env override to ≥200 — env-robustness floor, does not affect the tested default) |
| §2.3 `COMPACTION_RETAIN` defaults 0; mirroring installed ONLY when =1; `retained_reasoning_msgs` stays, expected 0 | COMPLIANT | `_retain_enabled` default "0" L133-136; `if retain:` guards the `_chat_completion` wrapper install L815-827; smoke V0 asserts the wrapper is NOT installed by default and V9 (subprocess) that RETAIN=1 installs it + banner `mirroring=ON`; event carries `retain` + `retained_reasoning_msgs` (0 in all default-path smoke events) |
| §2.4 stuck rubric: last K executed actions all `board_changed==false`, mechanical parse, fewer-than-K ⇒ not stuck, env `COMPACTION_STUCK_K`, default K=5 | COMPLIANT | `_recent_board_flags`/`_is_stuck` L211-236 (`len(flags) >= k and not any(flags[-k:])`); `_stuck_k` default 5 L148-154. Smoke V4: 8 cases incl. fewer-than-K, change-within-K, non-tool/malformed ignored, env override |
| §2.4(a) persistence cap DEFERRED outright while stuck (no eviction, no event) | COMPLIANT | `compaction_keep_recent` L724-747: stuck + would-cut ⇒ return the uncut list, `stuck_suppressed += 1`, nothing ingested |
| §2.4(b) no reserve subtracted / no digest injected while stuck | COMPLIANT | `(not stuck)` term in `reserve_applied` L768; smoke V8: refuted-bearing store + stuck trim ⇒ no digest |
| §2.4(c) budget-forced evictions while stuck: still occur, still region-aware, captured, NO event; counts flush into next non-stuck event; `stuck_suppressed` counts every suppressed cut/emission opportunity | COMPLIANT (after edit E2) | eviction loop runs regardless of stuck (physics) L772-783; event gated on `not stuck` L794-795; **E2** widened the stuck counter to `evicted_this_call or store.pending_msgs > 0` L796-801 so a suppressed event emission (pending counters held while stuck) is also counted, per the §2.5 sentence. Smoke V8: stuck trim evicts + emits nothing; next non-stuck trim emits ONE event carrying the exact accumulated `evicted_msgs` and `stuck_suppressed ≥ 1` |
| §2.5 one-flag graft / graft cell 12 via `--compaction` / kill switch / greppable `COMPACTION ` events + per-game sidecars | COMPLIANT | `EVENT_ANCHOR` L100; `_emit_event` stdout line L626-638 + `*_compaction_events.jsonl` sidecar L639-666; builder regression below confirms the graft cell |
| §2.5 NEW event fields `ev_episode`/`ev_user`/`ev_reasoning`/`ev_fallback`, `stuck_suppressed`, `reserve_applied`, `gated_facts`, `retain` | COMPLIANT | all present in both the stdout line L626-638 and the sidecar record L641-661 (smoke V5 asserts every field, stdout + jsonl) |
| §2.5 `episodes` increments only on trims that actually evicted | COMPLIANT | `if evicted_this_call: store.episodes += 1` L790-791; smoke V5 non-evicting-trim check |
| §3 canary strings (`compaction v2: ACTIVE`, `COMPACTION=1` stamp, `COMPACTION ` events, RETAIN-OFF banner) | COMPLIANT | banner + `mirroring=OFF (v2 default)` smoke-verified; cell-2 stamp verified in the built notebook |

**Compliant-with-note (implementation details inside spec latitude, unchanged):**
- Class 1 also sweeps orphan tool blocks (`kind == "other"`, L326-331) — tool
  results whose assistant turn is already gone are stale-episode debris; no
  named class fits better and they are fully captured.
- Head-invariant drops (L786-789) and vanilla-drop-path captures (L713) are
  classed `fallback` — the spec names only classes 1-4 + fallback, and these
  are exactly the outside-the-region-model removals.
- Class 5 still honors the `preserve_recent` tail (`hi < guard`, L353-355):
  vanilla `_drop_oldest_history_block` never cuts below `preserve_recent`
  either, so no brick-risk is introduced; pins (scientist note / reasoning)
  do yield at class 5 as specified (smoke V3).

## 2. Edits made this session (all INTO sealed-spec compliance; documented per task rules)

- **E1 (`render_digest`, §2.2):** FACT lines were rendered oldest-first (an
  unsealed "early facts are durable" rationale) and both record loops
  continued past a budget overflow, so the dropped set was not guaranteed to
  be the oldest lines. Changed to `reversed(gated_facts)` (newest-first, same
  as refuted) and `if not emit(...): break` in both record loops — the sealed
  text "rendered newest-first so if the reserve is ever exceeded the oldest
  lines drop silently" now holds exactly. REFUTED>FACT>meta priority
  unchanged.
- **E2 (`compaction_trim`, §2.4/§2.5):** `stuck_suppressed` incremented only
  when a stuck trim evicted; a stuck trim that merely HELD pending counters
  (a suppressed event-emission opportunity) went uncounted. Condition widened
  to `stuck and (evicted_this_call or store.pending_msgs > 0)`. Telemetry
  only (feeds the M2 attribution split); no behavioral path changed.

No other lines of the patch were modified. `ledger_core.py`,
`build_eval_notebook.py`, and every other dataset file untouched (mtimes:
builder 2026-08-01 12:00, ledger_core 2026-07-16 — both pre-date the 08-04
build session).

## 3. Smoke results

New v2 smoke: `duck_eval/warpack/compaction_smoke_v2.py` (sha256
`7d31def6a5091f95bbf28719b36dac3ac8768e0d7111b994576f6bb3a2aa5ea2`),
patterned on the v1 runnable smoke (`compaction_smoke.py`, kept unmodified as
the v1-mechanism record). Run: `uv run python
duck_eval/warpack/compaction_smoke_v2.py`.

**Result: 142 passed, 0 failed (100% PASS — the §6 GO requirement).**

Sections V0-V12 cover every §2 design point: flag/kill/idempotency/v2
identity + RETAIN-OFF default (V0); 13 hygiene-gate cases (V1);
anti-self-ingestion incl. end-to-end no-reseed (V2); pin detection + the full
deterministic class 1→2→3→4→5 walk incl. head-span pin skip, last-resort pin
yield, preserve-tail immunity (V3); stuck rubric incl. fewer-than-K and env
override (V4); end-to-end trim with deferred reserve, event + sidecar field
audit, stale-digest hygiene, episodes discipline (V5); refuted-never-elided /
newest-first / oldest-tail overflow / REFUTED-starves-FACT priority / hedged-
FACT gating (V6); empty-gate-inject-nothing (V7); all three while-stuck
behaviors incl. flush-into-next-event exact-count check (V8); RETAIN sub-arm
incl. subprocess RETAIN=1 install proof (V9); vanilla fallback on wrapper
failure (V10); no-locks/no-HTTP/ledger_core-byte-identity (V11); builder
--compaction artifact checks (V12). One intermediate smoke-harness fix during
development (V6 initially asked for FACT ordering out of a digest whose
budget was fully consumed by 30 refuted lines — the observed starvation is
the spec's own priority rule; test restructured, no patch change involved).

## 4. Builder regression (sha256, rebuild vs pre-existing artifact)

Builder: `duck_eval/warpack/build_eval_notebook.py` (byte-unchanged since
2026-08-01, before the v2 build session). Sentinel rebuilt with
`--sentinel --sentinel-budget 150` (the C7-as-amended flag the live artifact
carries, per its cell-2 stamp).

| mode | artifact | pre-rebuild sha256 | post-rebuild sha256 | verdict |
|---|---|---|---|---|
| default | duckwar-eval/arc3-duck-war-eval.ipynb | `8445e0fc…d287b3` | same | BYTE-IDENTICAL |
| default | duckwar-eval/kernel-metadata.json | `d29c15bc…a948a5` | same | BYTE-IDENTICAL |
| --sentinel (150) | ducksentinel-eval/arc3-duck-sentinel-eval.ipynb | `7adf6bec…364d15` | same | BYTE-IDENTICAL |
| --sentinel (150) | ducksentinel-eval/kernel-metadata.json | `ff4872eb…85012f` | same | BYTE-IDENTICAL |
| --w0 | duckw0-eval/arc3-duck-w0-continuation-eval.ipynb | `54e5341b…c6925e` | same | BYTE-IDENTICAL |
| --w0 | duckw0-eval/kernel-metadata.json | `732fcef7…c8f4d` | same | BYTE-IDENTICAL |
| --compaction | duckcompaction-eval/arc3-duck-compaction-eval.ipynb | `1c4e51eb…50edb` | same | BYTE-IDENTICAL (= the sha the 08-02 push report pinned for pushed kernel version 1) |
| --compaction | duckcompaction-eval/kernel-metadata.json | `fcd4eb83…1ff977` | same | BYTE-IDENTICAL |
| --a17-canary | a17-canary/arc3-a17-72b-canary.ipynb | `d27ac0a4…684096` | `de02e768…8234b2` (discarded) | N/A — see below; artifact RESTORED to `d27ac0a4…684096` |
| --a17-canary | a17-canary/kernel-metadata.json | `ed03d138…c69767c` | `381a90f0…b61d56c` (discarded) | N/A — artifact RESTORED to `ed03d138…c69767c` |

**--a17-canary N/A ruling:** the committed on-disk artifact is NOT a
`build_eval_notebook.py` output any more — it is the A17 v6/v7 full-window
**dataset-weights** build (cell-2 banner: `seed=2 …
mode=throughput-canary-v6-dataset-weights … weights=DATASET
canivel/qwen25-vl-72b-awq (model_sources route DEAD, silently dropped at push
07-25/07-26)`, + fenced-recovery graft), produced by the dedicated
`duck_eval/a17/build_v6_full_window.py` lane (commits 66bc223/dca3e50); git
history holds no pre-v5 base-builder output to byte-compare against. The base
builder still emits the old model_sources-route composition by design. The
regression gate's intent — prove the v2 build session did not drift the shared
builder — is discharged by (a) the builder file being byte-unchanged since
08-01 and (b) 8/8 artifacts of the other four modes byte-identical. The
momentarily overwritten a17 files were restored from git and re-verified to
the exact pre-rebuild shas. (A17/72B lane is DEAD per B2a 07-30 regardless;
nothing in this session touches it.)

**Compaction notebook note for the orchestrator:** the `--compaction`
notebook is byte-identical to already-pushed kernel version 1 — the v1→v2
mechanism swap lives entirely in the arc-war-kit dataset file. Per prereg §6
the push order is: `canivel/arc-war-kit` dataset version (new
compaction_patch.py, sha `5d8579ad…e1804f`) with pull-back byte-audit, THEN
the kernel seed push, THEN the runtime `compaction v2: ACTIVE` banner check
(§3 canary; a stale-dataset run would banner `v1` and is VOID for v2 —
feedback_kaggle_dataset_code_sync).

## 5. Verification checklist (task order)

- [x] Prereg read first; §2/§3/§6 treated as binding
- [x] Line-by-line §2 audit (table above): all points COMPLIANT after 2
      documented into-compliance edits; no deviation remains that changes the
      tested mechanism
- [x] No threading locks; zero LLM calls in eviction path; vanilla fallback
      on any failure; ledger_core.py untouched (sha-verified vs twin)
- [x] v2 smoke written (`compaction_smoke_v2.py`) covering every §2 design
      point; `uv run python` ⇒ **142/142 PASS**
- [x] Builder regression: default/--sentinel(150)/--w0/--compaction
      byte-identical; --a17-canary N/A (superseded artifact; restored)
- [x] --compaction rebuild verified: cell-2 `COMPACTION=1` stamp + seed
      banner present; cell-12 imports compaction_patch (no warpack), (f)
      continuation default rides; dataset patch carries `VERSION = "v2"` and
      the `compaction v2: ACTIVE` banner (runtime-verified in smoke V0)
- [x] No Kaggle pushes, no submission, no queue change, ledger_core.py not
      modified

---

**VERDICT: GO**
