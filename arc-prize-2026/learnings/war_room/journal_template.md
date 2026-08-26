# Journal row template — `kaos experiment log` at seal time

**Written 2026-08-16 after logging exp_id 1–9 (the first nine rows pushed to the Attraktor public registry). Follow this instead of reconstructing the reasoning.**

**Verdict string.** The bench mints by PREFIX: `ACCEPT…` → ACCEPT, `REJECT…` → REJECT, anything else → VOID — so the prefix is a routing decision, not decoration. Use `REJECT:` for any verdict where the sealed gate legitimately fired (REFUTE, NO-PROMOTE, KILL, FAIL), then the sealed verdict **verbatim** with its key numbers inline (`REJECT: REFUTE-2x (decisive) - mean dlc +0.0667 <= +0.2500: …`). Use `VOID:` when the run produced **no evidence about the mechanism in either direction**: infra death, parked lanes, and — critically — **vacated verdicts**. A retraction is journaled as `VOID: VACATED <date> (<forum>) - original '<verdict>' RETRACTED: <why the instrument was broken>; <re-baselined numbers>; ruling verbatim` — the vacate is the HEADLINE, so the row is structurally incapable of being read as a clean kill (`pull()` serves only ACCEPT; VOID can never be served as a finding). Same honesty rule for instrument defects: if the kill was a mis-specified gate, say `INSTRUMENT MIS-SPECIFICATION, not a mechanism finding` **in the verdict string**, with the opposite-sign measurement, so the row cannot teach the wrong lesson. The schema has no "superseded" field — that state lives in the verdict string by design.

**Metadata, gates, arms, results_path.** `--metadata-json` carries the structure that makes a null informative; the keys that worked: `prereg` (file + seal date + verdict date), `mechanism`/`design` (what was actually intervened on), `sealed_lines` or `sealed_reading_rule` (thresholds as pre-registered), `measured` (the numbers), `robustness` (floor/LOO/sensitivity checks), and **`why_informative` — mandatory: one paragraph on what this null rules out** (e.g. "delivery was proven, so the null is about the agent, not the pipe"). `--gates-json` is a list of `{gate, name, passed, kill, detail}`; `--arms-json` per-arm aggregates. `--results-path` points at the REAL on-disk artifact only: a score/benchmark JSON where one exists; the pull directory if no JSON was ever produced (b122); the postmortem md for an authoring-defect death. **Never point at a file that does not exist.** Run from the campaign repo root so `git_sha` auto-fills the campaign HEAD, and pass `--db f:/kaggle/arc-prize-2026/kaos.db`. Then: `kaos bench harvest` → `kaos bench validate --no-model` → `kaos bench push` (validate is required between harvest and push — it is what mints the record; token env-only via `KAOS_BENCH_TOKEN`, never in a file). Before push remember `publish_scope=public_queue` is PUBLIC: no secrets, no unshipped-arm details, sealed reads only.

```bash
cd /f/kaggle/arc-prize-2026
uv run --project /f/kaggle/kaos kaos experiment log \
  --db f:/kaggle/arc-prize-2026/kaos.db \
  --name "q38-engine-swap" --family probe \
  --verdict "REJECT: REFUTE-2x (decisive) - mean dlc +0.0667 <= +0.2500: the 'consistent 2x on the local 25' claim is NOT reproduced on our harness (21 levels over 25 games vs baseline 18/19/21)" \
  --metadata-json rows/q38_meta.json \
  --gates-json rows/q38_gates.json \
  --arms-json rows/q38_arms.json \
  --results-path "runs/kernel_pulls/q38_v2/benchmark.json"
```

---

## ★ REGISTRY ADMISSION REQUIRES THREE KEYS — learned the hard way 2026-08-18

`kaos bench push` **refuses** any experiment record whose metadata lacks consumable knowledge:

> `rejected: no consumable knowledge (add mechanism/summary/lesson to the experiment metadata, or template/description for skills)`

**The check is SERVER-SIDE at the registry** (there is no local `consumable` check anywhere in the
kaos tree — do not grep for one), so `harvest` and `validate --no-model` both pass, the record mints
cleanly, and the refusal only appears at `push`. `harvest` reporting `e1_passed: 1` and `validate`
reporting `minted VOID` are **NOT** evidence of admission. *Standing lesson again: silence — or a
green intermediate stage — from an automation is not success.*

**Required (any of, and in practice write all three):** `mechanism` · `summary` · `lesson`.
Keep `why_informative` as well; it is what makes a null worth serving. The keys used before 08-18
(`finding`, `why_it_matters`, …) are **not** recognised, which is why exp_id **12** and **13** were
both refused and had to be re-logged as **14** and **15**.

- Write for a READER IN ANOTHER WORKSPACE who has none of our context: name the mechanism concretely
  (file:line, the actual gate expression, the score formula) rather than by our internal arm nickname.
- `lesson` should be transferable advice, not a restatement of the result.
- **The journal is APPEND-ONLY and metadata is INLINED at log time** — editing the JSON file after
  logging changes nothing. A malformed row can only be superseded by a new row; say so in the new
  verdict string (`re-log of exp_id N … this row supersedes it`), since the schema has no
  `superseded` field. The old record stays permanently in the refused list on every subsequent push;
  that is expected and is not a new failure.

**CERTIFIED-verdict snapshot step (coordinator-adopted 2026-08-22, part of the verdict procedure — do not skip).** Any sealed verdict that CERTIFIES an artifact as head-eligible must, in the same session: (1) snapshot the exact notebook bytes to `runs/certified_artifacts/<slug>__v<N>__<code_sha16>.ipynb` (immutable, git-tracked; the verdict row records the sha); (2) queue entries that self-trust the artifact set `preflight_mode: trusted-fork` with `upstream` = the SNAPSHOT path (never a mutable staging dir). Limitation on the record: self-trust holds only while the certified version is the slug's LATEST — promoted artifacts keep dedicated slugs. Design: `learnings/war_room/selftrust_preflight_design_2026-08-22.md`.
