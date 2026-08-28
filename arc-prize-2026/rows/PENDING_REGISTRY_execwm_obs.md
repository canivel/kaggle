# PENDING REGISTRY ADMISSION — exec-WM observation-layer verdict (2026-08-27, Mac)

**BLOCKED ON THIS BOX, NOT SKIPPED.** `kaos experiment log` / `kaos bench harvest|validate|push`
cannot run on the MacBook: `../kaos` is an **empty directory** — a known, documented migration gap
(`MIGRATION_MACBOOK.md` item 5: *"Clone + `uv sync` + `kaos.yaml`"*; the notes already flag it at
lines 89 and 109 as the reason a nightly rail must not depend on it). `scripts/bench_token.sh` also
does not exist yet (item 2 — recreate, never commit).

Per the KAOS-native mandate a verdict is not DONE until admitted to the registry, so this row is
**staged, not closed.** Everything needed is on disk; run the block below from the Windows box, or
from the Mac once `../kaos` is cloned and `uv sync` has run.

```bash
cd /f/kaggle/arc-prize-2026        # or ~/Projects/kaggle/arc-prize-2026 post-clone
uv run --project /f/kaggle/kaos kaos experiment log \
  --db f:/kaggle/arc-prize-2026/kaos.db \
  --name "execwm-observation-layer" --family probe \
  --verdict "REJECT: DATA-STARVATION REFUTED - 1243/2394 (51.9%) of exec-WM move-action transitions are already clean single-sprite translations, so the observation layer is NOT starved and the 08-27 re-scope premise is false; the loss is inside the extractor, in two over-strict gates - GATE B (a move must explain EVERY interior diff cell) discards 641 transitions (26.8%), GATE A (len(deltas)==1) discards a rule on one dissenting delta and a union gate recovers +12 rules across 6 games with none lost, taking m0r0 0->4 and tu93 0->3 across MIN_VERIFIED_MOVES=2. THIRD successive wrong diagnosis of the same number (probe budget, then BREAK clustering, then starvation); each fell to the artifact, none to the previous summary." \
  --metadata-json rows/execwm_obs_meta.json \
  --gates-json rows/execwm_obs_gates.json \
  --results-path "runs/execwm_obs_replay.json" \
  --lock-sha256

call scripts\bench_token.cmd            # Windows; on Mac: . scripts/bench_token.sh
cd /f/kaggle/kaos
KAOS_DB=f:/kaggle/arc-prize-2026/kaos.db uv run kaos bench harvest   --config-file /f/kaggle/kaos/kaos.yaml
KAOS_DB=f:/kaggle/arc-prize-2026/kaos.db uv run kaos bench validate --no-model --config-file /f/kaggle/kaos/kaos.yaml
KAOS_DB=f:/kaggle/arc-prize-2026/kaos.db uv run kaos bench push      --config-file /f/kaggle/kaos/kaos.yaml
```

`--config-file` is mandatory from this repo (08-27 finding: without it `push` silently reports
"local-only mode", `pushed: 0`, **exit 0**). The three consumable keys (`mechanism`, `summary`,
`lesson`) are present in `rows/execwm_obs_meta.json` — the server-side check that rejected rows on
08-18 will pass. `--results-path` points at a file that exists (verified).

**Also unavailable on this box for the same reason:** `kaos bench rejections` (the CONSUME step at
session start) and `scripts/kaos_ingest.py`'s downstream `dream run`. The consume step was therefore
**not performed today** — recorded here rather than quietly omitted.
