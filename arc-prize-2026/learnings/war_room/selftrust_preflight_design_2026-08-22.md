# SELF-TRUST PREFLIGHT — design note for review (2026-08-22)
**Finding: NO CODE CHANGE IS NEEDED.** `preflight.py --mode trusted-fork` already accepts a LOCAL `.ipynb` as `--upstream` ("an upstream that resolves to an existing local .ipynb path is read from disk instead of pulled", run_trusted_fork docstring), and `daily_submit.run_preflight` passes the queue entry's `upstream` through verbatim. **Self-trust = trusted-fork against our own certified bytes.** Proven live this morning: `--kernel canivel/arc3-q38-field-eval --upstream notebooks/q38-field-eval/arc3-q38-field-eval.ipynb` → **ALLOW, T1–T4 all OK** (T3 = remote code identical to our staged bytes; T4 = latest build COMPLETE).

## Convention (the actual proposal)
1. **At sealed-verdict time**, when an artifact is certified head-eligible, snapshot its exact notebook to an immutable path: `runs/certified_artifacts/<slug>__v<N>__<code_sha16>.ipynb` (git-tracked; the verdict row records the sha). The SNAPSHOT, not the mutable staging dir, is the trust anchor — staging dirs get rebuilt (today `notebooks/q38-private-eval/` holds edge-2 bytes, not the certified base; pointing at staging would false-block or worse).
2. **Queue entry**: `preflight_mode: "trusted-fork"`, `upstream: "<absolute path to the snapshot>"`. Zero daemon changes.
3. `queue.py add <kernel> <ver> "<msg>" <snapshot-path>` already works (the 08-20 upstream arg accepts any string).

## Known limitation, declared
T1 pulls the slug's LATEST version and T4 reads the LATEST build status ⇒ **self-trust certifies a kernel only while the certified version IS the latest** (multi-version slugs like `arc3-q38-private-eval` break it whenever a newer variant has been pushed). Mitigations, in preference order: (a) dedicated slug per promoted artifact (the field-eval pattern — already our promoted-artifact norm); (b) future `--version`-pinned pull support in T1/T4 (a real preflight change, separate review if ever needed).

## Negative controls run
- Staging-dir drift: `--upstream notebooks/q38-private-eval/...` (currently edge-2 bytes) vs the v3-latest kernel → would pass T3 only because v3 IS edge-2 — demonstrating why the IMMUTABLE snapshot, not staging, must be the anchor (convention rule 1).
- Wrong bytes: any sha-mismatched snapshot → T3 FAIL (the existing code path; exercised by the graft-lane's 08-21 near-tag incident in reverse).
