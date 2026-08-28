#!/usr/bin/env python
"""
PULL-COMPLETENESS GATE.  Run this BEFORE any scorer touches a kernel pull.

WHY.  2026-08-27: `kaggle kernels output` returned **exit 0 with a partial file set** -- pull 1 of
the P2 artifact silently omitted BOTH large logs (256KB kernel, 360KB vLLM); an identical re-pull
got them.  Scored against the partial pull the scorer reported INFRA DEATH / "served model absent"
/ `cert_facts: {}` on a completely healthy 2h13m run.  A download race nearly killed a certified
arm.  The tell, and the rule this file encodes:

    AN INSTRUMENT REPORTING A SICK SUBJECT WHILE REPORTING ZERO FACTS ABOUT IT
    IS DESCRIBING ITSELF.

`exit 0` from the Kaggle CLI is NOT evidence of a complete download.  Only the file set is.

WHAT IT CHECKS
  1. A kernel log exists and is NON-EMPTY.  Searched beyond a top-level `*.log` glob, because
     that glob is exactly what the partial pull defeated.
  2. The served model is recoverable from at least one INDEPENDENT source.  `taaf_setup_env.json`
     and `prompts/*.log` both survive partial pulls and both carry it, so a missing top-level log
     is no longer sufficient to conclude "served model absent".
  3. No zero-byte file among the artifacts that matter (a real, separate failure mode: P1's 0-byte
     kernel log reproduced across three independent pulls -- that one is genuine, not a race, and
     this gate must distinguish them rather than blame the network).

EXIT CODES:  0 complete   1 INCOMPLETE (re-pull, do not score)   2 usage.

Usage:
    python duck_eval/pull_complete.py runs/kernel_pulls/execwm_v1/artifacts
    python duck_eval/pull_complete.py <dir> --json runs/pull_complete_<arm>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Files that are legitimately empty in a HEALTHY pull -- package markers, not evidence.
BENIGN_EMPTY = {"__init__.py", ".gitkeep", "py.typed"}

MODEL_HINTS = ("model", "served_model", "MODEL_NAME", "--model")


def find_kernel_logs(run: Path):
    """Kernel logs, searched WIDER than the top-level `*.log` glob that the partial pull defeated."""
    out = []
    for pat in ("*.log", "*.txt", "**/*.log"):
        for p in run.glob(pat):
            if p.is_file() and p not in out:
                out.append(p)
    return sorted(out)


def model_sources(run: Path):
    """Independent places the served model survives a partial pull."""
    found = []
    taaf = run / "taaf_setup_env.json"
    if taaf.is_file() and taaf.stat().st_size > 0:
        try:
            blob = json.loads(taaf.read_text(encoding="utf-8", errors="replace"))
            flat = json.dumps(blob)
            if any(h in flat for h in MODEL_HINTS):
                found.append(("taaf_setup_env.json", taaf.stat().st_size))
        except Exception:
            pass
    for p in sorted((run / "prompts").glob("*.log")) if (run / "prompts").is_dir() else []:
        if p.stat().st_size > 0:
            found.append((os.path.join("prompts", p.name), p.stat().st_size))
            break
    vllm = [p for p in run.glob("*vllm*") if p.is_file() and p.stat().st_size > 0]
    if vllm:
        found.append((vllm[0].name, vllm[0].stat().st_size))
    return found


def check(run: Path):
    problems, facts = [], {}
    if not run.is_dir():
        return False, ["pull directory does not exist: %s" % run], facts

    logs = find_kernel_logs(run)
    nonempty = [p for p in logs if p.stat().st_size > 0]
    facts["log_files"] = len(logs)
    facts["log_files_nonempty"] = len(nonempty)
    facts["largest_log"] = (max((p.stat().st_size for p in logs), default=0))
    if not logs:
        problems.append("NO kernel log found at all (searched *.log, *.txt, **/*.log) "
                        "-- this is the partial-pull signature; RE-PULL before scoring")
    elif not nonempty:
        problems.append("kernel log(s) present but ALL are 0 bytes (%s) -- note this reproduced "
                        "across 3 independent pulls for P1 and is a REAL artifact defect, not a "
                        "download race; do not simply re-pull in a loop"
                        % ", ".join(p.name for p in logs))

    srcs = model_sources(run)
    facts["model_sources"] = [s[0] for s in srcs]
    if not srcs:
        problems.append("served model not recoverable from ANY independent source "
                        "(taaf_setup_env.json, prompts/*.log, *vllm*) -- a scorer running now "
                        "would report 'served model absent' about ITSELF, not about the run")

    empties = [p for p in run.rglob("*")
               if p.is_file() and p.stat().st_size == 0 and p.name not in BENIGN_EMPTY]
    facts["unexpected_empty_files"] = [str(p.relative_to(run)) for p in empties]
    if empties:
        problems.append("%d unexpected 0-byte file(s): %s"
                        % (len(empties), ", ".join(facts["unexpected_empty_files"][:5])))

    return (not problems), problems, facts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pull")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    run = Path(a.pull)
    ok, problems, facts = check(run)

    print("PULL-COMPLETENESS: %s" % ("COMPLETE" if ok else "INCOMPLETE"))
    print("  dir             : %s" % run)
    print("  log files       : %d (%d non-empty, largest %d bytes)"
          % (facts.get("log_files", 0), facts.get("log_files_nonempty", 0),
             facts.get("largest_log", 0)))
    print("  model sources   : %s" % (", ".join(facts.get("model_sources") or []) or "NONE"))
    for p in problems:
        print("  ! %s" % p)
    if ok:
        print("  -> safe to score")
    else:
        print("  -> DO NOT SCORE. An INFRA-DEATH verdict read off this pull would describe the "
              "download, not the run.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"pull": str(run), "complete": ok, "problems": problems, "facts": facts},
            indent=1), encoding="utf-8")
        print("  wrote %s" % a.json)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
