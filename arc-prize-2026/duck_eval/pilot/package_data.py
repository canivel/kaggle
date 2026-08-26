"""Bundle the Phase-0c pilot (code + observations + template sims) into
duck_eval/pilot_bundle.tar.gz for a single scp/runpodctl transfer to the pod.

Bundle layout (matches run_pilot.py's data-root resolution):
  pilot_bundle/
    pilot/run_pilot.py, scoring.py, prompts.py
    exec_wm/observations/<game>.json (+ .summary.json)
    exec_wm/sims/<game>_sim.py        (LOGO templates)

Local-only script: uses pathlib, Windows-safe. Run with:
  uv run python duck_eval/pilot/package_data.py
"""
from __future__ import annotations

import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # duck_eval/pilot
REPO = HERE.parents[1]                          # arc-prize-2026
OUT = HERE.parent / "pilot_bundle.tar.gz"       # duck_eval/pilot_bundle.tar.gz

PILOT_FILES = ["run_pilot.py", "scoring.py", "prompts.py", "POD_RUNBOOK.md"]


def main() -> None:
    obs_dir = REPO / "exec_wm" / "observations"
    sims_dir = REPO / "exec_wm" / "sims"
    assert obs_dir.exists(), obs_dir
    assert sims_dir.exists(), sims_dir

    n = 0
    with tarfile.open(OUT, "w:gz") as tar:
        for name in PILOT_FILES:
            p = HERE / name
            if p.exists():
                tar.add(p, arcname=f"pilot_bundle/pilot/{name}")
                n += 1
        # ALL games' observations + sims ship (LOGO templates need the full
        # pool; observations carry available_actions/change-rate for nearest-
        # template selection). ~25 games x ~2-6 MB JSON, gz-compressed.
        for p in sorted(obs_dir.glob("*.json")):
            tar.add(p, arcname=f"pilot_bundle/exec_wm/observations/{p.name}")
            n += 1
        for p in sorted(sims_dir.glob("*_sim.py")):
            if p.name.startswith("_"):
                continue
            tar.add(p, arcname=f"pilot_bundle/exec_wm/sims/{p.name}")
            n += 1
        # cd82's sim loads a data module next to it
        extra = sims_dir / "cd82_rotation_table_data.py"
        if extra.exists():
            tar.add(extra, arcname=f"pilot_bundle/exec_wm/sims/{extra.name}")
            n += 1

    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {n} files)")


if __name__ == "__main__":
    main()
