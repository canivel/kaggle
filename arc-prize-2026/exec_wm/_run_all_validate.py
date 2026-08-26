"""One-shot: run validate_sim.py for every game in GAMES and collect top-level scores."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GAMES = [
    "ar25", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25",
    "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
]


def main():
    rows = []
    for g in GAMES:
        sim = ROOT / "exec_wm" / "sims" / f"{g}_sim.py"
        if not sim.exists():
            rows.append({"game": g, "status": "MISSING_SIM", "n": 0})
            continue
        try:
            proc = subprocess.run(
                ["uv", "run", "python", "exec_wm/validate_sim.py", "--game", g],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            rows.append({"game": g, "status": "TIMEOUT"})
            continue
        out = proc.stdout.strip()
        # find first '{' to skip uv warnings
        i = out.find("{")
        if i < 0:
            rows.append({"game": g, "status": "NO_JSON", "stderr": proc.stderr[:200]})
            continue
        try:
            data = json.loads(out[i:])
        except Exception as e:
            rows.append({"game": g, "status": f"PARSE_ERR:{e}"})
            continue
        n_actions = len(data.get("by_action", {}))
        rows.append({
            "game": g,
            "status": "OK",
            "n": data["n"],
            "errors": data["errors"],
            "state_exact_pct": data["state_exact_pct"],
            "pixel_match_pct": data["pixel_match_pct"],
            "reward_acc_pct": data["reward_acc_pct"],
            "done_acc_pct": data["done_acc_pct"],
            "n_actions": n_actions,
        })
        print(f"{g}: state_exact={data['state_exact_pct']:.2f} pixel={data['pixel_match_pct']:.4f} "
              f"reward={data['reward_acc_pct']:.2f} done={data['done_acc_pct']:.2f} "
              f"err={data['errors']} n_act={n_actions}", flush=True)
    (ROOT / "exec_wm" / "_scale_results.json").write_text(json.dumps(rows, indent=2))
    print("WROTE", ROOT / "exec_wm" / "_scale_results.json")


if __name__ == "__main__":
    main()
