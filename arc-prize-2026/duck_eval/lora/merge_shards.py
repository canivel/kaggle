"""Merge sharded gen_dataset output into one JSONL + one datasheet.

Reads the shard JSONLs for the examples and the shard LOGS for the per-game
outcome table, so a shard that is still running (or was cut short) still
contributes everything it has produced. Shard datasheets, when present, add the
raw-vs-pruned action counts that only the search knows.

    ../../.venv/Scripts/python.exe merge_shards.py --dir ../../runs/lora_lane/v0 \
        --logs ../../runs/lora_lane --split train
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

RE_OK = re.compile(
    r"^\s*(?P<game>\S+): OK (?P<turns>\d+) turns, (?P<actions>\d+) actions vs human "
    r"(?P<human>\d+) \(ratio (?P<ratio>[\d.]+)\), (?P<levels>\d+) levels"
)
RE_NOPLAN = re.compile(r"^\s*(?P<game>\S+): no plan")
RE_LOWRATIO = re.compile(r"^\s*(?P<game>\S+): ratio (?P<ratio>[\d.]+) < [\d.]+ -> dropped")
RE_OTHER = re.compile(r"^\s*(?P<game>\S+): (?P<why>prune unverified|harness replay cleared|render error|search error|prune error)")


def parse_logs(log_dir: Path, split: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for log in sorted(log_dir.glob(f"gen_{split}.*.log")):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = RE_OK.match(line)
            if m:
                rows[m["game"]] = {
                    "game_id": m["game"], "status": "ok",
                    "turns": int(m["turns"]), "pruned_actions": int(m["actions"]),
                    "human_actions": int(m["human"]), "ratio": float(m["ratio"]),
                    "levels": int(m["levels"]),
                }
                continue
            m = RE_NOPLAN.match(line)
            if m:
                rows.setdefault(m["game"], {"game_id": m["game"], "status": "no_plan"})
                continue
            m = RE_LOWRATIO.match(line)
            if m:
                rows.setdefault(m["game"], {"game_id": m["game"],
                                            "status": "below_efficiency_floor",
                                            "ratio": float(m["ratio"])})
                continue
            m = RE_OTHER.match(line)
            if m:
                rows.setdefault(m["game"], {"game_id": m["game"], "status": m["why"]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    root = Path(args.dir)

    shards = sorted(root.glob(f"{args.split}.[0-9]*.jsonl"))
    rows: list[dict] = []
    for shard in shards:
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a shard cut mid-write leaves one partial line
    out = root / f"{args.split}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_game = parse_logs(Path(args.logs), args.split)
    # Shard datasheets add raw (pre-prune) action counts where available.
    raw_by_game: dict[str, int] = {}
    for sheet_path in sorted(root.glob(f"datasheet_{args.split}.[0-9]*.json")):
        sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
        for game in sheet["per_game"]:
            if "raw_actions" in game:
                raw_by_game[game["game_id"]] = game["raw_actions"]
    for game_id, raw in raw_by_game.items():
        if game_id in per_game:
            per_game[game_id]["raw_actions"] = raw

    kept_games = {r["meta"]["game_id"] for r in rows}
    ok = [g for g in per_game.values() if g.get("status") == "ok" and g["game_id"] in kept_games]
    ratios = [g["ratio"] for g in ok]
    with_raw = [g for g in ok if "raw_actions" in g]

    derived = sum(r["meta"].get("derived_clicks", 0) for r in rows)
    literal = sum(r["meta"].get("literal_clicks", 0) for r in rows)

    def prompt_chars(row: dict) -> int:
        total = 0
        for message in row["messages"]:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for part in content:
                    total += len(part["text"]) if part.get("type") == "text" else len(json.dumps(part))
            for call in message.get("tool_calls") or []:
                total += len(json.dumps(call))
        return total

    chars = [prompt_chars(r) for r in rows] or [0]
    turns_per_game = Counter(r["meta"]["game_id"] for r in rows)
    batch_sizes = [r["meta"]["batch_size"] for r in rows] or [0]

    merged = {
        "split": args.split,
        "shards": len(shards),
        "counts": {
            "envs_attempted": len(per_game),
            "envs_with_usable_plan": len(ok),
            "examples": len(rows),
            "distinct_games": len(turns_per_game),
            "distinct_images": len(list((root / "images").glob("*.png"))),
        },
        "status_breakdown": dict(Counter(g["status"] for g in per_game.values())),
        "efficiency": {
            "ratio_human_over_agent_mean": round(statistics.fmean(ratios), 3) if ratios else None,
            "ratio_median": round(statistics.median(ratios), 3) if ratios else None,
            "ratio_min": round(min(ratios), 3) if ratios else None,
            "ratio_max": round(max(ratios), 3) if ratios else None,
            "plans_at_or_better_than_human": sum(1 for r in ratios if r >= 1.0),
            "plans_within_1_5x_of_human": sum(1 for r in ratios if r >= 1 / 1.5),
            "total_plan_actions": sum(g["pruned_actions"] for g in ok),
            "total_human_actions": sum(g["human_actions"] for g in ok),
            "levels_cleared_total": sum(g["levels"] for g in ok),
            "prune_subsample_n": len(with_raw),
            "prune_raw_actions": sum(g["raw_actions"] for g in with_raw),
            "prune_kept_actions": sum(g["pruned_actions"] for g in with_raw),
            "prune_reduction": (
                round(1 - sum(g["pruned_actions"] for g in with_raw)
                      / max(1, sum(g["raw_actions"] for g in with_raw)), 3)
                if with_raw else None
            ),
        },
        "prompt_size": {
            "chars_mean": round(statistics.fmean(chars)),
            "chars_median": round(statistics.median(chars)),
            "chars_max": max(chars),
            "approx_tokens_mean": round(statistics.fmean(chars) / 3.5),
            "approx_tokens_median": round(statistics.median(chars) / 3.5),
            "approx_tokens_max": round(max(chars) / 3.5),
            "approx_total_tokens_unpacked": round(sum(chars) / 3.5),
        },
        "turn_structure": {
            "turns_per_game_mean": round(statistics.fmean(turns_per_game.values()), 2) if turns_per_game else 0,
            "turns_per_game_max": max(turns_per_game.values()) if turns_per_game else 0,
            "actions_per_batch_mean": round(statistics.fmean(batch_sizes), 2),
            "actions_per_batch_max": max(batch_sizes),
            "single_action_batches_pct": round(
                100 * sum(1 for b in batch_sizes if b == 1) / max(1, len(batch_sizes)), 1
            ),
        },
        "click_rendering": {
            "derived_clicks": derived,
            "literal_clicks": literal,
            "derived_click_fraction": round(derived / max(1, derived + literal), 3),
        },
        "per_game": sorted(per_game.values(), key=lambda g: g["game_id"]),
    }
    (root / f"datasheet_{args.split}.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in merged.items() if k != "per_game"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
