"""Stage 1 of the LoRA lane: build SFT examples from verified engine plans.

CPU only. No LLM call, no network, no Kaggle, no spend.

    cd duck_eval/lora
    ../../.venv/Scripts/python.exe gen_dataset.py --limit 40 --out ../../runs/lora_lane/v0

Pipeline per environment (training pool only -- see splits.py):

    greedy_search  ->  prune (leave-one-out, replay-verified)
                   ->  efficiency filter vs the human baseline
                   ->  split into batched turns
                   ->  replay through a REAL _HarnessGameSession with a
                       TeacherAgent, capturing (messages, target) per turn
                   ->  JSONL + datasheet

Every prompt byte is produced by the harness's own code; see teacher.py.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_env import INTERNAL_ENVS, OUT_DIR, PUBLIC_ENVS, arcade_spec, bootstrap  # noqa: E402

bootstrap()

import oracle  # noqa: E402
import splits  # noqa: E402
import teacher  # noqa: E402


# --------------------------------------------------------------------------
def _level_boundaries(game_id: str, spec, actions: list[dict[str, Any]]) -> set[int]:
    """Indices in `actions` after which levels_completed increases."""
    game = oracle.open_game(game_id, spec, allow_deepcopy=False)
    state = game.current_state
    completed = int(state.levels_completed)
    out: set[int] = set()
    for index, action in enumerate(actions):
        if oracle._terminal(state):
            break
        state = game.execute_action(oracle._to_input(action))
        now = int(state.levels_completed)
        if now > completed:
            out.add(index)
            completed = now
    return out


def _make_session(game, tmpdir: Path, analyzer):
    import inference.framework.solver as solver_mod

    class _Solver:
        max_actions_per_game = None
        max_runtime_s_per_game = 1800.0
        job_dir = tmpdir
        label = "lora-teacher"

        def soft_time_remaining_seconds(self):
            return None

    return solver_mod._HarnessGameSession(
        solver=_Solver(),
        game=game,
        analyzer=analyzer,
        game_index=0,
        pass_index=0,
        state_path=tmpdir / "runtime_state.json",
        transcript_path=tmpdir / "transcript.txt",
        analysis_html_relpath="solver_analysis/lora.html",
        stop_event=threading.Event(),
        viewer_data_path=tmpdir / "viewer_data.json",
    )


def augment_with_noops(game_id: str, spec, plan: oracle.Plan, raw_actions, *, per_level: int):
    """Re-insert up to `per_level` genuine no-ops from the raw search trace.

    Rationale in the war-room note §7.2/§7.4: pruning deletes every
    `board_changed: False` event, so the corpus has zero positive examples of
    "that did nothing, and I did not retry it" -- the exact behaviour P2 is
    supposed to teach. A no-op is state-neutral, so re-inserting it at the same
    level cannot change what the plan achieves, and the harness-replay check
    downstream still proves the level clears.

    Inserted at the FRONT of the level's actions, so they land in the first
    (probe) turn and the next turn's note can name them as settled.
    """
    if per_level <= 0 or not raw_actions:
        return plan, 0
    # Prefer the lookahead's PROVEN dead candidates; fall back to no-ops that
    # actually got committed in the raw trace (rare -- greedy avoids them).
    noops = [(0, lvl, act) for lvl, act in plan.noop_candidates]
    if not noops:
        noops = oracle.find_noops(game_id, spec, raw_actions)
    if not noops:
        return plan, 0
    by_level: dict[int, list] = {}
    for _, level, action in noops:
        bucket = by_level.setdefault(level, [])
        if len(bucket) < per_level and action not in bucket:
            # `_probe` marks it for a turn of its own (teacher.split_into_turns).
            # `_to_input` reads only id/x/y, so the extra key is inert everywhere
            # else. It has to be its own turn because `step_env` aggregates
            # `board_changed` with any() across a batch -- mixed in with real
            # actions the no-op reports True and teaches nothing (§7.2).
            bucket.append({**action, "_probe": True})
    if not by_level:
        return plan, 0

    # Walk the pruned plan, tracking level, and inject each level's no-ops the
    # first time we are inside that level.
    game = oracle.open_game(game_id, spec, allow_deepcopy=False)
    state = game.current_state
    merged: list = []
    injected = 0
    done: set[int] = set()
    for action in plan.actions:
        level = int(state.levels_completed)
        if level not in done:
            done.add(level)
            for extra in by_level.get(level, []):
                merged.append(extra)
                injected += 1
        merged.append(action)
        try:
            state = game.execute_action(oracle._to_input(action))
        except Exception:
            break
    if injected == 0:
        return plan, 0
    augmented = oracle.Plan(
        game_id=plan.game_id,
        actions=merged,
        levels_cleared=plan.levels_cleared,
        baseline_actions=plan.baseline_actions,
        raw_action_count=plan.raw_action_count,
        verified=False,
    )
    levels, _ = oracle.replay(game_id, spec, merged)
    augmented.verified = levels >= plan.levels_cleared
    if not augmented.verified:
        return plan, 0          # never trade a level for a teaching example
    return augmented, injected


def render_examples(game_id: str, spec, plan: oracle.Plan, *, first_batch: int, batch_size: int):
    boundaries = _level_boundaries(game_id, spec, plan.actions)
    turns = teacher.split_into_turns(
        plan.actions, boundaries, first_batch=first_batch, batch_size=batch_size
    )
    game = oracle.open_game(game_id, spec, allow_deepcopy=False)
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        stop_event = threading.Event()

        def grid_provider():
            return oracle._grid(game.current_state)

        def level_provider():
            return int(game.current_state.levels_completed) + 1

        agent = teacher.TeacherAgent(
            turns,
            game_id=game_id,
            baseline_actions=plan.baseline_actions,
            stop_event=stop_event,
            grid_provider=grid_provider,
            level_provider=level_provider,
        )
        session = _make_session(game, tmpdir, agent)
        session.stop_event = stop_event
        agent._stop_event = stop_event
        session.play()
        cleared = int(game.current_state.levels_completed)
    return agent, cleared


# --------------------------------------------------------------------------
def _externalize_images(messages: list[dict[str, Any]], image_dir: Path) -> int:
    """Replace inline data: URLs with `arcimg://<sha1>.png` refs, writing the
    PNGs out once each. Keeps the JSONL readable and small; fully reversible."""
    count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = part.get("image_url", {}).get("url", "")
            if not url.startswith("data:image/png;base64,"):
                continue
            raw = base64.b64decode(url.split(",", 1)[1])
            digest = hashlib.sha1(raw).hexdigest()[:16]
            path = image_dir / f"{digest}.png"
            if not path.exists():
                image_dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            part["image_url"]["url"] = f"arcimg://{digest}.png"
            count += 1
    return count


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "v0"))
    ap.add_argument("--limit", type=int, default=40, help="max training envs to attempt")
    ap.add_argument("--target-examples", type=int, default=500)
    ap.add_argument("--max-actions", type=int, default=140, help="greedy search budget")
    ap.add_argument("--target-levels", type=int, default=2)
    ap.add_argument("--first-batch", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="max actions per action() call. NOT 8: P2 measured that the "
                         "level-completing action is the LAST of a long stall-gated "
                         "batch (ar25/tu93 need N>=15), so a cap of 8 structurally "
                         "forbids the behaviour that completes levels")
    ap.add_argument("--min-ratio", type=float, default=0.60,
                    help="keep a plan only if human_actions/plan_actions >= this")
    ap.add_argument("--split", default="train", choices=["train", "dev", "eval"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="greedy restarts per env")
    ap.add_argument("--noops-per-level", type=int, default=1,
                    help="re-insert up to N genuine no-ops per level (war-room note 7.4 item 1)")
    ap.add_argument("--shard", default="0/1", help="i/n -- process every n-th env starting at i")
    ap.add_argument("--games", default=None, help="comma list; overrides the split ordering")
    ap.add_argument("--inline-images", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"

    split = splits.build_split()
    envs_dir = PUBLIC_ENVS if args.split == "eval" else INTERNAL_ENVS
    spec = arcade_spec(envs_dir)
    game_ids = (args.games.split(",") if args.games else split[args.split])[: args.limit]
    _i, _n = (int(x) for x in args.shard.split("/"))
    game_ids = [g for k, g in enumerate(game_ids) if k % _n == _i]

    (out_dir / "split.json").write_text(
        json.dumps({k: v for k, v in split.items()}, indent=2), encoding="utf-8"
    )

    suffix = "" if args.shard == "0/1" else f".{_i}"
    jsonl = (out_dir / f"{args.split}{suffix}.jsonl").open("w", encoding="utf-8")
    per_game: list[dict[str, Any]] = []
    n_examples = 0
    n_images = 0
    render_stats = {"derived_clicks": 0, "literal_clicks": 0}
    started = time.time()

    for game_id in game_ids:
        if n_examples >= args.target_examples:
            break
        row: dict[str, Any] = {"game_id": game_id}
        t0 = time.time()
        try:
            plan = None
            for attempt in range(args.seeds):
                candidate = oracle.greedy_search(
                    game_id,
                    spec,
                    max_actions=args.max_actions,
                    target_levels=args.target_levels,
                    seed=args.seed + attempt,
                )
                if candidate is None:
                    continue
                # Prefer more levels, then fewer raw actions.
                if plan is None or (
                    candidate.levels_cleared,
                    -candidate.raw_action_count,
                ) > (plan.levels_cleared, -plan.raw_action_count):
                    plan = candidate
                if plan.levels_cleared >= args.target_levels:
                    break
        except Exception as exc:
            row.update(status=f"search_error:{type(exc).__name__}")
            per_game.append(row)
            print(f"  {game_id}: search error {exc}")
            continue
        if plan is None:
            row.update(status="no_plan", search_s=round(time.time() - t0, 1))
            per_game.append(row)
            print(f"  {game_id}: no plan ({time.time() - t0:.1f}s)")
            continue

        try:
            pruned = oracle.prune(plan, spec)
        except Exception as exc:
            row["status"] = f"prune_error:{type(exc).__name__}"
            per_game.append(row)
            print(f"  {game_id}: prune error {exc}")
            continue
        row.update(
            raw_actions=plan.raw_action_count,
            pruned_actions=pruned.action_count,
            levels=pruned.levels_cleared,
            human_actions=pruned.human_actions,
            ratio=round(pruned.rhae, 3),
            verified=pruned.verified,
            search_s=round(time.time() - t0, 1),
        )
        if not pruned.verified:
            row["status"] = "unverified"
            per_game.append(row)
            print(f"  {game_id}: prune unverified -> dropped")
            continue
        if pruned.rhae < args.min_ratio:
            row["status"] = "below_efficiency_floor"
            per_game.append(row)
            print(f"  {game_id}: ratio {pruned.rhae:.2f} < {args.min_ratio} -> dropped")
            continue

        injected = 0
        if args.noops_per_level > 0:
            pruned, injected = augment_with_noops(
                game_id, spec, pruned, plan.actions, per_level=args.noops_per_level
            )
            row["noops_injected"] = injected
            row["pruned_actions"] = pruned.action_count
            row["ratio"] = round(pruned.rhae, 3)
        try:
            agent, cleared = render_examples(
                game_id, spec, pruned, first_batch=args.first_batch, batch_size=args.batch_size
            )
        except Exception:
            row["status"] = "render_error"
            row["traceback"] = traceback.format_exc(limit=3)
            per_game.append(row)
            print(f"  {game_id}: render error\n{traceback.format_exc(limit=3)}")
            continue

        if cleared < pruned.levels_cleared:
            row["status"] = f"harness_replay_short({cleared}<{pruned.levels_cleared})"
            per_game.append(row)
            print(f"  {game_id}: harness replay cleared {cleared} < {pruned.levels_cleared} -> dropped")
            continue

        for example in agent.examples:
            if not args.inline_images:
                n_images += _externalize_images(example.messages, image_dir)
            jsonl.write(
                json.dumps(
                    {
                        "messages": example.messages,
                        "target": example.target,
                        "meta": {**example.meta, "split": args.split},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_examples += 1
        for key in render_stats:
            render_stats[key] += agent.render_stats[key]
        row.update(status="ok", turns=len(agent.examples),
                   derived_clicks=agent.render_stats["derived_clicks"],
                   literal_clicks=agent.render_stats["literal_clicks"])
        per_game.append(row)
        print(
            f"  {game_id}: OK {len(agent.examples)} turns, {pruned.action_count} actions "
            f"vs human {pruned.human_actions} (ratio {pruned.rhae:.2f}), {cleared} levels"
        )

    jsonl.close()

    ok_rows = [r for r in per_game if r.get("status") == "ok"]
    ratios = [r["ratio"] for r in ok_rows]
    datasheet = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": args.split,
        "environments_dir": str(envs_dir),
        "config": vars(args),
        "counts": {
            "envs_attempted": len(per_game),
            "envs_with_usable_plan": len(ok_rows),
            "examples": n_examples,
            "distinct_images": len(list(image_dir.glob("*.png"))) if image_dir.exists() else 0,
            "image_refs": n_images,
        },
        "split_sizes": {k: len(v) for k, v in split.items()},
        "efficiency": {
            "ratio_human_over_agent_mean": round(statistics.fmean(ratios), 3) if ratios else None,
            "ratio_median": round(statistics.median(ratios), 3) if ratios else None,
            "ratio_min": round(min(ratios), 3) if ratios else None,
            "ratio_max": round(max(ratios), 3) if ratios else None,
            "at_or_better_than_human": sum(1 for r in ratios if r >= 1.0),
            "total_plan_actions": sum(r["pruned_actions"] for r in ok_rows),
            "total_human_actions": sum(r["human_actions"] for r in ok_rows),
            "total_raw_search_actions": sum(r["raw_actions"] for r in ok_rows),
        },
        "click_rendering": {
            **render_stats,
            "derived_click_fraction": (
                round(
                    render_stats["derived_clicks"]
                    / max(1, render_stats["derived_clicks"] + render_stats["literal_clicks"]),
                    3,
                )
            ),
        },
        "wall_clock_s": round(time.time() - started, 1),
        "per_game": per_game,
    }
    (out_dir / f"datasheet_{args.split}{suffix}.json").write_text(
        json.dumps(datasheet, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in datasheet.items() if k != "per_game"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
