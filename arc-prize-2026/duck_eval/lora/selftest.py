"""Gates for the stage-1 LoRA dataset. CPU only, no LLM, no network.

    cd duck_eval/lora
    ../../.venv/Scripts/python.exe selftest.py --data ../../runs/lora_lane/v0

G1  LEAKAGE     no example in train.jsonl comes from a family that is scored
                on the public leaderboard, and train/dev families are disjoint.
G2  FIDELITY    the system prompt we render is byte-identical to the SYSTEM
                PROMPT the real duck harness logged in a real run
                (`runs/gpt56_probe/experiment_full/transcripts/*.txt`).
G3  SHAPE       every example ends on a `user` turn, targets carry exactly one
                `python` tool call with valid JSON arguments and compiling code.
G4  COMPAT      the engine-loader patch does not change how the public 25 load.
G5  EFFICIENCY  the plans we distil are at or near the human action baseline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_env import PUBLIC_ENVS, REPO, bootstrap, family, list_environments  # noqa: E402

bootstrap()

PASS = FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def load(data_dir: Path, split: str) -> list[dict]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def g1_leakage(data_dir: Path, rows: list[dict]) -> None:
    print("G1: leakage")
    public = {family(g) for g in list_environments(PUBLIC_ENVS)}
    fams = {family(r["meta"]["game_id"]) for r in rows}
    overlap = sorted(fams & public)
    check("no training example from a scored family", not overlap, f"overlap={overlap}")
    split = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    train_f = {family(g) for g in split["train"]}
    dev_f = {family(g) for g in split["dev"]}
    eval_f = {family(g) for g in split["eval"]}
    check("train/dev families disjoint", not (train_f & dev_f), str(sorted(train_f & dev_f))[:120])
    check("train/eval families disjoint", not (train_f & eval_f), str(sorted(train_f & eval_f))[:120])
    check("dev/eval families disjoint", not (dev_f & eval_f), str(sorted(dev_f & eval_f))[:120])


def _transcript_system_prompt() -> str | None:
    tdir = REPO / "runs" / "gpt56_probe" / "experiment_full" / "transcripts"
    files = sorted(tdir.glob("*_p0.txt"))
    if not files:
        return None
    text = files[0].read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\[SYSTEM PROMPT\]\n(.*?)\n\[USER PROMPT\]", text, re.S)
    return match.group(1).strip() if match else None


def g2_fidelity(rows: list[dict]) -> None:
    print("G2: prompt fidelity vs a real duck run")
    if not rows:
        check("examples present", False, "no rows")
        return
    ours = rows[0]["messages"][0]
    check("first message is the system prompt", ours.get("role") == "system")
    logged = _transcript_system_prompt()
    if logged is None:
        check("real-run transcript available", False, "no gpt56 transcript on disk")
        return
    mine = str(ours.get("content", "")).strip()
    check(
        "system prompt byte-identical to the logged real run",
        mine == logged,
        f"ours={len(mine)}B logged={len(logged)}B first-diff="
        f"{next((i for i, (a, b) in enumerate(zip(mine, logged)) if a != b), min(len(mine), len(logged)))}",
    )
    # The user turn must carry the image part the duck sends at MULTIMODAL_CONTEXT=current_grid.
    last_user = next((m for m in reversed(rows[0]["messages"]) if m.get("role") == "user"), None)
    has_image = isinstance(last_user, dict) and isinstance(last_user.get("content"), list) and any(
        p.get("type") == "image_url" for p in last_user["content"] if isinstance(p, dict)
    )
    check("final user turn carries the grid image part", has_image)


def g3_shape(rows: list[dict]) -> None:
    print("G3: example shape")
    bad_tail = [i for i, r in enumerate(rows) if r["messages"][-1].get("role") != "user"]
    check("every example ends on a user turn", not bad_tail, f"{len(bad_tail)} bad")
    bad_tool = []
    bad_code = []
    for i, r in enumerate(rows):
        calls = r["target"].get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"]["name"] != "python":
            bad_tool.append(i)
            continue
        try:
            code = json.loads(calls[0]["function"]["arguments"])["code"]
            compile(code, "<t>", "exec")
            if "action(" not in code:
                bad_code.append(i)
        except Exception:
            bad_code.append(i)
    check("target is exactly one `python` tool call", not bad_tool, f"{len(bad_tool)} bad")
    check("target code compiles and calls action()", not bad_code, f"{len(bad_code)} bad")
    roles = {m.get("role") for r in rows for m in r["messages"]}
    check("only system/user/assistant/tool roles", roles <= {"system", "user", "assistant", "tool"}, str(roles))


def g4_compat() -> None:
    """Differential, not absolute: the patch must not change WHICH public games
    load. (cn04-65d47d14 fails to load with or without it -- GameAPI asserts
    `base_actions_per_level has 6 entries; number_of_levels is 5`. That is a
    pre-existing metadata defect in the shipped public environment, not ours,
    and it means the local rig covers 24 of the 25 scored games.)"""
    print("G4: engine-loader patch is inert on the public 25")
    import arc_agi.local_wrapper as lw

    from harness_env import arcade_spec
    from taaf.game_api import GameAPI

    spec = arcade_spec(PUBLIC_ENVS)

    def loadable() -> set[str]:
        out: set[str] = set()
        for game_id in list_environments(PUBLIC_ENVS):
            try:
                game = GameAPI(env_name=game_id, arcade_spec=spec, allow_deepcopy=False)
                game.start_game()
                out.add(game_id)
            except Exception:
                pass
        return out

    original = lw.LocalEnvironmentWrapper._load_game_class
    before = loadable()

    import engine_compat

    engine_compat._APPLIED = False
    engine_compat.apply()
    check("patch installed", lw.LocalEnvironmentWrapper._load_game_class is not original)
    after = loadable()

    check(
        "patch changes nothing about which public games load",
        before == after,
        f"only-unpatched={sorted(before - after)} only-patched={sorted(after - before)}",
    )
    check(
        "local rig covers 24/25 public games (cn04 metadata defect is pre-existing)",
        len(after) == 24,
        f"{len(after)}/25 loadable: missing {sorted(set(list_environments(PUBLIC_ENVS)) - after)}",
    )


def g5_efficiency(data_dir: Path) -> None:
    print("G5: efficiency of the distilled plans")
    sheet_path = data_dir / "datasheet_train.json"
    if not sheet_path.exists():
        check("datasheet present", False)
        return
    sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
    eff = sheet["efficiency"]
    check("plans exist", (eff["ratio_median"] or 0) > 0, json.dumps(eff))
    check(
        "median plan is within 1.5x of the human baseline",
        (eff["ratio_median"] or 0) >= 1 / 1.5,
        f"median human/agent = {eff['ratio_median']}",
    )
    check(
        "pruning is doing real work (raw >> pruned)",
        (eff.get("prune_reduction") or 0) > 0,
        f"raw={eff.get('prune_raw_actions')} kept={eff.get('prune_kept_actions')}",
    )


def dump_examples(rows: list[dict], out: Path, n: int = 3) -> None:
    """Verbatim renders for human eyeballing: full message list, images elided
    to their ref so the text stays readable."""
    out.mkdir(parents=True, exist_ok=True)
    # Deliberately diverse: a cold-open turn, a mid-conversation turn that
    # carries real history, and a segmentation-derived multi-click commit.
    # Three copies of "turn 0" would hide exactly what needs eyeballing.
    def first(pred) -> dict | None:
        return next((r for r in rows if pred(r)), None)

    wanted = [
        first(lambda r: r["meta"]["turn_index"] == 0),
        first(lambda r: r["meta"]["turn_index"] >= 2 and r["meta"]["batch_size"] >= 4),
        first(lambda r: r["meta"].get("derived_clicks", 0) >= 2),
        first(lambda r: r["meta"]["turn_index"] >= 1),
    ]
    picked: list[dict] = []
    for row in wanted + rows:
        if row is None or any(row is p for p in picked):
            continue
        picked.append(row)
        if len(picked) >= n:
            break
    for index, row in enumerate(picked, start=1):
        lines: list[str] = []
        lines.append(f"### EXAMPLE {index} -- {row['meta']['game_id']} "
                     f"turn {row['meta']['turn_index']} level {row['meta']['level']}\n")
        for message in row["messages"]:
            role = message["role"].upper()
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        parts.append(f"<<IMAGE {part['image_url']['url']}>>")
                content = "\n".join(parts)
            lines.append(f"[{role}]")
            if content:
                lines.append(str(content))
            for call in message.get("tool_calls") or []:
                lines.append(f"[TOOL CALL {call['function']['name']} id={call['id']}]")
                lines.append(json.loads(call["function"]["arguments"])["code"])
            lines.append("")
        target = row["target"]
        lines.append("[TARGET ASSISTANT]")
        lines.append(str(target.get("content", "")))
        for call in target.get("tool_calls") or []:
            lines.append(f"[TARGET TOOL CALL {call['function']['name']} id={call['id']}]")
            lines.append(json.loads(call["function"]["arguments"])["code"])
        (out / f"example_{index}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {len(picked)} verbatim examples to {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "runs" / "lora_lane" / "v0"))
    ap.add_argument("--dump", type=int, default=3)
    args = ap.parse_args()
    data_dir = Path(args.data)
    rows = load(data_dir, "train")
    print(f"selftest | {len(rows)} training examples in {data_dir}")
    g1_leakage(data_dir, rows)
    g2_fidelity(rows)
    g3_shape(rows)
    g4_compat()
    g5_efficiency(data_dir)
    if args.dump and rows:
        dump_examples(rows, data_dir / "verbatim", n=args.dump)
    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
