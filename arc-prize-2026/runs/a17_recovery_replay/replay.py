"""A17 canary v3 offline recovery replay (2026-07-26).

Question: had the harness recovered markdown-fenced ```python blocks as python
tool calls (extending the existing tool_calls_recovered_from_markup path), what
fraction of the 72B-VL's 1200 zero-tool-call turns would have produced an
executable step?

Inputs: runs/kernel_pulls/a17_canary_v3/transcripts/*_p0.txt (recorded on-node
model responses; deterministic replay, no model needed).

Recovery rule evaluated (candidate for canary v4):
  1. take the [ASSISTANT] section text of each turn
  2. extract all ```python fenced blocks (also bare ``` blocks that parse as py)
  3. concatenate in order -> one python tool call body
  4. RECOVERED if the body ast-parses
  5. ACTIONABLE if it additionally references action( / current_frame / history
     (i.e. would actually drive the game, not just print)

Output: recovery_report.json + stdout table.
"""
from __future__ import annotations
import ast
import json
import re
from pathlib import Path

PULL = Path(__file__).resolve().parents[1] / "kernel_pulls" / "a17_canary_v3" / "transcripts"
OUT = Path(__file__).resolve().parent / "recovery_report.json"

TURN_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| ", re.M)
FENCE_RE = re.compile(r"```(python)?[ \t]*\n(.*?)```", re.S)


def turns(text: str):
    starts = [m.start() for m in TURN_RE.finditer(text)]
    for i, s in enumerate(starts):
        yield text[s : starts[i + 1] if i + 1 < len(starts) else len(text)]


def assistant_section(turn: str) -> str:
    m = re.search(r"^\[ASSISTANT\]\n(.*?)(?=^\[[A-Z ]+\]$|\Z)", turn, re.S | re.M)
    return m.group(1) if m else ""


def analyze(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    n = fenced = recovered = actionable = multi = 0
    parse_fail_samples = []
    for turn in turns(text):
        n += 1
        body_blocks = []
        for lang, code in FENCE_RE.findall(assistant_section(turn)):
            if lang == "python":
                body_blocks.append(code)
            else:
                try:
                    ast.parse(code)
                    body_blocks.append(code)
                except SyntaxError:
                    pass
        if not body_blocks:
            continue
        fenced += 1
        if len(body_blocks) > 1:
            multi += 1
        body = "\n".join(body_blocks)
        try:
            ast.parse(body)
        except SyntaxError as e:
            if len(parse_fail_samples) < 3:
                parse_fail_samples.append(f"step~{n}: {e}")
            continue
        recovered += 1
        if re.search(r"\baction\s*\(|current_frame|history\b", body):
            actionable += 1
    return {
        "game": path.stem,
        "turns": n,
        "turns_with_fenced_python": fenced,
        "recovered_ast_ok": recovered,
        "actionable": actionable,
        "multi_block_turns": multi,
        "parse_fail_samples": parse_fail_samples,
    }


def main() -> None:
    results = [analyze(p) for p in sorted(PULL.glob("*_p0.txt"))]
    tot = {k: sum(r[k] for r in results) for k in
           ("turns", "turns_with_fenced_python", "recovered_ast_ok", "actionable", "multi_block_turns")}
    report = {"per_game": results, "total": tot,
              "recovery_rate": tot["recovered_ast_ok"] / tot["turns"],
              "actionable_rate": tot["actionable"] / tot["turns"]}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'game':28s} {'turns':>6s} {'fenced':>7s} {'ast_ok':>7s} {'action':>7s} {'multi':>6s}")
    for r in results:
        print(f"{r['game']:28s} {r['turns']:6d} {r['turns_with_fenced_python']:7d} "
              f"{r['recovered_ast_ok']:7d} {r['actionable']:7d} {r['multi_block_turns']:6d}")
    print(f"{'TOTAL':28s} {tot['turns']:6d} {tot['turns_with_fenced_python']:7d} "
          f"{tot['recovered_ast_ok']:7d} {tot['actionable']:7d} {tot['multi_block_turns']:6d}")
    print(f"recovery_rate={report['recovery_rate']:.3f} actionable_rate={report['actionable_rate']:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
