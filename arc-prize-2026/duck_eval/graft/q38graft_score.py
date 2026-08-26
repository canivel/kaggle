"""Sealed scorer - ARM 3 Q38xGRAFT compound (q38_graft_prereg_2026-08-21.md).

Per-arm marker table (the q38-low landmine class, handled by construction):
[clickmap] armed is REQUIRED here; searchmap joins banking/transfer as FORBIDDEN.
Reuses graft_score's corrected log decoding (CLI-2.2.3 JSON) via import.
Bands (sealed, 1-sigma screens vs the n=1 field-floor comparator lc 28):
  INFRA DEATH  any certification failure / no benchmark / n != 25
  HARM         lc_total <= 23      NULL 24..32      SIGNAL >= 33
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import graft_score as gs

FLAGS_ON = ("efficiency", "retry_guard", "shortcircuit", "goalkeep", "hudmask", "clickmap")
FLAGS_FORBIDDEN = ("banking", "transfer", "searchmap")
ARMED_REQUIRED = ("[goalkeep] armed", "[hudmask] armed", "[clickmap] armed")
ARMED_FORBIDDEN = ("[banking] armed", "[transfer] armed", "[searchmap] armed")
SERVED_MODEL = "Qwen/Qwen3.8-27B-FP8"
HARM_MAX, SIGNAL_MIN, N_GAMES = 23, 33, 25


def certify(log: str):
    facts = {}
    for sig in gs.STOCK_FALLBACK_SIGNATURES:
        if sig in log:
            return False, f"stock fallback: {sig!r}", facts
    m = gs.BANNER_RE.search(log)
    if not m:
        return False, "no TAAF_GRAFTS FEATURES banner", facts
    raw, api = m.group(1), int(m.group(2))
    facts["banner"], facts["api_version"] = raw, api
    if api != 1:
        return False, f"API_VERSION={api} != 1", facts
    present = {f for f in (*FLAGS_ON, *FLAGS_FORBIDDEN) if f"'{f}'" in raw or f'"{f}"' in raw}
    facts["features_present"] = sorted(present)
    missing = [f for f in FLAGS_ON if f not in present]
    if missing:
        return False, f"flags missing from FEATURES: {missing}", facts
    wrong = [f for f in FLAGS_FORBIDDEN if f in present]
    if wrong:
        return False, f"FORBIDDEN flag in FEATURES (wrong arm): {wrong}", facts
    for line in ARMED_REQUIRED:
        if line not in log:
            return False, f"missing required armed line {line!r}", facts
    for line in ARMED_FORBIDDEN:
        if line in log:
            return False, f"forbidden armed line (wrong arm): {line!r}", facts
    if SERVED_MODEL not in log:
        return False, f"served model banner {SERVED_MODEL!r} not found", facts
    if "reasoning_effort" in log:
        return False, "reasoning_effort present in log (xhigh default violated)", facts
    return True, "certified", facts


def score(run_dir: Path) -> dict:
    log = gs._read_log(run_dir)
    ok, reason, facts = certify(log)
    if not ok:
        return gs._infra(f"install/serve not certified: {reason}", **facts)
    path = run_dir / "benchmark.json"
    if not path.exists():
        return gs._infra("no benchmark.json", **facts)
    runs = (json.loads(path.read_text(encoding="utf-8")).get("game_runs")) or []
    if len(runs) != N_GAMES:
        return gs._infra(f"n_games={len(runs)} != {N_GAMES}", **facts)
    lc = sum(int(r.get("levels_completed") or 0) for r in runs)
    ms = sum(float(r.get("final_score") or 0.0) for r in runs) / len(runs)
    acts = sum(sum(r.get("actions_per_level") or []) for r in runs)
    won = sum(1 for r in runs if r.get("state") == "won")
    verdict = "HARM" if lc <= HARM_MAX else ("SIGNAL" if lc >= SIGNAL_MIN else "NULL")
    return {"verdict": verdict, "decisive": True, "lc_total": lc,
            "mean_score": round(ms, 6), "total_actions": acts, "games_won": won,
            "comparator": {"lc": 28, "mean_score": 6.173, "actions": 1639},
            "head_gate_lc28": lc >= 28,
            "reason": f"lc_total {lc} vs bands HARM<=23 | NULL 24-32 | SIGNAL>=33 (1-sigma screen)",
            **facts}


def _selftest() -> int:
    import tempfile
    failures, n = [], [0]

    def expect(name, got, want):
        n[0] += 1
        if got != want:
            failures.append(f"{name}: {got!r} != {want!r}")

    GOOD = ('TAAF_GRAFTS FEATURES={"clickmap":true,"efficiency":true,"goalkeep":true,'
            '"hudmask":true,"retry_guard":true,"shortcircuit":true} API_VERSION=1\n'
            "[goalkeep] armed\n[hudmask] armed\n[clickmap] armed\n"
            f"vLLM server ready: {SERVED_MODEL}\n")

    def make(lc_list, log=GOOD, bench=True, nn=N_GAMES):
        root = Path(tempfile.mkdtemp())
        (root / "kernel.log").write_text(log, encoding="utf-8")
        if bench:
            lcs = list(lc_list) + [0] * (nn - len(lc_list))
            runs = [{"game_id": f"g{i}", "levels_completed": lcs[i], "state": "gave_up",
                     "final_score": 1.0, "actions_per_level": [5]} for i in range(nn)]
            (root / "benchmark.json").write_text(json.dumps({"game_runs": runs}), encoding="utf-8")
        return root

    expect("lc 23 -> HARM", score(make([23]))["verdict"], "HARM")
    expect("lc 24 -> NULL", score(make([24]))["verdict"], "NULL")
    expect("lc 28 -> NULL + head gate", (score(make([28]))["verdict"], score(make([28]))["head_gate_lc28"]), ("NULL", True))
    expect("lc 27 head gate False", score(make([27]))["head_gate_lc28"], False)
    expect("lc 32 -> NULL", score(make([32]))["verdict"], "NULL")
    expect("lc 33 -> SIGNAL", score(make([33]))["verdict"], "SIGNAL")
    expect("no banner -> INFRA", score(make([28], log="x"))["verdict"], "INFRA DEATH")
    expect("clickmap armed line missing -> INFRA",
           score(make([28], log=GOOD.replace("[clickmap] armed\n", "")))["verdict"], "INFRA DEATH")
    expect("clickmap flag missing -> INFRA",
           score(make([28], log=GOOD.replace('"clickmap":true,', "")))["verdict"], "INFRA DEATH")
    expect("searchmap armed -> INFRA",
           score(make([28], log=GOOD + "[searchmap] armed\n"))["verdict"], "INFRA DEATH")
    expect("banking in FEATURES -> INFRA",
           score(make([28], log=GOOD.replace('"clickmap":true,', '"banking":true,"clickmap":true,')))["verdict"], "INFRA DEATH")
    expect("wrong served model -> INFRA",
           score(make([28], log=GOOD.replace(SERVED_MODEL, "vrfai/Qwen3.6-27B-FP8")))["verdict"], "INFRA DEATH")
    expect("effort pin present -> INFRA",
           score(make([28], log=GOOD + "reasoning_effort=low\n"))["verdict"], "INFRA DEATH")
    expect("stock fallback -> INFRA",
           score(make([33], log=GOOD + "[taaf_grafts] install failed -> stock: x\n"))["verdict"], "INFRA DEATH")
    expect("no benchmark -> INFRA", score(make([28], bench=False))["verdict"], "INFRA DEATH")
    expect("n=24 -> INFRA", score(make([28], nn=24))["verdict"], "INFRA DEATH")
    cli = json.dumps([{"stream_name": "stdout", "time": 1.0, "data": l + chr(10)}
                      for l in GOOD.splitlines()])
    expect("CLI-JSON log certifies", score(make([28], log=cli))["verdict"], "NULL")

    if failures:
        print(f"SELFTEST FAILED ({len(failures)}/{n[0]}):")
        for f in failures:
            print("  " + f)
        return 1
    print(f"selftest OK ({n[0]}/{n[0]} checks, 0 failures)")
    print("  HARM <=23 | NULL 24-32 | SIGNAL >=33 | head gate lc>=28 | clickmap REQUIRED, searchmap FORBIDDEN")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        raise SystemExit(_selftest())
    r = score(Path(sys.argv[1]))
    print(json.dumps(r, indent=2, sort_keys=True))
