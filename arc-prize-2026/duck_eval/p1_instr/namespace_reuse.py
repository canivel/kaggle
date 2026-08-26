"""ITEM 2 (R24 §5.3 / §3.4, R25 N1/N2 FATAL) — `namespace_reuse_rate`, defined and validated.

`namespace_reuse_rate < 0.15` is K4 and, via Option-4's off-ramp, the trigger
that CONCEDES the campaign. R25 filed it FATAL as undefined and confounded.
This module gives it an operational definition, an estimator, and a measured
baseline distribution on no-P1 transcripts.

================================ DEFINITION ================================

UNIT.  One `python` tool call. At baseline that is exactly one sandbox child
process (`python_tool_sandbox.py:458` spawns one `Popen` per call), and under
P1 it is exactly one turn against the persistent namespace. Analysis steps are
NOT the unit: 25/25 war_eval_v1 games issue more than one call per step.

For call t of game g, from the code string the model actually sent:

  M(t)  module-scope bindings created by t -- assignments, imports, def, class
        at module scope. These, and only these, are the names that would
        survive into a persistent namespace. Function locals and comprehension
        targets are excluded because they would not survive either.

  F(t)  free global references of t -- every name Loaded by t (at module scope
        or in any nested scope where it resolves to global) that is NOT in
        M(t), not a sandbox-exposed builtin (SAFE_BUILTINS, parsed from the
        fork), and not a harness pre-loaded global (`current_frame`, `action`,
        ... see sandbox_facts.HARNESS_GLOBALS).

  EPOCH  a maximal run of consecutive calls in one game with no intervening
        namespace-destroying event.  §5.4: the destroying event is the per-call
        TIMEOUT (`_kill_process_group`, python_tool_sandbox.py:423/503), NOT
        RLIMIT_CPU. Any host-side fault in FAULT_PATTERNS kills the child, so
        the next call starts a fresh epoch. At baseline the epoch is
        counterfactual -- "the namespace P1 would have given you".

  P(t)  = union of M(t') over t' < t in the SAME epoch.

  REUSE call:  F(t) & P(t) != {}.
  ELIGIBLE call: t has >= 1 predecessor in its own epoch.

  namespace_reuse_rate  ==  |REUSE| / |ELIGIBLE|          (epoch-conditioned)

============================ CONFOUND SEPARATION ============================

The infrastructure channel and the hypothesis channel are reported SEPARATELY
and the gate reads the epoch-conditioned rate:

  nrr_epoch  |REUSE| / |ELIGIBLE|                <- K4 reads THIS
  nrr_raw    |REUSE| / |calls with index >= 1|   <- what a naive estimator reads
  epoch_intact_fraction = |ELIGIBLE| / |calls with index >= 1|
  infra_attributable_drop = nrr_epoch - nrr_raw  (>= 0 by construction)
  max_infra_drop = nrr_epoch * (1 - epoch_intact_fraction)
                   the largest depression infrastructure could produce

If `epoch_intact_fraction` is below `MIN_EPOCH_INTACT`, the estimator reports
`INFRA-CONFOUNDED` and K4 MUST NOT FIRE -- the run is void on the substrate
question, not evidence against the hypothesis. This is the explicit answer to
"K4 can fire on infrastructure": the instrument either separates the channels
or refuses to report.

========================= INDEPENDENT EVIDENCE CHANNEL ======================

`expected_persistence_errors`: baseline calls whose result carries
`NameError: name 'X' is not defined` for an X in P(t). These are calls where
the model BEHAVIOURALLY assumed persistence and the ephemeral substrate refused
-- a direct, score-free read of latent demand for the substrate that does not
depend on the syntactic estimator at all. (M3 lesson: never seal a floor on one
instrument.)

Usage:
  uv run python duck_eval/p1_instr/namespace_reuse.py                     # baseline family
  uv run python duck_eval/p1_instr/namespace_reuse.py PULL [PULL ...]
  uv run python duck_eval/p1_instr/namespace_reuse.py --json out.json PULL ...
"""
from __future__ import annotations

import ast
import json
import re
import statistics as st
import symtable
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_io import ToolCall, load_calls, load_pull  # noqa: E402
from sandbox_facts import HARNESS_GLOBALS, safe_builtins, safe_modules  # noqa: E402

ESTIMATOR_VERSION = "nrr-1.0.0"
MIN_EPOCH_INTACT = 0.90   # below this the arm is INFRA-CONFOUNDED, K4 must not fire
BASELINE_FAMILY = ["war_eval_v1", "war_eval_v2", "war_eval_v3"]

_NAMEERROR_RE = re.compile(r"NameError: name '([A-Za-z_]\w*)' is not defined")


# --------------------------------------------------------------------------
# static analysis of one code block
# --------------------------------------------------------------------------
@dataclass
class BlockNames:
    parsed: bool
    module_bound: set[str] = field(default_factory=set)
    free_refs: set[str] = field(default_factory=set)


def _comprehension_targets(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in node.generators:
                for sub in ast.walk(gen.target):
                    if isinstance(sub, ast.Name):
                        out.add(sub.id)
    return out


def _referenced_globals(table: symtable.SymbolTable, acc: set[str]) -> None:
    for sym in table.get_symbols():
        if sym.is_referenced() and sym.is_global():
            acc.add(sym.get_name())
    for child in table.get_children():
        _referenced_globals(child, acc)


_SCOPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_SIMPLE = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr, ast.Return,
           ast.Delete, ast.Assert, ast.Raise, ast.Import, ast.ImportFrom)


def _stmt_stores(stmt: ast.stmt) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out.add((a.asname or a.name).split(".", 1)[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            # `except X as e` binds e inside the handler (and unbinds after)
            out.add(node.name)
        elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
            out.add(node.optional_vars.id)
    return out


def _direct_loads(stmt: ast.stmt) -> set[str]:
    """Name loads evaluated at module scope in this statement.

    Bodies of `def`/`class`/`lambda` are NOT module-scope evaluation -- they run
    later (or never) -- so they are skipped. Decorators, default arguments and
    base-class expressions DO evaluate at definition time and are included.
    """
    out: set[str] = set()

    def walk_expr(node: ast.AST | None) -> None:
        if node is None:
            return
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                out.add(sub.id)

    def walk(node: ast.AST) -> None:
        if isinstance(node, _SCOPED):
            for sub in getattr(node, "decorator_list", []) or []:
                walk_expr(sub)
            for sub in getattr(node, "bases", []) or []:
                walk_expr(sub)
            args = getattr(node, "args", None)
            if isinstance(args, ast.arguments):
                for sub in list(args.defaults) + [d for d in args.kw_defaults if d]:
                    walk_expr(sub)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            out.add(node.id)
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(stmt)
    return out


def _use_before_binding(tree: ast.Module) -> set[str]:
    """Module-level names loaded before they are bound in the SAME block.

    These are real NameErrors at runtime even though `symtable` reports the name
    as module-bound (it binds later in the block). Under a persistent namespace
    they resolve silently to a prior turn's value -- i.e. they are exactly the
    reuse events the naive estimator would miss. Conservative by construction:
    compound statements have their whole store set admitted before their loads
    are checked, so loop-carried bindings cannot create a false positive.
    """
    bound: set[str] = set()
    free: set[str] = set()
    for stmt in tree.body:
        stores = _stmt_stores(stmt)
        if not isinstance(stmt, _SIMPLE):
            bound |= stores
        free |= _direct_loads(stmt) - bound
        bound |= stores
    return free


def analyse_block(code: str) -> BlockNames:
    """Module-scope bindings and unresolved global references of one code block."""
    try:
        tree = ast.parse(code)
        top = symtable.symtable(code, "<python_tool>", "exec")
    except (SyntaxError, ValueError):
        return BlockNames(parsed=False)

    comps = _comprehension_targets(tree)
    bound = {
        s.get_name()
        for s in top.get_symbols()
        if (s.is_assigned() or s.is_imported()) and s.get_name() not in comps
    }
    refs: set[str] = set()
    _referenced_globals(top, refs)
    for s in top.get_symbols():
        if s.is_referenced() and not (s.is_assigned() or s.is_imported()):
            refs.add(s.get_name())

    free = refs - bound - set(safe_builtins()) - set(HARNESS_GLOBALS) - comps
    # names bound later in the same block still NameError at runtime
    free |= _use_before_binding(tree) - set(safe_builtins()) - set(HARNESS_GLOBALS) - comps
    free -= {"__import__"}
    return BlockNames(parsed=True, module_bound=bound, free_refs=free)


# --------------------------------------------------------------------------
# per-game estimator
# --------------------------------------------------------------------------
@dataclass
class GameNRR:
    game_id: str
    n_calls: int
    n_unparsable: int
    n_epochs: int
    destruction_events: dict[str, int]
    n_index_ge1: int          # calls that have a predecessor in the game
    n_eligible: int           # calls that have a predecessor in their EPOCH
    n_reuse: int
    n_post_fault: int         # calls living in an epoch after a destruction
    n_recovery: int           # post-fault calls re-defining a destroyed name
    nrr_epoch: float
    nrr_raw: float
    nrr_prefault: float       # rate on epoch 0 only (infrastructure-free segment)
    epoch_intact_fraction: float
    post_fault_fraction: float
    infra_attributable_drop: float
    max_infra_drop: float
    expected_persistence_errors: int
    dangling_ref_calls: int
    reuse_examples: list[str]


def game_nrr(calls: list[ToolCall], game_id: str) -> GameNRR:
    blocks = [analyse_block(c.code) for c in calls]
    destruction: dict[str, int] = {}
    prior: set[str] = set()
    epoch_len = 0
    n_epochs = 1 if calls else 0
    n_eligible = n_reuse = n_index_ge1 = 0
    n_pre_elig = n_pre_reuse = 0
    n_post_fault = n_recovery = 0
    unparsable = sum(1 for b in blocks if not b.parsed)
    persistence_errors = 0
    dangling_calls = 0
    examples: list[str] = []
    lost: set[str] = set()          # names a fault destroyed
    faulted_yet = False

    mods = set(safe_modules())

    for i, (call, blk) in enumerate(zip(calls, blocks)):
        if i >= 1:
            n_index_ge1 += 1
            if faulted_yet:
                n_post_fault += 1
        eligible = epoch_len >= 1
        if eligible:
            n_eligible += 1
            if not faulted_yet:
                n_pre_elig += 1
        if blk.parsed:
            hits = blk.free_refs & prior
            if eligible and hits:
                n_reuse += 1
                if not faulted_yet:
                    n_pre_reuse += 1
                if len(examples) < 5:
                    examples.append(f"call#{i}:{sorted(hits)[:4]}")
            if faulted_yet and (blk.module_bound & lost):
                n_recovery += 1
            dangling = blk.free_refs - prior - mods
            if dangling:
                dangling_calls += 1
            # independent channel: the substrate actually refused a prior name
            for name in _NAMEERROR_RE.findall(call.result):
                if name in prior:
                    persistence_errors += 1
                    break

        # advance namespace state
        if blk.parsed:
            prior = prior | blk.module_bound
        epoch_len += 1
        if call.is_fault:
            destruction[call.fault] = destruction.get(call.fault, 0) + 1
            lost |= prior
            prior = set()
            epoch_len = 0
            faulted_yet = True
            if i + 1 < len(calls):
                n_epochs += 1

    nrr_epoch = n_reuse / n_eligible if n_eligible else 0.0
    nrr_raw = n_reuse / n_index_ge1 if n_index_ge1 else 0.0
    nrr_pre = n_pre_reuse / n_pre_elig if n_pre_elig else 0.0
    intact = n_eligible / n_index_ge1 if n_index_ge1 else 1.0
    post_frac = n_post_fault / n_index_ge1 if n_index_ge1 else 0.0
    return GameNRR(
        game_id=game_id,
        n_calls=len(calls),
        n_unparsable=unparsable,
        n_epochs=n_epochs,
        destruction_events=destruction,
        n_index_ge1=n_index_ge1,
        n_eligible=n_eligible,
        n_reuse=n_reuse,
        n_post_fault=n_post_fault,
        n_recovery=n_recovery,
        nrr_epoch=round(nrr_epoch, 6),
        nrr_raw=round(nrr_raw, 6),
        nrr_prefault=round(nrr_pre, 6),
        epoch_intact_fraction=round(intact, 6),
        post_fault_fraction=round(post_frac, 6),
        infra_attributable_drop=round(nrr_epoch - nrr_raw, 6),
        # Upper bound on the depression infrastructure can produce: epoch
        # conditioning removes the call that IMMEDIATELY follows a destruction,
        # but every later call in a reconstituted epoch also lost its helpers.
        # If all of them would otherwise have reused, the loss is post_frac.
        max_infra_drop=round(post_frac, 6),
        expected_persistence_errors=persistence_errors,
        dangling_ref_calls=dangling_calls,
        reuse_examples=examples,
    )


def pull_nrr(pull_name: str) -> dict[str, object]:
    pull = load_pull(pull_name)
    calls = load_calls(pull)
    rows = [game_nrr(calls[g], g) for g in sorted(calls)]
    tot_reuse = sum(r.n_reuse for r in rows)
    tot_elig = sum(r.n_eligible for r in rows)
    tot_idx1 = sum(r.n_index_ge1 for r in rows)
    tot_calls = sum(r.n_calls for r in rows)
    destruction: dict[str, int] = {}
    for r in rows:
        for k, v in r.destruction_events.items():
            destruction[k] = destruction.get(k, 0) + v
    per_game = [r.nrr_epoch for r in rows]
    pooled_epoch = tot_reuse / tot_elig if tot_elig else 0.0
    intact = tot_elig / tot_idx1 if tot_idx1 else 1.0
    tot_post = sum(r.n_post_fault for r in rows)
    post_frac = tot_post / tot_idx1 if tot_idx1 else 0.0
    status = "OK" if post_frac <= (1.0 - MIN_EPOCH_INTACT) else "INFRA-CONFOUNDED"
    n_faults = sum(destruction.values())
    # Rule of three: with 0/N observed faults the 95% upper bound on the
    # per-call fault probability is 3/N.
    q_hat = n_faults / tot_calls if tot_calls else 0.0
    q_upper95 = max(q_hat, 3.0 / tot_calls) if tot_calls else 1.0
    mean_calls_per_game = tot_calls / len(rows) if rows else 0.0
    epoch_survival_lb = (1.0 - q_upper95) ** mean_calls_per_game
    return {
        "estimator_version": ESTIMATOR_VERSION,
        "pull": pull_name,
        "label": pull.label,
        "banner": pull.banner,
        "n_games": len(rows),
        "n_calls": tot_calls,
        "n_unparsable": sum(r.n_unparsable for r in rows),
        "pooled_nrr_epoch": round(pooled_epoch, 6),
        "pooled_nrr_raw": round(tot_reuse / tot_idx1, 6) if tot_idx1 else 0.0,
        "per_game_nrr_mean": round(st.mean(per_game), 6) if per_game else 0.0,
        "per_game_nrr_sd": round(st.stdev(per_game), 6) if len(per_game) > 1 else 0.0,
        "per_game_nrr_min": round(min(per_game), 6) if per_game else 0.0,
        "per_game_nrr_max": round(max(per_game), 6) if per_game else 0.0,
        "per_game_nrr_median": round(st.median(per_game), 6) if per_game else 0.0,
        "n_games_below_0_15": sum(1 for x in per_game if x < 0.15),
        "epoch_intact_fraction": round(intact, 6),
        "post_fault_fraction": round(post_frac, 6),
        "max_infra_drop": round(post_frac, 6),
        "n_recovery_calls": sum(r.n_recovery for r in rows),
        "destruction_events": destruction,
        "per_call_fault_rate": round(q_hat, 8),
        "per_call_fault_rate_upper95": round(q_upper95, 8),
        "mean_calls_per_game": round(mean_calls_per_game, 2),
        "projected_epoch_survival_lower_bound": round(epoch_survival_lb, 6),
        "expected_persistence_errors": sum(r.expected_persistence_errors for r in rows),
        "dangling_ref_calls": sum(r.dangling_ref_calls for r in rows),
        "status": status,
        "games": [asdict(r) for r in rows],
    }


def k4_read(report: dict[str, object], floor: float) -> dict[str, object]:
    """Apply a K4 floor to an estimator report, with the infra guard in front."""
    if report["status"] != "OK":
        return {"k4": "VOID-INFRA-CONFOUNDED", "may_fire": False,
                "reason": f"post_fault_fraction {report.get('post_fault_fraction')} "
                          f"> {1.0 - MIN_EPOCH_INTACT:.2f}"}
    value = float(report["pooled_nrr_epoch"])
    max_drop = float(report.get("max_infra_drop", 0.0))
    if value < floor and (value + max_drop) >= floor:
        return {"k4": "INDETERMINATE", "may_fire": False, "value": value,
                "reason": "reading is below floor only within the infrastructure "
                          "attribution band"}
    return {"k4": "FIRE" if value < floor else "PASS",
            "may_fire": value < floor, "value": value, "floor": floor}


def main(argv: list[str]) -> int:
    out_path = None
    args = argv[1:]
    if args and args[0] == "--json":
        out_path = Path(args[1])
        args = args[2:]
    pulls = args or BASELINE_FAMILY
    reports = []
    for name in pulls:
        rep = pull_nrr(name)
        reports.append(rep)
        print(f"\n=== {name} ({rep['label']}) :: {ESTIMATOR_VERSION} ===")
        print(f"calls={rep['n_calls']} games={rep['n_games']} "
              f"unparsable={rep['n_unparsable']}")
        print(f"pooled nrr_epoch = {rep['pooled_nrr_epoch']:.4f}   "
              f"pooled nrr_raw = {rep['pooled_nrr_raw']:.4f}")
        print(f"per-game nrr: mean={rep['per_game_nrr_mean']:.4f} "
              f"sd={rep['per_game_nrr_sd']:.4f} "
              f"min={rep['per_game_nrr_min']:.4f} "
              f"median={rep['per_game_nrr_median']:.4f} "
              f"max={rep['per_game_nrr_max']:.4f}")
        print(f"games below 0.15 floor: {rep['n_games_below_0_15']}/{rep['n_games']}")
        print(f"epoch_intact_fraction = {rep['epoch_intact_fraction']:.4f}  "
              f"post_fault_fraction = {rep['post_fault_fraction']:.4f}  "
              f"destruction_events={rep['destruction_events']}  status={rep['status']}")
        print(f"per-call fault rate = {rep['per_call_fault_rate']:.6f} "
              f"(95% upper {rep['per_call_fault_rate_upper95']:.6f}); "
              f"projected per-game epoch survival >= "
              f"{rep['projected_epoch_survival_lower_bound']:.4f}")
        print(f"expected_persistence_errors (NameError on a prior-turn name) = "
              f"{rep['expected_persistence_errors']}")
        print(f"K4 @0.15 -> {k4_read(rep, 0.15)}")

    if len(reports) > 1:
        pooled = [float(r["pooled_nrr_epoch"]) for r in reports]
        print(f"\n=== baseline family across {len(reports)} pulls ===")
        print(f"pooled nrr_epoch per pull: "
              f"{[round(x, 4) for x in pooled]}")
        print(f"family mean={st.mean(pooled):.4f} "
              f"sd={st.stdev(pooled):.4f}" if len(pooled) > 1 else "")
        all_games = [g["nrr_epoch"] for r in reports for g in r["games"]]
        all_games_sorted = sorted(all_games)
        n = len(all_games_sorted)
        pct = lambda q: all_games_sorted[min(n - 1, max(0, int(round(q * (n - 1)))))]  # noqa: E731
        print(f"pooled per-game distribution (n={n}): "
              f"p05={pct(0.05):.4f} p10={pct(0.10):.4f} p25={pct(0.25):.4f} "
              f"p50={pct(0.50):.4f} p75={pct(0.75):.4f} p95={pct(0.95):.4f}")
        print(f"per-game readings below 0.15: "
              f"{sum(1 for x in all_games if x < 0.15)}/{n}")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
