"""ITEM 5 (R24 §3.4 / §5.3) — the `SAFE_MODULES` gap, resolved + risk-classed.

GAP
---
`python_tool_sandbox.py:42-57` allows 14 stdlib modules and none of
`dataclasses`, `typing`, `enum`. Tycho's `State` dataclass — the artifact schema
lane (a) was ratified on — is therefore NOT CONSTRUCTIBLE in our sandbox.
`sandbox_facts.tycho_state_importable()` returns
`(False, ['dataclasses', 'enum', 'typing'])`.

RESOLUTION
----------
Add exactly those three to the allowlist, warpack-style: rewrite the
`_SANDBOX_BOOTSTRAP` string at import time. No notebook change, no fork edit.
prompts.py:81 (manifest entry PS-02) is patched in the same bundle so the
model-facing allowlist and the real allowlist agree — otherwise item 5 creates
a NEW instruction contradiction and re-breaks K4.

RISK CLASS vs MINUTES §2 ITEM 5
-------------------------------
§2 item 5 ratified: "host-mode execution acceptable only while executed code is
ours, generated offline, byte-audited; if model-authored in-kernel code ever
lands, this verdict must be re-taken."

Measured finding (`escape_reachability()`, reproduced in smoke_test):
`sys` is ALREADY reachable from the CURRENT allowlist via `fractions.sys` and
`statistics.sys`, and `sys.modules['builtins'].__import__` is reachable from
there. The restricted `__builtins__` and `_safe_import` gate `import` statements
only; they do not gate attribute access on an allowed module. So the sandbox is
a HYGIENE boundary, not a security boundary, TODAY, before this patch.

Therefore:
  * The marginal risk of adding dataclasses/typing/enum is ZERO on the escape
    class — the class is already open. (`dataclasses.sys`, `typing.sys`,
    `enum.sys` add nothing that `fractions.sys` does not already give.)
  * The correct reading of §2 item 5 is that the trigger is ALREADY LIVE for the
    `python` tool: it executes model-authored code in-kernel with a reachable
    path to the real `builtins`. This patch does not move that line; it should
    be recorded, not used to block item 5.
  * The blast radius is the Kaggle container on the FREE BUILD RAIL, never a
    scored submission, and the failure mode is a notebook ERROR, not data loss.

MEMORY (§5.4, and this is the P1-specific hazard)
-------------------------------------------------
`_set_limits` sets RLIMIT_CPU / RLIMIT_FSIZE / RLIMIT_NOFILE and NO memory
limit (`sandbox_facts.rlimit_kinds()`). Today that is safe because every child
dies within <=30 s. P1 makes children live for the whole 7,920 s game at a real
steady state of ~25 simultaneous children (banner concurrency=28). Accumulated
namespace content is the mechanism's POINT, so it grows monotonically and is
never reclaimed. `MEMORY_RLIMIT_PATCH` below adds RLIMIT_AS; it is a REQUIRED
companion to any P1 patch, not an optional canary.

Usage:
  uv run python duck_eval/p1_instr/safe_modules_patch.py
"""
from __future__ import annotations

import builtins
import contextlib
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox_facts import (  # noqa: E402
    SANDBOX_SRC,
    TYCHO_STATE_PROBE,
    _bootstrap_source,
    rlimit_kinds,
    safe_builtins,
    safe_modules,
    sandbox_line_index,
    tycho_state_importable,
)

ADDED_MODULES = ("dataclasses", "enum", "typing")

# Baseline SAFE_MODULES literal, exactly as it appears in the DEDENTED bootstrap
# string (`_SANDBOX_BOOTSTRAP` is `textwrap.dedent(...)`ed, so 4 spaces of common
# indent are already gone). Source form is python_tool_sandbox.py:42-57.
_BASE_LITERAL = """SAFE_MODULES = {
    "bisect",
    "collections",
    "copy",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
}"""


def patched_safe_modules() -> tuple[str, ...]:
    return tuple(sorted(set(safe_modules()) | set(ADDED_MODULES)))


def _patched_literal() -> str:
    body = "\n".join(f'    "{m}",' for m in patched_safe_modules())
    return "SAFE_MODULES = {\n" + body + "\n}"


def patch_bootstrap_text(text: str) -> str:
    """Exact-string rewrite of the SAFE_MODULES set inside _SANDBOX_BOOTSTRAP."""
    if _BASE_LITERAL not in text:
        raise ValueError(
            "SAFE_MODULES literal not found verbatim in the bootstrap source; "
            "the fork drifted -- refuse to patch (byte-audit discipline)."
        )
    return text.replace(_BASE_LITERAL, _patched_literal(), 1)


MEMORY_RLIMIT_PATCH = '''\
def _set_limits(timeout_seconds):
    if resource is None:
        return
    cpu_limit = max(1, int(timeout_seconds)) + 1
    try:
        mem_limit = int(os.environ.get("P1_SANDBOX_MEM_BYTES", "0")) or (512 << 20)
    except ValueError:
        mem_limit = 512 << 20
    for limit, value in (
        (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
        (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
        (getattr(resource, "RLIMIT_NOFILE", None), 32),
        (getattr(resource, "RLIMIT_AS", None), mem_limit),
    ):
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (value, value))
        except (OSError, ValueError):
            pass
'''

_BASE_SET_LIMITS = '''\
def _set_limits(timeout_seconds):
    if resource is None:
        return
    cpu_limit = max(1, int(timeout_seconds)) + 1
    for limit, value in (
        (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
        (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
        (getattr(resource, "RLIMIT_NOFILE", None), 32),
    ):
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (value, value))
        except (OSError, ValueError):
            pass
'''


def patch_memory_rlimit(text: str) -> str:
    if _BASE_SET_LIMITS not in text:
        raise ValueError("_set_limits not found verbatim; refuse to patch")
    return text.replace(_BASE_SET_LIMITS, MEMORY_RLIMIT_PATCH, 1)


# --------------------------------------------------------------------------
# The SECOND half of the gap, found while validating this item:
# `class` statements are impossible in the sandbox at all. SAFE_BUILTINS omits
# `__build_class__`, and `runtime_globals` omits `__name__`. Measured:
#     class Foo: pass          -> NameError: __build_class__ not found
#     + __build_class__        -> NameError: name '__name__' is not defined
#     + __build_class__, __name__ -> OK
# So `dataclasses` alone does NOT make Tycho's `State` constructible; without
# these two the module allowlist fix is cosmetic.
# --------------------------------------------------------------------------
ADDED_BUILTINS = ("__build_class__",)
SANDBOX_MODULE_NAME = "__python_tool__"

_BASE_BUILTINS_HEAD = 'SAFE_BUILTINS = {\n    "abs",'
_P1_BUILTINS_HEAD = 'SAFE_BUILTINS = {\n    "__build_class__",\n    "abs",'

_BASE_IMPORTS = "import sys\nimport traceback\n"
_P1_IMPORTS = "import sys\nimport traceback\nimport types\n"

# A bare `__name__` global is NOT enough: `dataclasses._is_type` does an
# UNGUARDED `sys.modules.get(cls.__module__).__dict__`, so the tool module has
# to be a real, registered module. Using its `__dict__` as `runtime_globals`
# also makes P1's persistence natural (the namespace IS a module).
_BASE_RUNTIME_GLOBALS = '''\
    runtime_globals = {
        "__builtins__": {
            name: getattr(builtins, name)
            for name in SAFE_BUILTINS
        },
        "result": None,
    }'''
_P1_RUNTIME_GLOBALS = '''\
    _tool_module = types.ModuleType("__python_tool__")
    sys.modules["__python_tool__"] = _tool_module
    runtime_globals = _tool_module.__dict__
    runtime_globals["__builtins__"] = {
        name: getattr(builtins, name)
        for name in SAFE_BUILTINS
    }
    runtime_globals["result"] = None'''


def patch_class_support(text: str) -> str:
    """Make `class` / `@dataclass` usable at all inside the sandbox."""
    if _BASE_BUILTINS_HEAD not in text:
        raise ValueError("SAFE_BUILTINS head not found verbatim; refuse to patch")
    out = text.replace(_BASE_BUILTINS_HEAD, _P1_BUILTINS_HEAD, 1)
    if _BASE_IMPORTS not in out:
        raise ValueError("bootstrap import block not found verbatim; refuse to patch")
    out = out.replace(_BASE_IMPORTS, _P1_IMPORTS, 1)
    if _BASE_RUNTIME_GLOBALS not in out:
        raise ValueError("runtime_globals literal not found verbatim; refuse to patch")
    return out.replace(_BASE_RUNTIME_GLOBALS, _P1_RUNTIME_GLOBALS, 1)


def patch_all(text: str, *, with_memory_rlimit: bool = True) -> str:
    out = patch_bootstrap_text(text)
    out = patch_class_support(out)
    if with_memory_rlimit:
        out = patch_memory_rlimit(out)
    return out


def install(*, with_memory_rlimit: bool = True) -> str:
    """Warpack-style monkeypatch: rewrite the live _SANDBOX_BOOTSTRAP string."""
    from inference.agent import python_tool_sandbox as pts  # noqa: PLC0415

    pts._SANDBOX_BOOTSTRAP = patch_all(
        pts._SANDBOX_BOOTSTRAP, with_memory_rlimit=with_memory_rlimit)
    banner = (
        f"p1_instr: SAFE_MODULES += {list(ADDED_MODULES)}; "
        f"SAFE_BUILTINS += {list(ADDED_BUILTINS)}; __name__ set; "
        f"RLIMIT_AS={'on' if with_memory_rlimit else 'off'}"
    )
    print(banner)
    return banner


# --------------------------------------------------------------------------
# risk assessment (measured, not asserted)
# --------------------------------------------------------------------------
def escape_reachability(modules: tuple[str, ...] | frozenset[str]) -> dict[str, list[str]]:
    """Which allowlisted modules expose an attribute that escapes the sandbox."""
    import importlib

    interesting = ("builtins", "sys", "os", "subprocess", "importlib", "inspect")
    out: dict[str, list[str]] = {}
    for name in sorted(modules):
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
        hits = [a for a in interesting if hasattr(mod, a)]
        if hits:
            out[name] = hits
    return out


def escape_demo() -> dict[str, object]:
    """Prove, in-process, that the CURRENT allowlist already reaches `builtins`."""
    mods = safe_modules()

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        root = str(name or "").split(".", 1)[0]
        if root not in mods:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)

    g = {"__builtins__": {n: getattr(builtins, n) for n in safe_builtins()}, "result": None}
    g["__builtins__"]["__import__"] = _safe_import

    direct_blocked = False
    try:
        exec("import os", g, g)  # noqa: S102
    except ImportError:
        direct_blocked = True

    exec(  # noqa: S102
        "import fractions\n"
        "reached_sys = hasattr(fractions, 'sys')\n"
        "reached_builtins = hasattr(fractions.sys.modules.get('builtins'), '__import__')\n",
        g, g,
    )
    return {
        "direct_os_import_blocked": direct_blocked,
        "sys_reachable_via_allowlisted_module": bool(g["reached_sys"]),
        "real_import_reachable": bool(g["reached_builtins"]),
        "conclusion": (
            "escape class ALREADY OPEN before the patch; adding "
            f"{list(ADDED_MODULES)} does not change the risk class"
        ),
    }


def tycho_probe(modules: frozenset[str] | tuple[str, ...], *,
                extra_builtins: tuple[str, ...] = (),
                with_module_name: bool = False,
                register_module: bool = False) -> dict[str, object]:
    """Run the Tycho `State` probe under a given allowlist, in sandbox-like globals."""
    mods = frozenset(modules)

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        root = str(name or "").split(".", 1)[0]
        if root not in mods:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)

    names = list(safe_builtins()) + list(extra_builtins)
    if register_module:
        import types  # noqa: PLC0415
        mod = types.ModuleType(SANDBOX_MODULE_NAME)
        sys.modules[SANDBOX_MODULE_NAME] = mod
        g: dict[str, object] = mod.__dict__
        g["result"] = None
    else:
        g = {"result": None}
        if with_module_name:
            g["__name__"] = SANDBOX_MODULE_NAME
    g["__builtins__"] = {n: getattr(builtins, n) for n in names}
    g["__builtins__"]["__import__"] = _safe_import  # type: ignore[index]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(TYCHO_STATE_PROBE, "<python_tool>", "exec"), g, g)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "result": g.get("result")}




# --------------------------------------------------------------------------
# §5.4 canary spec corrections, carried
# --------------------------------------------------------------------------
P1_SANDBOX_CANARY_SPEC: dict[str, object] = {
    "source": "R24 minutes §5.4 (systems), applied to proposal §6.2",
    "live_children": {
        "WRONG": "live-child count <= concurrency (16)",
        "CORRECT": (
            "live_children <= observed_concurrent_games, with the constant read "
            "from the run's own solver banner (war_eval_v1 log:657 -> "
            "concurrency=28, steady state ~25 simultaneous games)"
        ),
        "read_from": "pull_io.parse_solver_banner()['concurrency']",
    },
    "memory": {
        "finding": "_set_limits sets RLIMIT_CPU/FSIZE/NOFILE and NO memory limit",
        "baseline_safe_because": "every child dies in <=30 s",
        "p1_hazard": (
            "persistent children x ~25 concurrent games x 7920 s per game, with "
            "monotonically growing namespaces (accumulation is the mechanism), "
            "and no rlimit -> whole-notebook ERROR"
        ),
        "required_patch": "RLIMIT_AS via MEMORY_RLIMIT_PATCH",
        "default_per_child_bytes": 512 << 20,
        "sizing_rule": (
            "per_child = max(256 MiB, floor(0.25 * MemAvailable / "
            "observed_concurrent_games)); publish the value in the prereg"
        ),
    },
    "namespace_destruction": {
        "event": "per-call timeout -> _kill_process_group "
                 "(python_tool_sandbox.py:423, fired at :503)",
        "NOT": "RLIMIT_CPU",
        "consequence": "one timeout silently reverts a game to ephemeral",
        "instrument": "namespace_reuse.py epoch-conditions on FAULT_PATTERNS and "
                      "refuses to fire K4 below MIN_EPOCH_INTACT",
    },
    "rlimit_cpu_reaccounting": {
        "finding": (
            "cpu_limit = int(timeout_seconds)+1 is set ONCE at child start "
            "(python_tool_sandbox.py:318). Under P1 that per-call limit becomes a "
            "per-GAME limit: a persistent child accumulates CPU seconds across "
            "every call and is SIGXCPU-killed after ~31 s of total CPU."
        ),
        "required": (
            "P1 must re-account RLIMIT_CPU per game or remove it in favour of the "
            "host-side wall-clock deadline, and must report SIGXCPU kills "
            "separately from timeouts (they are a different destruction cause)."
        ),
    },
    "build_budget": "~12-13 builds/week (30 GPU-h / 2.2-2.4 h), below 2 pushes/day",
}


def main(argv: list[str]) -> int:
    print("=== item 5: SAFE_MODULES gap ===")
    print(f"sandbox source : {SANDBOX_SRC}")
    print(f"line index     : {sandbox_line_index()}")
    print(f"baseline       : {sorted(safe_modules())}")
    ok, missing = tycho_state_importable()
    print(f"Tycho State constructible on baseline? {ok}  missing={missing}")
    print(f"patched        : {list(patched_safe_modules())}")
    ok2, missing2 = tycho_state_importable(frozenset(patched_safe_modules()))
    print(f"Tycho State constructible after patch?  {ok2}  missing={missing2}")

    print("\n-- probe executed under sandbox-like globals --")
    print(f"baseline allowlist            : {json.dumps(tycho_probe(safe_modules()))}")
    print(f"modules patched only          : "
          f"{json.dumps(tycho_probe(patched_safe_modules()))}")
    print(f"modules + __build_class__     : "
          f"{json.dumps(tycho_probe(patched_safe_modules(), extra_builtins=ADDED_BUILTINS))}")
    print(f"modules + bc + bare __name__  : "
          f"{json.dumps(tycho_probe(patched_safe_modules(), extra_builtins=ADDED_BUILTINS, with_module_name=True))}")
    print(f"FULL PATCH (module registered): "
          f"{json.dumps(tycho_probe(patched_safe_modules(), extra_builtins=ADDED_BUILTINS, register_module=True))}")
    print(f"SAFE_BUILTINS has __build_class__: {'__build_class__' in safe_builtins()} "
          f"=> `class Foo: pass` is impossible in the baseline sandbox")

    print("\n-- risk class --")
    print(f"rlimits set by _set_limits: {sorted(rlimit_kinds())} "
          f"(memory limit present: {'RLIMIT_AS' in rlimit_kinds()})")
    print(f"escape reachability (baseline allowlist): "
          f"{json.dumps(escape_reachability(safe_modules()))}")
    print(f"escape reachability (added modules):      "
          f"{json.dumps(escape_reachability(ADDED_MODULES))}")
    print(f"escape demo: {json.dumps(escape_demo(), indent=2)}")

    print("\n-- patch dry run --")
    text = _bootstrap_source()
    patched = patch_all(text)
    print(f"bootstrap chars {len(text)} -> {len(patched)}  "
          f"(SAFE_MODULES +{len(ADDED_MODULES)}, "
          f"SAFE_BUILTINS +{len(ADDED_BUILTINS)}, "
          f"tool module registered: {'sys.modules[\"__python_tool__\"]' in patched}, "
          f"RLIMIT_AS added: {'RLIMIT_AS' in patched})")
    import ast as _ast
    _ast.parse(patched)
    print("patched bootstrap parses clean: True")

    print("\n-- §5.4 canary spec --")
    print(json.dumps(P1_SANDBOX_CANARY_SPEC, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
