"""ITEM 3 (R24 §3.4 / §5.3) — prompt + tool-schema strings as a DECLARED patch surface.

llm-agents' FATAL: P1's mechanism is contradicted by ~6 strings in our own
harness, so `namespace_reuse_rate` "can pass validly but cannot FAIL validly" —
a low reading would measure instruction conflict, not the substrate. At 27B the
TOOL SCHEMA beats the system prompt, and two of the contradictions live in the
schema.

This module makes that surface explicit, enumerated, hashed and auditable:

  * `SURFACE`            -- the declared entries (id, file, line, action)
  * `build_manifest()`   -- resolve each entry against the frozen fork and hash it
  * `audit()`            -- fail loudly if the fork drifted from the manifest
  * `sweep_unlisted()`   -- regex sweep of the whole agent package for
                            ephemerality / allowlist / anti-framework language
                            that is NOT in the manifest (catches new
                            contradictions before they silently void K4)
  * `verify_patched()`   -- byte-audit a patched tree: every PATCH entry must
                            equal its declared replacement, every KEEP entry
                            must be byte-identical to baseline

The S2 prereg must cite `patch_surface_manifest.json` by sha256; the screen must
patch every `action == "PATCH"` entry and byte-audit the result before push.

Usage:
  uv run python duck_eval/p1_instr/patch_surface.py build   # (re)write manifest
  uv run python duck_eval/p1_instr/patch_surface.py audit
  uv run python duck_eval/p1_instr/patch_surface.py sweep
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox_facts import PROMPTS_SRC, ROOT, TOOL_AGENT_SRC, safe_modules  # noqa: E402

MANIFEST_PATH = Path(__file__).resolve().parent / "patch_surface_manifest.json"
AGENT_DIR = PROMPTS_SRC.parent

# The P1 allowlist line has to agree with the patched SAFE_MODULES (item 5), or
# prompts.py:81 becomes a NEW contradiction the moment item 5 lands.
P1_SAFE_MODULES = sorted(set(safe_modules()) | {"dataclasses", "enum", "typing"})
_P1_MODULE_LINE = (
    '    "- The only importable standard-library modules are: '
    + ", ".join(P1_SAFE_MODULES)
    + '.\\n"'
)

SURFACE: list[dict[str, object]] = [
    {
        "id": "PS-01",
        "file": "prompts.py",
        "line": 80,
        "symbol": "PYTHON_ADDENDUM",
        "channel": "system_prompt",
        "contradiction": "ephemerality",
        "severity": "high",
        "action": "PATCH",
        "baseline": '    "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\\n"',
        "p1": '    "- Names you bind at module level in a `python` tool call PERSIST into every later call in this game. Define your world-model helpers once and reuse them by name; re-define only if a call reports a sandbox fault.\\n"',
    },
    {
        "id": "PS-02",
        "file": "prompts.py",
        "line": 81,
        "symbol": "PYTHON_ADDENDUM",
        "channel": "system_prompt",
        "contradiction": "module_allowlist",
        "severity": "medium",
        "action": "PATCH",
        "note": "must be kept in sync with SAFE_MODULES (item 5) or it becomes a new contradiction",
        "baseline": '    "- The only importable standard-library modules are: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string.\\n"',
        "p1": _P1_MODULE_LINE,
    },
    {
        "id": "PS-03",
        "file": "prompts.py",
        "line": 82,
        "symbol": "PYTHON_ADDENDUM",
        "channel": "system_prompt",
        "contradiction": "ephemerality",
        "severity": "high",
        "action": "PATCH",
        "baseline": '    "- The only tool is `python`; call it with one ephemeral `code` string.\\n"',
        "p1": '    "- The only tool is `python`; call it with one `code` string that runs in a namespace shared across the whole game.\\n"',
    },
    {
        "id": "PS-04",
        "file": "prompts.py",
        "line": 107,
        "symbol": "COMPACT_TOOL_SESSION_ADDENDUM",
        "channel": "system_prompt",
        "contradiction": "ephemerality",
        "severity": "high",
        "action": "PATCH",
        "baseline": '    "- The `python` tool code is not saved between calls, so rewrite any custom utility logic you still need.\\n"',
        "p1": '    "- The `python` tool namespace is preserved between calls: functions, classes and variables you defined earlier stay available. Rewrite them only after a reported sandbox fault.\\n"',
    },
    {
        "id": "PS-05",
        "file": "prompts.py",
        "line": 111,
        "symbol": "COMPACT_TOOL_SESSION_ADDENDUM",
        "channel": "system_prompt",
        "contradiction": "timeout_semantics",
        "severity": "high",
        "action": "PATCH",
        "note": (
            "The 30s cap is still TRUE and must stay, but under P1 a timeout is "
            "the namespace-destroying event (_kill_process_group). The model must "
            "be told, or a post-timeout NameError storm reads as non-adoption."
        ),
        "baseline": '    "- Each `python` tool call has a hard time limit of 30 seconds.\\n"',
        "p1": '    "- Each `python` tool call has a hard time limit of 30 seconds. If a call times out the shared namespace is reset, so re-define your helpers on the next call.\\n"',
    },
    {
        "id": "PS-06",
        "file": "prompts.py",
        "line": 113,
        "symbol": "COMPACT_TOOL_SESSION_ADDENDUM",
        "channel": "system_prompt",
        "contradiction": "anti_framework",
        "severity": "medium",
        "action": "PATCH",
        "baseline": '    "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\\n"',
        "p1": '    "- Keep each call\'s OUTPUT short. You may build a larger framework up across calls, since definitions persist; just keep any single call focused.\\n"',
    },
    {
        "id": "PS-07",
        "file": "tool_agent.py",
        "line": 230,
        "symbol": "_PYTHON_TOOL_DESCRIPTION",
        "channel": "tool_schema",
        "contradiction": "ephemerality",
        "severity": "critical",
        "note": "at 27B the tool schema beats the system prompt (llm-agents FATAL)",
        "action": "PATCH",
        "baseline": '    "Run one ephemeral Python snippet against preloaded ASCII game state. Available globals: "',
        "p1": '    "Run a Python snippet in a namespace that PERSISTS across calls for this game. Available globals: "',
    },
    {
        "id": "PS-08",
        "file": "tool_agent.py",
        "line": 1347,
        "symbol": "_tools()::code.description",
        "channel": "tool_schema",
        "contradiction": "ephemerality",
        "severity": "critical",
        "note": "the `code` parameter's own JSON schema description",
        "action": "PATCH",
        "baseline": '                                    "Python code to run. The snippet is ephemeral and is not saved across tool calls."',
        "p1": '                                    "Python code to run. Module-level names you define are saved and stay available to later tool calls in this game."',
    },
    {
        "id": "PS-09",
        "file": "prompts.py",
        "line": 61,
        "symbol": "STRUCTURED_RUNTIME_STATE_ADDENDUM",
        "channel": "system_prompt",
        "contradiction": "partial_persistence_claim",
        "severity": "low",
        "action": "KEEP",
        "note": (
            "The ONE thing the baseline prompt says does persist. It does not "
            "contradict P1, but it is model-facing persistence language and must "
            "be declared so the sweep stays clean and so the arm cannot silently "
            "reword it (that would confound K4)."
        ),
        "baseline": None,   # captured verbatim from the fork at manifest build
        "p1": None,
    },
    {
        "id": "PS-10",
        "file": "python_tool_sandbox.py",
        "line": 506,
        "symbol": "run_sandboxed_python::timeout branch",
        "channel": "tool_result",
        "contradiction": "namespace_destruction_signal",
        "severity": "critical",
        "action": "KEEP",
        "note": (
            "This is the observable signature of the namespace-destroying event "
            "(§5.4). namespace_reuse.FAULT_PATTERNS parses it to epoch-condition "
            "K4. If the arm reworded it, the confound separation would silently "
            "stop working. KEEP byte-identical."
        ),
        "baseline": None,   # captured verbatim from the fork at manifest build
        "p1": None,
    },
]

# Language that would contradict P1 if it appeared anywhere else in the agent
# package. Any hit outside the manifest is an UNDECLARED contradiction.
_SWEEP_PATTERNS = [
    r"ephemeral",
    r"starts fresh",
    r"not saved (?:between|across)",
    r"re-?import",
    r"re-?define",
    r"only importable",
    r"large frameworks",
    r"each (?:call|snippet) (?:is|runs)",
    r"does not persist",
    r"no state is kept",
    r"hard time limit",
    r"time limit of \d+ second",
    r"timed out",
    r"rewrite any",
    r"persist",
]
_SWEEP_RE = re.compile("|".join(f"(?:{p})" for p in _SWEEP_PATTERNS), re.I)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _src(name: str) -> Path:
    return {
        "prompts.py": PROMPTS_SRC,
        "tool_agent.py": TOOL_AGENT_SRC,
        "python_tool_sandbox.py": PROMPTS_SRC.parent / "python_tool_sandbox.py",
    }[name]


_MODEL_FACING_RE = re.compile(r'["\']([^"\']{25,})["\']')


def _is_model_facing(line: str) -> bool:
    """Heuristic: the hit sits inside a prose string literal, not an identifier."""
    for m in _MODEL_FACING_RE.finditer(line):
        body = m.group(1)
        if len(body.split()) >= 4 and _SWEEP_RE.search(body):
            return True
    return False


def build_manifest() -> dict[str, object]:
    entries = []
    problems = []
    for spec in SURFACE:
        path = _src(str(spec["file"]))
        lines = path.read_text(encoding="utf-8").splitlines()
        idx = int(spec["line"]) - 1
        actual = lines[idx] if 0 <= idx < len(lines) else "<OUT OF RANGE>"
        baseline = spec["baseline"]
        if baseline is None:            # KEEP entries capture the fork verbatim
            baseline = actual
            located = actual != "<OUT OF RANGE>"
        else:
            located = actual == baseline
            if not located:
                try:                    # tolerate line drift: find the exact text
                    idx = lines.index(str(baseline))
                    located = True
                except ValueError:
                    problems.append(
                        f"{spec['id']}: baseline text not found in {path.name}")
        entries.append({
            **{k: v for k, v in spec.items() if k not in ("baseline", "p1")},
            "resolved_line": idx + 1,
            "baseline": baseline,
            "baseline_sha256": _sha(str(baseline)),
            "p1": spec["p1"],
            "p1_sha256": _sha(str(spec["p1"])) if spec["p1"] is not None else None,
            "located": located,
        })
    manifest = {
        "manifest_version": "p1-patch-surface-1.0.0",
        "arm": "P1 persistent sandbox namespace",
        "rule": (
            "The screen MUST apply every entry with action=PATCH and MUST leave "
            "every entry with action=KEEP byte-identical. The patched tree is "
            "byte-audited against this manifest before the dataset push, and the "
            "manifest sha256 is cited in the S2 prereg. Any string matching the "
            "sweep patterns that is not listed here is an UNDECLARED "
            "contradiction and blocks the push."
        ),
        "k4_dependency": (
            "namespace_reuse_rate (K4) is only interpretable as a substrate "
            "reading when this manifest audits clean; otherwise a low reading "
            "measures instruction conflict."
        ),
        "files": {
            name: {
                "path": str(_src(name).relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha(_src(name).read_text(encoding="utf-8")),
            }
            for name in sorted({str(s["file"]) for s in SURFACE})
        },
        "entries": entries,
        "sweep_patterns": _SWEEP_PATTERNS,
        "problems": problems,
    }
    return manifest


def audit(manifest: dict[str, object] | None = None) -> tuple[bool, list[str]]:
    man = manifest or json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    issues: list[str] = []
    for entry in man["entries"]:  # type: ignore[index]
        path = _src(str(entry["file"]))
        lines = path.read_text(encoding="utf-8").splitlines()
        idx = int(entry["resolved_line"]) - 1
        actual = lines[idx] if 0 <= idx < len(lines) else None
        if actual is None or _sha(actual) != entry["baseline_sha256"]:
            issues.append(
                f"{entry['id']} DRIFT: {entry['file']}:{entry['resolved_line']} "
                f"no longer matches declared baseline sha"
            )
    return (not issues), issues


def sweep_unlisted(manifest: dict[str, object] | None = None
                   ) -> tuple[list[str], list[str]]:
    """Return (blocking_model_facing_hits, informational_code_hits)."""
    man = manifest or json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    declared = {(str(e["file"]), int(e["resolved_line"])) for e in man["entries"]}  # type: ignore[index]
    blocking: list[str] = []
    info: list[str] = []
    for path in sorted(AGENT_DIR.glob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if (path.name, i) in declared:
                continue
            if not _SWEEP_RE.search(line):
                continue
            rec = f"{path.name}:{i}: {line.strip()[:140]}"
            (blocking if _is_model_facing(line) else info).append(rec)
    return blocking, info


def verify_patched(patched_root: Path, manifest: dict[str, object] | None = None
                   ) -> tuple[bool, list[str]]:
    """Byte-audit a patched copy of the agent package against the manifest."""
    man = manifest or json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    issues: list[str] = []
    for entry in man["entries"]:  # type: ignore[index]
        path = patched_root / str(entry["file"])
        if not path.exists():
            issues.append(f"{entry['id']}: {path} missing")
            continue
        text = path.read_text(encoding="utf-8")
        want = str(entry["p1"] if entry["action"] == "PATCH" else entry["baseline"])
        forbidden = str(entry["baseline"]) if entry["action"] == "PATCH" else None
        if want not in text:
            issues.append(f"{entry['id']}: required text absent from patched {entry['file']}")
        if forbidden is not None and forbidden in text:
            issues.append(f"{entry['id']}: baseline contradiction STILL PRESENT in {entry['file']}")
    return (not issues), issues


def apply_patch(text: str, file_name: str, manifest: dict[str, object] | None = None) -> str:
    """Return `text` with every PATCH entry for `file_name` applied (exact-string)."""
    man = manifest or json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    out = text
    for entry in man["entries"]:  # type: ignore[index]
        if entry["file"] != file_name or entry["action"] != "PATCH":
            continue
        base = str(entry["baseline"])
        if base not in out:
            raise ValueError(f"{entry['id']}: baseline text not present, cannot patch")
        out = out.replace(base, str(entry["p1"]), 1)
    return out


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "audit"
    if cmd == "build":
        man = build_manifest()
        MANIFEST_PATH.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST_PATH}")
        print(f"entries={len(man['entries'])} problems={man['problems']}")
        for e in man["entries"]:
            print(f"  {e['id']} {e['file']}:{e['resolved_line']} "
                  f"[{e['channel']}/{e['contradiction']}/{e['severity']}] "
                  f"{e['action']} located={e['located']}")
        return 0 if not man["problems"] else 1
    if cmd == "audit":
        ok, issues = audit()
        print("MANIFEST AUDIT:", "CLEAN" if ok else "DRIFT")
        for i in issues:
            print("  ", i)
        return 0 if ok else 1
    if cmd == "sweep":
        blocking, info = sweep_unlisted()
        print(f"UNDECLARED model-facing contradictions (BLOCKING): {len(blocking)}")
        for h in blocking:
            print("   !", h)
        print(f"code-identifier hits (informational): {len(info)}")
        for h in info:
            print("    -", h)
        return 0 if not blocking else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
