"""Class-A scoring for the Phase-0c synthesis pilot.

Implements the ONE Class-A definition from learnings/winning_solution_FINAL.md:

  Class-A: held-out exact-match on 5-step OPEN-LOOP rollouts
           >= max(identity-frame, pure-lookup, lookup-with-identity-fallback) + 10 pp.
  Split:   temporal 70/30 per game; changed-frame stratum reported separately.
  Acceptance score = held-out exact-match (pp) - 2.0 * gzip-KB of full source
                     (data literals included).
  Train-vs-held-out gap = memorization flag.

Also computes the same one-step replay-exact metric as exec_wm/validate_sim.py
for comparability with the 22 offline opus sims.

Pod-side dependencies: numpy + stdlib only.
"""
from __future__ import annotations

import gzip
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

CLASS_A_MARGIN_PP = 10.0
LAMBDA_PP_PER_KB = 2.0
ROLLOUT_STEPS = 5
EXEC_TIMEOUT_S = 5.0
REPLAY_TIMEOUT_S = 120.0

# ---------------------------------------------------------------------------
# Sandboxed execution of candidate model source
# ---------------------------------------------------------------------------

_ALLOWED_ROOTS = {"numpy", "math", "typing", "collections", "itertools",
                  "functools", "copy", "__future__"}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in _ALLOWED_ROOTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"import of {name!r} is blocked in the pilot sandbox")


def _safe_builtins() -> Dict[str, Any]:
    import builtins as _b
    allowed = [
        "abs", "all", "any", "bool", "bytes", "callable", "chr", "dict",
        "divmod", "enumerate", "filter", "float", "frozenset", "getattr",
        "hasattr", "hash", "int", "isinstance", "issubclass", "iter", "len",
        "list", "map", "max", "min", "next", "object", "ord", "pow", "print",
        "range", "repr", "reversed", "round", "set", "setattr", "slice",
        "sorted", "str", "sum", "tuple", "type", "zip",
        "ArithmeticError", "AssertionError", "AttributeError", "BaseException",
        "Exception", "IndexError", "KeyError", "LookupError", "NameError",
        "NotImplementedError", "OverflowError", "RuntimeError", "StopIteration",
        "TypeError", "ValueError", "ZeroDivisionError", "True", "False", "None",
    ]
    out = {}
    for name in allowed:
        if hasattr(_b, name):
            out[name] = getattr(_b, name)
    out["__import__"] = _restricted_import
    out["__build_class__"] = _b.__build_class__
    out["__name__"] = "pilot_candidate"
    return out


def _run_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Tuple[bool, Any]:
    """Run fn in a daemon thread; (ok, result_or_error). Windows-safe."""
    box: Dict[str, Any] = {}

    def _target():
        try:
            box["result"] = fn()
            box["ok"] = True
        except BaseException as e:  # noqa: BLE001
            box["ok"] = False
            box["error"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        return False, f"timeout after {timeout_s}s"
    if box.get("ok"):
        return True, box.get("result")
    return False, box.get("error", "unknown sandbox error")


def compile_candidate(source: str) -> Tuple[Optional[Callable], Optional[str]]:
    """exec candidate source in a restricted namespace (numpy only).

    Returns (simulate_fn, error). simulate(state, action_id, x, y)
    -> (next_state, reward_class, done).
    """
    ns: Dict[str, Any] = {"__builtins__": _safe_builtins(), "np": np, "numpy": np}

    def _do():
        exec(compile(source, "<candidate_sim>", "exec"), ns)  # noqa: S102
        return True

    ok, err = _run_with_timeout(_do, EXEC_TIMEOUT_S)
    if not ok:
        return None, str(err)
    fn = ns.get("simulate")
    if not callable(fn):
        return None, "source did not define a callable simulate(state, action_id, x, y)"
    return fn, None


# ---------------------------------------------------------------------------
# Splits and windows
# ---------------------------------------------------------------------------

def temporal_split(tuples: List[dict], train_frac: float = 0.7):
    cut = int(len(tuples) * train_frac)
    return tuples[:cut], tuples[cut:]


def _grid(t: Any) -> np.ndarray:
    return np.asarray(t, dtype=np.int16)


def _key(state: np.ndarray, action_id: int, x: int, y: int) -> bytes:
    return state.astype(np.uint8).tobytes() + bytes([action_id & 0xFF, x & 0xFF, y & 0xFF])


def windows_5step(heldout: List[dict], steps: int = ROLLOUT_STEPS) -> List[List[dict]]:
    """Consecutive `steps`-length windows on the held-out segment.

    Requires step-contiguity and no done=True before the final transition.
    """
    wins = []
    for i in range(len(heldout) - steps + 1):
        w = heldout[i:i + steps]
        contiguous = all(w[j + 1]["step"] == w[j]["step"] + 1 for j in range(steps - 1))
        no_mid_done = not any(bool(t["done"]) for t in w[:-1])
        if contiguous and no_mid_done:
            wins.append(w)
    return wins


# ---------------------------------------------------------------------------
# Rollout engines (candidate + 3 baselines)
# ---------------------------------------------------------------------------

def _rollout_candidate(sim_fn: Callable, window: List[dict]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    state = _grid(window[0]["state_t"])
    for t in window:
        try:
            pred = sim_fn(state.tolist(), int(t["action_id"]), int(t["x"]), int(t["y"]))
            ns = _grid(pred[0])
            if ns.shape != state.shape:
                return None, f"bad shape {ns.shape}"
            state = ns
        except BaseException as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
    return state, None


def _rollout_identity(window: List[dict]) -> np.ndarray:
    return _grid(window[0]["state_t"])


def _build_lookup(train: List[dict]) -> Dict[bytes, np.ndarray]:
    lut: Dict[bytes, np.ndarray] = {}
    for t in train:
        s = _grid(t["state_t"])
        lut[_key(s, int(t["action_id"]), int(t["x"]), int(t["y"]))] = _grid(t["state_t1"])
    return lut


def _rollout_lookup(lut: Dict[bytes, np.ndarray], window: List[dict],
                    identity_fallback: bool) -> Optional[np.ndarray]:
    state = _grid(window[0]["state_t"])
    for t in window:
        k = _key(state, int(t["action_id"]), int(t["x"]), int(t["y"]))
        nxt = lut.get(k)
        if nxt is None:
            if identity_fallback:
                nxt = state  # predict no change
            else:
                return None  # pure lookup fails on unseen (state, action)
        state = nxt
    return state


def _score_windows(predict: Callable[[List[dict]], Optional[np.ndarray]],
                   windows: List[List[dict]]) -> Dict[str, Any]:
    n = len(windows)
    exact = 0
    ch_n = ch_exact = 0
    un_n = un_exact = 0
    for w in windows:
        truth = _grid(w[-1]["state_t1"])
        start = _grid(w[0]["state_t"])
        changed = not np.array_equal(truth, start)
        pred = predict(w)
        hit = pred is not None and np.array_equal(pred, truth)
        exact += int(hit)
        if changed:
            ch_n += 1
            ch_exact += int(hit)
        else:
            un_n += 1
            un_exact += int(hit)
    return {
        "n_windows": n,
        "exact_pct": 100.0 * exact / max(1, n),
        "changed_stratum": {"n": ch_n, "exact_pct": 100.0 * ch_exact / max(1, ch_n)},
        "unchanged_stratum": {"n": un_n, "exact_pct": 100.0 * un_exact / max(1, un_n)},
    }


# ---------------------------------------------------------------------------
# One-step replay metric (validate_sim.py-compatible)
# ---------------------------------------------------------------------------

def one_step_replay(sim_fn: Callable, tuples: List[dict]) -> Dict[str, Any]:
    n_total = n_exact = n_err = n_reward = n_done = 0
    pixel_sum = 0.0
    ch_n = ch_exact = 0
    mismatches: List[dict] = []
    for t in tuples:
        n_total += 1
        s_t = _grid(t["state_t"])
        truth = _grid(t["state_t1"])
        changed = not np.array_equal(s_t, truth)
        try:
            pred = sim_fn(s_t.tolist(), int(t["action_id"]), int(t["x"]), int(t["y"]))
            ns, rc, dn = pred[0], pred[1], pred[2]
            ns_arr = _grid(ns)
            if ns_arr.shape != truth.shape:
                raise ValueError(f"shape {ns_arr.shape}")
        except BaseException as e:  # noqa: BLE001
            n_err += 1
            mismatches.append({"step": t["step"], "error": f"{type(e).__name__}: {e}"})
            if changed:
                ch_n += 1
            continue
        exact = bool(np.array_equal(ns_arr, truth))
        n_exact += int(exact)
        pixel_sum += float((ns_arr == truth).mean())
        n_reward += int(int(rc) == int(t["reward_class"]))
        n_done += int(bool(dn) == bool(t["done"]))
        if changed:
            ch_n += 1
            ch_exact += int(exact)
        if not exact:
            diff = np.argwhere(ns_arr != truth)
            cells = [[int(r), int(c), int(ns_arr[r, c]), int(truth[r, c])]
                     for r, c in diff[:40]]
            mismatches.append({
                "step": t["step"], "action_id": t["action_id"],
                "x": t["x"], "y": t["y"],
                "n_wrong_cells": int(diff.shape[0]),
                "wrong_cells_pred_vs_truth": cells,
            })
    n_ok = max(1, n_total - n_err)
    return {
        "n": n_total,
        "errors": n_err,
        "state_exact_pct": 100.0 * n_exact / n_ok,
        "pixel_match_pct": 100.0 * pixel_sum / n_ok,
        "reward_acc_pct": 100.0 * n_reward / n_ok,
        "done_acc_pct": 100.0 * n_done / n_ok,
        "changed_stratum": {"n": ch_n, "exact_pct": 100.0 * ch_exact / max(1, ch_n)},
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Full Class-A evaluation
# ---------------------------------------------------------------------------

def gzip_kb(source: str) -> float:
    return len(gzip.compress(source.encode("utf-8"))) / 1024.0


def score_baselines(train: List[dict], heldout: List[dict]) -> Dict[str, Any]:
    windows = windows_5step(heldout)
    lut = _build_lookup(train)
    return {
        "identity_frame": _score_windows(lambda w: _rollout_identity(w), windows),
        "pure_lookup": _score_windows(lambda w: _rollout_lookup(lut, w, False), windows),
        "lookup_identity_fallback": _score_windows(lambda w: _rollout_lookup(lut, w, True), windows),
    }


def score_candidate(source: str, train: List[dict], heldout: List[dict],
                    baselines: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Full Class-A scorecard for one candidate model source string."""
    result: Dict[str, Any] = {"compile_error": None}
    sim_fn, err = compile_candidate(source)
    if sim_fn is None:
        result["compile_error"] = err
        result["class_a"] = False
        result["heldout_5step"] = None
        result["acceptance_score"] = None
        return result

    if baselines is None:
        baselines = score_baselines(train, heldout)
    result["baselines_5step"] = baselines

    windows = windows_5step(heldout)

    def _do_all():
        cand = _score_windows(lambda w: _rollout_candidate(sim_fn, w)[0], windows)
        replay_heldout = one_step_replay(sim_fn, heldout)
        replay_train = one_step_replay(sim_fn, train)
        return cand, replay_heldout, replay_train

    ok, out = _run_with_timeout(_do_all, REPLAY_TIMEOUT_S)
    if not ok:
        result["compile_error"] = f"replay failed: {out}"
        result["class_a"] = False
        result["heldout_5step"] = None
        result["acceptance_score"] = None
        return result

    cand, replay_heldout, replay_train = out
    # trim mismatch detail from the stored scorecard (kept for reports elsewhere)
    replay_heldout = dict(replay_heldout)
    replay_train = dict(replay_train)
    replay_heldout.pop("mismatches", None)
    replay_train.pop("mismatches", None)

    best_baseline = max(v["exact_pct"] for v in baselines.values())
    kb = gzip_kb(source)
    result.update({
        "heldout_5step": cand,
        "best_baseline_pct": best_baseline,
        "class_a": bool(cand["exact_pct"] >= best_baseline + CLASS_A_MARGIN_PP),
        "class_a_margin_pp": cand["exact_pct"] - best_baseline,
        "gzip_kb": kb,
        "acceptance_score": cand["exact_pct"] - LAMBDA_PP_PER_KB * kb,
        "one_step_replay_heldout": replay_heldout,
        "one_step_replay_train": replay_train,
        "memorization_gap_pp": (replay_train["state_exact_pct"]
                                - replay_heldout["state_exact_pct"]),
    })
    return result


if __name__ == "__main__":
    # smoke: score the identity function against a game's observations
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--obs-dir", default=None)
    args = ap.parse_args()
    here = Path(__file__).resolve()
    obs_dir = Path(args.obs_dir) if args.obs_dir else None
    if obs_dir is None:
        for cand in (here.parents[1] / "exec_wm" / "observations",
                     here.parents[2] / "exec_wm" / "observations"):
            if cand.exists():
                obs_dir = cand
                break
    data = json.loads((obs_dir / f"{args.game}.json").read_text())
    train, heldout = temporal_split(data["tuples"])
    src = ("def simulate(state, action_id, x, y):\n"
           "    return state, 0, False\n")
    print(json.dumps(score_candidate(src, train, heldout), indent=2))
