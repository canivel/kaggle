"""Phase-0c synthesis pilot orchestrator (winning_solution_FINAL.md, Phase 0c).

Tests the load-bearing premise: can Qwen3-27B (local vLLM, OpenAI-compatible
endpoint) synthesize Class-A executable transition models for ARC-AGI-3 games
under the Phase-2 runtime context regime?

Per (game, scaffold) it runs TWO regimes:
  capped6k    - DECIDING arm: transition evidence capped at ~6k tokens under
                the runtime's own selection policy (changed-frame transitions
                first, then most recent; oldest evicted first) + 1k
                verify-report slot.
  uncapped32k - unconstrained-history arm (<=32k), reported separately as an
                UPPER BOUND ONLY; it decides nothing.

Refactor loop: up to 4 iterations of write -> sandboxed exec -> train replay
-> mismatch report -> revise. Held-out data never enters the loop; it is only
used for scoring/logging.

Resumable: one JSON per (game, scaffold) written immediately; existing files
are skipped on restart.

Pod-side dependencies: numpy + stdlib + httpx only. LLM calls are raw httpx
against http://127.0.0.1:8000/v1 (never litellm, never the openai SDK).

Usage:
  python run_pilot.py                       # full 10 games x 3 scaffolds
  python run_pilot.py --mock --games bp35,r11l --scaffolds freeform
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402
import scoring  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered game set (10 games), chosen for difficulty diversity from
# exec_wm/scale_summary.md (opus hand-built sim exact-match on 200 tuples):
#   easy / perfect-sim  : ft09 (100%), lp85 (100%), ls20 (100%), tu93 (100%)
#   mid                 : re86 (90.5%), cn04 (77.5%), cd82 (60.5%)
#   hard                : sk48 (38%), r11l (23%)
#   unbenchmarked       : bp35 (has observations + a hand sim, absent from the
#                         24-game summary table; kept as an out-of-table probe)
# ---------------------------------------------------------------------------
GAMES = ["ft09", "lp85", "ls20", "tu93", "re86", "cn04", "cd82", "sk48", "r11l", "bp35"]
SCAFFOLDS = ["skeleton", "freeform", "diff"]
REGIMES = ["capped6k", "uncapped32k"]

EVIDENCE_TOKEN_CAP = {"capped6k": 6000, "uncapped32k": 24000}
MISMATCH_REPORT_TOKEN_CAP = 1000
MAX_ITERS = 4
MAX_GEN_TOKENS = 6144            # planning headroom; p90 is MEASURED by this pilot
PROMPT_TOKEN_EVICT_AT = 26000    # evict oldest loop turns past this (history slot)
TEMPLATE_CHAR_CAP = 10000        # ~2.5k tokens, matches Phase-2 skeleton slot
TRAIN_FRAC = 0.7

DATA_CAVEAT = (
    "Observation histories are LEVEL-0 RANDOM-EXPLORATION recordings "
    "(exec_wm/collect_observations.py), 200 tuples/game - NOT duck agent "
    "trajectories. Pre-registered caveat: transition coverage reflects random "
    "policy state visitation only."
)


def est_tokens(text: str) -> int:
    return len(text) // 4 + 1


# ---------------------------------------------------------------------------
# Data root resolution (repo layout AND pod bundle layout)
# ---------------------------------------------------------------------------

def find_data_root(cli_root: Optional[str]) -> Path:
    here = Path(__file__).resolve().parent
    candidates = []
    if cli_root:
        candidates.append(Path(cli_root))
    candidates += [here.parent, here.parents[1] if len(here.parents) > 1 else here]
    for c in candidates:
        if (c / "exec_wm" / "observations").exists():
            return c
    raise FileNotFoundError(
        "could not find exec_wm/observations near " + str(here)
        + " ; pass --data-root")


def load_game(root: Path, game: str) -> dict:
    return json.loads((root / "exec_wm" / "observations" / f"{game}.json")
                      .read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Evidence encoding
# ---------------------------------------------------------------------------

def rle_frame(state: List[List[int]]) -> str:
    rows = []
    for r, row in enumerate(state):
        parts = []
        cur, cnt = row[0], 1
        for v in row[1:]:
            if v == cur:
                cnt += 1
            else:
                parts.append(f"{cur}*{cnt}")
                cur, cnt = v, 1
        parts.append(f"{cur}*{cnt}")
        rows.append(f"r{r:02d}: " + " ".join(parts))
    return "\n".join(rows)


def encode_transition(t: dict, max_cells: int = 60) -> str:
    a = np.asarray(t["state_t"], dtype=np.int16)
    b = np.asarray(t["state_t1"], dtype=np.int16)
    head = f"#{t['step']} a{t['action_id']}"
    if int(t["action_id"]) == 6:
        head += f"@({t['x']},{t['y']})"
    head += f" rc{t['reward_class']}"
    if t.get("done"):
        head += "|done"
    diff = np.argwhere(a != b)
    if diff.shape[0] == 0:
        return head + " :: no-change"
    cells = [f"{r},{c}:{a[r, c]}>{b[r, c]}" for r, c in diff[:max_cells]]
    tail = f" (+{diff.shape[0] - max_cells} more)" if diff.shape[0] > max_cells else ""
    return head + " :: " + "; ".join(cells) + tail


def select_evidence(train: List[dict], token_cap: int) -> Tuple[str, int]:
    """Runtime eviction policy: changed-frame first, then most-recent
    unchanged; within each class most-recent first; oldest evicted first.
    Returns (evidence_text, n_transitions_included). Presented chronologically.
    """
    changed = [t for t in train
               if t["state_t"] != t["state_t1"]]
    unchanged = [t for t in train if t["state_t"] == t["state_t1"]]
    order = list(reversed(changed)) + list(reversed(unchanged))
    chosen: List[dict] = []
    budget = token_cap
    for t in order:
        line = encode_transition(t)
        cost = est_tokens(line) + 1
        if cost > budget:
            continue  # oldest / lowest-priority evicted first
        budget -= cost
        chosen.append(t)
    chosen.sort(key=lambda t: t["step"])
    lines = [encode_transition(t) for t in chosen]
    return "\n".join(lines), len(chosen)


# ---------------------------------------------------------------------------
# LOGO template selection (diff scaffold): nearest sim from OTHER games
# ---------------------------------------------------------------------------

def _action_set(root: Path, game: str) -> set:
    try:
        return set(load_game(root, game)["available_actions"])
    except Exception:
        return set()


def _change_rate(root: Path, game: str) -> float:
    try:
        d = load_game(root, game)
        s = d.get("summary", {})
        return s.get("n_state_changes", 0) / max(1, s.get("n_tuples", 1))
    except Exception:
        return 0.0


def list_template_games(root: Path, exclude: str) -> List[str]:
    sims = root / "exec_wm" / "sims"
    out = []
    for p in sorted(sims.glob("*_sim.py")):
        g = p.name[:-len("_sim.py")]
        if g.startswith("_") or g == exclude:
            continue
        if (root / "exec_wm" / "observations" / f"{g}.json").exists():
            out.append(g)
    return out


def nearest_template(root: Path, game: str) -> Tuple[str, str]:
    """LOGO: pick the template sim most similar to `game`, EXCLUDING `game`.
    Similarity: Jaccard of available-action sets, tie-broken by closest
    changed-frame rate. Fully mechanical, no human choice per game.
    """
    mine_a = _action_set(root, game)
    mine_c = _change_rate(root, game)
    best, best_key = None, None
    for g in list_template_games(root, exclude=game):
        a = _action_set(root, g)
        jac = len(mine_a & a) / max(1, len(mine_a | a))
        key = (jac, -abs(_change_rate(root, g) - mine_c), g)
        if best_key is None or key > best_key:
            best, best_key = g, key
    assert best is not None, "no template games found"
    src = (root / "exec_wm" / "sims" / f"{best}_sim.py").read_text(encoding="utf-8")
    if len(src) > TEMPLATE_CHAR_CAP:
        src = src[:TEMPLATE_CHAR_CAP] + "\n# ... (template truncated)\n"
    return best, src


# ---------------------------------------------------------------------------
# LLM client (raw httpx; mock stub for CPU-only end-to-end testing)
# ---------------------------------------------------------------------------

MOCK_RESPONSE = (
    "Based on the evidence I will start with the identity transition model.\n\n"
    "```python\n"
    "def simulate(state, action_id, x, y):\n"
    "    # degenerate mock candidate: frame never changes\n"
    "    return state, 0, False\n"
    "```\n"
)


class LLMClient:
    def __init__(self, endpoint: str, model: Optional[str], mock: bool,
                 temperature: float = 0.2, seed: int = 0):
        self.endpoint = endpoint.rstrip("/")
        self.mock = mock
        self.temperature = temperature
        self.seed = seed
        self.model = model
        if not mock:
            import httpx  # lazy: not needed for --mock
            self._httpx = httpx
            if not self.model:
                r = httpx.get(self.endpoint + "/models", timeout=60)
                r.raise_for_status()
                self.model = r.json()["data"][0]["id"]
                print(f"[llm] auto-detected model: {self.model}")

    def chat(self, messages: List[dict], max_tokens: int = MAX_GEN_TOKENS) -> dict:
        """Returns {text, prompt_tokens, completion_tokens, finish_reason}."""
        if self.mock:
            ptok = sum(est_tokens(m["content"]) for m in messages)
            return {"text": MOCK_RESPONSE, "prompt_tokens": ptok,
                    "completion_tokens": est_tokens(MOCK_RESPONSE),
                    "finish_reason": "stop"}
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": self.temperature,
            "seed": self.seed,
        }
        last_err = None
        for attempt in range(3):
            try:
                r = self._httpx.post(self.endpoint + "/chat/completions",
                                     json=body, timeout=900)
                r.raise_for_status()
                d = r.json()
                ch = d["choices"][0]
                msg = ch["message"]
                text = msg.get("content") or ""
                # some vLLM builds put chain-of-thought in reasoning_content;
                # completion_tokens already includes it
                usage = d.get("usage", {})
                return {"text": text,
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "finish_reason": ch.get("finish_reason", "unknown")}
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = 15 * (attempt + 1)
                print(f"[llm] attempt {attempt + 1} failed ({e}); retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after 3 attempts: {last_err}")


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> Optional[str]:
    blocks = CODE_BLOCK_RE.findall(text)
    for b in reversed(blocks):
        if "def simulate" in b:
            return b.strip() + "\n"
    if "def simulate" in text:
        return text.strip() + "\n"
    return None


# ---------------------------------------------------------------------------
# Mismatch report (verify slot, <=1k tokens)
# ---------------------------------------------------------------------------

def build_mismatch_report(replay: dict, token_cap: int = MISMATCH_REPORT_TOKEN_CAP) -> str:
    lines: List[str] = []
    budget = token_cap
    for m in replay.get("mismatches", []):
        if "error" in m:
            line = f"#{m['step']} ERROR {m['error']}"
        else:
            cells = "; ".join(f"{r},{c}:{p}>{t}"
                              for r, c, p, t in m["wrong_cells_pred_vs_truth"][:12])
            extra = m["n_wrong_cells"] - min(12, m["n_wrong_cells"])
            tail = f" (+{extra} more cells)" if extra > 0 else ""
            line = (f"#{m['step']} a{m['action_id']}"
                    + (f"@({m['x']},{m['y']})" if m.get("action_id") == 6 else "")
                    + f" -> {cells}{tail}")
        cost = est_tokens(line) + 1
        if cost > budget:
            lines.append("(report truncated at 1k tokens)")
            break
        budget -= cost
        lines.append(line)
    return "\n".join(lines) if lines else "(no mismatches recorded)"


# ---------------------------------------------------------------------------
# One synthesis arm: (game, scaffold, regime) with refactor loop
# ---------------------------------------------------------------------------

def run_arm(client: LLMClient, game: str, scaffold: str, regime: str,
            data: dict, train: List[dict], heldout: List[dict],
            baselines: dict, root: Path, max_iters: int) -> dict:
    evidence, n_ev = select_evidence(train, EVIDENCE_TOKEN_CAP[regime])
    tmpl_game, tmpl_src = ("", "")
    if scaffold == "diff":
        tmpl_game, tmpl_src = nearest_template(root, game)

    user0 = prompts.TEMPLATES[scaffold].format(
        game_id=game,
        actions=data["available_actions"],
        reference_frame=rle_frame(train[0]["state_t"]),
        n_evidence=n_ev,
        n_train=len(train),
        evidence=evidence,
        template_game=tmpl_game,
        template_source=tmpl_src,
    )
    messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": user0}]

    iters: List[dict] = []
    total_prompt = total_completion = 0
    tokens_to_first_class_a: Optional[int] = None
    best = {"iter": None, "train_exact": -1.0, "source": None}

    for it in range(1, max_iters + 1):
        # evict oldest loop turns (history slot evicted first) if prompt too big
        while (sum(est_tokens(m["content"]) for m in messages) > PROMPT_TOKEN_EVICT_AT
               and len(messages) > 4):
            del messages[2:4]
        resp = client.chat(messages)
        total_prompt += resp["prompt_tokens"]
        total_completion += resp["completion_tokens"]
        messages.append({"role": "assistant", "content": resp["text"]})

        rec: Dict[str, Any] = {
            "iter": it,
            "prompt_tokens": resp["prompt_tokens"],
            "completion_tokens": resp["completion_tokens"],
            "finish_reason": resp["finish_reason"],
            "truncated": resp["finish_reason"] == "length",
        }
        source = extract_code(resp["text"])
        if source is None:
            rec["compile_error"] = "no simulate() code block in response"
            rec["train_exact_pct"] = None
            iters.append(rec)
            messages.append({"role": "user", "content":
                             "Your reply contained no code block defining "
                             "simulate(state, action_id, x, y). Reply with one "
                             "Python code block only."})
            continue
        rec["source_chars"] = len(source)
        rec["gzip_kb"] = scoring.gzip_kb(source)

        sim_fn, err = scoring.compile_candidate(source)
        if sim_fn is None:
            rec["compile_error"] = err
            rec["train_exact_pct"] = None
            iters.append(rec)
            messages.append({"role": "user", "content":
                             f"Your code failed to load: {err}\nReply with the "
                             "full corrected code block only."})
            continue

        train_replay = scoring.one_step_replay(sim_fn, train)
        rec["compile_error"] = None
        rec["train_exact_pct"] = train_replay["state_exact_pct"]
        rec["train_errors"] = train_replay["errors"]

        # held-out scorecard per iteration: LOGGING ONLY, never fed back
        card = scoring.score_candidate(source, train, heldout, baselines=baselines)
        rec["heldout_5step_exact_pct"] = (card["heldout_5step"]["exact_pct"]
                                          if card["heldout_5step"] else None)
        rec["class_a"] = card["class_a"]
        if card["class_a"] and tokens_to_first_class_a is None:
            tokens_to_first_class_a = total_prompt + total_completion
        iters.append(rec)

        if train_replay["state_exact_pct"] > best["train_exact"]:
            best = {"iter": it, "train_exact": train_replay["state_exact_pct"],
                    "source": source}

        if train_replay["state_exact_pct"] >= 100.0 or it == max_iters:
            break
        report = build_mismatch_report(train_replay)
        messages.append({"role": "user", "content": prompts.REFINE_TEMPLATE.format(
            train_exact_pct=train_replay["state_exact_pct"],
            n_mismatch=len([m for m in train_replay["mismatches"] if "error" not in m]),
            n_error=train_replay["errors"],
            mismatch_report=report,
        )})

    final_card = (scoring.score_candidate(best["source"], train, heldout,
                                          baselines=baselines)
                  if best["source"] else
                  {"class_a": False, "compile_error": "no runnable candidate",
                   "heldout_5step": None, "acceptance_score": None,
                   "baselines_5step": baselines})

    return {
        "regime": regime,
        "evidence_transitions": n_ev,
        "evidence_token_cap": EVIDENCE_TOKEN_CAP[regime],
        "template_game": tmpl_game or None,
        "iterations": iters,
        "best_iter": best["iter"],
        "best_source": best["source"],
        "final": final_card,
        "tokens": {
            "prompt_total": total_prompt,
            "completion_total": total_completion,
            "total": total_prompt + total_completion,
            "tokens_to_first_class_a": tokens_to_first_class_a,
            "completion_per_call": [r["completion_tokens"] for r in iters],
        },
    }


# ---------------------------------------------------------------------------
# Summary + LOGO scaffold selection
# ---------------------------------------------------------------------------

def _pctile(vals: List[int], q: float) -> Optional[float]:
    return float(np.percentile(vals, q)) if vals else None


def build_summary(out_dir: Path, games: List[str], scaffolds: List[str]) -> dict:
    results: Dict[Tuple[str, str], dict] = {}
    for g in games:
        for s in scaffolds:
            p = out_dir / f"{g}__{s}.json"
            if p.exists():
                results[(g, s)] = json.loads(p.read_text(encoding="utf-8"))

    def acc(g: str, s: str, regime: str) -> Optional[float]:
        r = results.get((g, s))
        if not r or regime not in r["regimes"]:
            return None
        return r["regimes"][regime]["final"].get("acceptance_score")

    # LOGO scaffold selection on the DECIDING (capped6k) regime:
    # for each game, the selected scaffold maximizes mean acceptance score
    # over the OTHER games.
    logo: Dict[str, Any] = {}
    for g in games:
        best_s, best_v = None, None
        for s in scaffolds:
            vals = [acc(og, s, "capped6k") for og in games if og != g]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            v = float(np.mean(vals))
            if best_v is None or v > best_v:
                best_s, best_v = s, v
        entry = {"selected_scaffold": best_s,
                 "selection_mean_acceptance_other_games": best_v}
        if best_s and (g, best_s) in results:
            fin = results[(g, best_s)]["regimes"].get("capped6k", {}).get("final", {})
            entry["class_a_capped6k"] = bool(fin.get("class_a"))
            entry["acceptance_capped6k"] = fin.get("acceptance_score")
            tok = results[(g, best_s)]["regimes"].get("capped6k", {}).get("tokens", {})
            entry["tokens_to_first_class_a"] = tok.get("tokens_to_first_class_a")
        logo[g] = entry

    gen_lengths: Dict[str, Dict[str, Any]] = {}
    truncation: Dict[str, Any] = {}
    for s in scaffolds:
        lens, ncalls, ntrunc = [], 0, 0
        for g in games:
            r = results.get((g, s))
            if not r:
                continue
            for regime in r["regimes"].values():
                for it in regime["iterations"]:
                    lens.append(it["completion_tokens"])
                    ncalls += 1
                    ntrunc += int(it.get("truncated", False))
        gen_lengths[s] = {"n_calls": ncalls, "p50": _pctile(lens, 50),
                          "p90": _pctile(lens, 90), "max": max(lens) if lens else None}
        truncation[s] = {"rate": ntrunc / max(1, ncalls), "n_truncated": ntrunc}

    n_class_a = sum(1 for g in games
                    if logo.get(g, {}).get("class_a_capped6k") is True)
    return {
        "data_caveat": DATA_CAVEAT,
        "class_a_definition": ("heldout 5-step open-loop exact >= "
                               "max(identity, pure-lookup, lookup+identity) + 10pp"),
        "acceptance_rule": "heldout_pp - 2.0 * gzip_KB(source)",
        "n_games_class_a_under_logo_capped6k": n_class_a,
        "phase2_entry_gate": ">=4/10 Class-A on 6k-regime LOGO numbers",
        "logo_per_game": logo,
        "generation_lengths_per_scaffold": gen_lengths,
        "truncation_per_scaffold": truncation,
        "acceptance_matrix_capped6k": {
            g: {s: acc(g, s, "capped6k") for s in scaffolds} for g in games},
        "acceptance_matrix_uncapped32k_upper_bound_only": {
            g: {s: acc(g, s, "uncapped32k") for s in scaffolds} for g in games},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-0c synthesis pilot")
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--scaffolds", default=",".join(SCAFFOLDS))
    ap.add_argument("--regimes", default=",".join(REGIMES),
                    help="capped6k,uncapped32k")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=None,
                    help="model id; auto-detected from /v1/models if omitted")
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--mock", action="store_true",
                    help="stub the LLM with a canned identity-model response")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if a result file exists")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    scaffolds = [s.strip() for s in args.scaffolds.split(",") if s.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    for s in scaffolds:
        assert s in SCAFFOLDS, f"unknown scaffold {s}"
    for r in regimes:
        assert r in REGIMES, f"unknown regime {r}"

    root = find_data_root(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent / ("results_mock" if args.mock else "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        client = LLMClient(args.endpoint, args.model, args.mock,
                           temperature=args.temperature, seed=args.seed)
        for game in games:
            data = load_game(root, game)
            train, heldout = scoring.temporal_split(data["tuples"], TRAIN_FRAC)
            baselines = scoring.score_baselines(train, heldout)
            for scaffold in scaffolds:
                out_path = out_dir / f"{game}__{scaffold}.json"
                if out_path.exists() and not args.force:
                    print(f"[skip] {game}/{scaffold} (exists)")
                    continue
                t0 = time.time()
                print(f"[run ] {game}/{scaffold} regimes={regimes}")
                rec = {
                    "game": game,
                    "scaffold": scaffold,
                    "mock": args.mock,
                    "data_caveat": DATA_CAVEAT,
                    "split": {"train": len(train), "heldout": len(heldout),
                              "train_frac": TRAIN_FRAC, "kind": "temporal"},
                    "baselines_5step": baselines,
                    "regimes": {},
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                for regime in regimes:
                    rec["regimes"][regime] = run_arm(
                        client, game, scaffold, regime, data, train, heldout,
                        baselines, root, args.max_iters)
                rec["wall_s"] = round(time.time() - t0, 1)
                out_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
                fin = {r: rec["regimes"][r]["final"].get("class_a")
                       for r in regimes}
                print(f"[done] {game}/{scaffold} class_a={fin} "
                      f"wall={rec['wall_s']}s")

    summary = build_summary(out_dir, games, scaffolds)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
