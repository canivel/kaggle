"""OBJ-I — Schema fixed-resolver verification (LOCAL, $0, no fitting).

Panel R17 OBJ-I / prog-synthesis R3. A *fixed* external hypothesis (sourced from
Schema-harness's released, backtest-certified world models on our EXACT engine
versions) is verified against ALL of our streams per game with NO fitting and NO
selection budget -- legitimate under C6 because the hypothesis is fully specified
before touching our data.

The three externally-sourced, fully-specified laws (extracted verbatim from the
Schema world_model_v5.py snapshots; exact file+line cited in report.md):

  wa30  "mod-rate"        : visible move-bar filled = (mult*n + off)//D with a
                            PER-LEVEL (D, off, mult) table (world_model_v5.py
                            lines 98-111). The hidden sub-tick phase that
                            disambiguates 'does the next action advance the bar'
                            is  n mod D(level).  Fixed key = phase = n % D(level).
                            n = actions-since-level-start (ALL actions tick; the
                            bar is a pure move counter in this box-push game).

  ka59  "parity-inverted" : bar zeros = round(64*n/budget) (world_model_v5.py
                            line 31 / _bar_for line 51); the FUSE ticks one unit
                            per MOVE and *clicks (ACTION6) do NOT tick it*
                            (docstring lines 20-22). The hidden phase is the
                            move parity with clicks excluded.  Fixed key =
                            move_count % 2  where move_count counts only
                            non-click, non-RESET actions since level start.

  tr87  "floor(n/2)"      : row 63 budget bar = floor(n_actions / 2) cells
                            (world_model_v5.py line 19); K = 2 on levels 0-4,
                            K = 4 on level 5 (lines 262-265). The hidden sub-tick
                            phase is n mod K.  Fixed key = n % K(level),
                            n = actions-since-level-start (ALL actions tick).

Method: reuse the sealed audit's transition extraction + determinism +
Wilson-LB computation VERBATIM (import from scripts/latent_state_audit.py) so
the numbers are directly comparable to the sealed certificate bar. We only
supply the fixed key function; we never fit or select it. Streams = exactly the
audit's benchmark-engine-version stream set (holdout_report.json), reproduced by
the same discover/exclude rule; the NEW post-report sentinel_eval_v1 stream and
the *_tool_sentinel_events.jsonl sidecar are excluded to match the report, then
re-included in an appendix sensitivity (more visits only helps a fixed law).

Bar: sealed certificate requires det >= 0.99 AND Wilson 95% LB >= 0.95 (same as
runs/latent_state_audit/holdout_report.json thresholds). No train/test split:
a fixed hypothesis consumes no selection budget, so the certificate is the
pooled augmented determinism over ALL streams.

Stdlib-only, CPU, offline, $0.  uv run python runs/schema_fixed_resolver/verify_fixed_resolvers.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

# reuse the SEALED audit machinery verbatim (same det + Wilson used for the bar)
from latent_state_audit import (  # noqa: E402
    board_digest, alias_stats, wilson_lb, HOLDOUT_DET, HOLDOUT_WILSON_LB,
)

OUT = ROOT / "runs" / "schema_fixed_resolver"

# ---------------------------------------------------------------------------
# stream discovery: reproduce the holdout report's benchmark-engine-version set
# ---------------------------------------------------------------------------
# benchmark engine versions per game (from holdout_report.json), and the
# post-report stream to hold out of the primary run (re-added in the appendix).
BENCH_VERSION = {"wa30": "wa30-ee6fef47", "ka59": "ka59-38d34dbb",
                 "tr87": "tr87-cd924810"}
POST_REPORT_PULLS = {"sentinel_eval_v1"}  # landed after holdout_report.json


def discover(game, include_post_report=False):
    """-> list of (stream_id, filepath) for the benchmark engine version only.

    Excludes: (a) minority/drift engine versions (different guid),
              (b) *_tool_sentinel_events.jsonl sidecars (not game traces),
              (c) sentinel_eval_v1 unless include_post_report (postdates report).
    """
    ev = BENCH_VERSION[game]
    pats = [str(ROOT / "runs" / "kernel_pulls" / "*" / "artifacts" / f"{ev}_*_events.jsonl"),
            str(ROOT / "runs" / "phase1_ab" / "*" / "artifacts" / f"{ev}_*_events.jsonl")]
    out = []
    for fp in sorted(set(f for p in pats for f in glob.glob(p))):
        name = Path(fp).name
        if name.endswith("_tool_sentinel_events.jsonl"):
            continue
        pull = Path(fp).parents[1].name
        if pull in POST_REPORT_PULLS and not include_post_report:
            continue
        out.append((pull, fp))
    return out


# ---------------------------------------------------------------------------
# fixed-law transition extraction (Schema-faithful, per-level counter)
# ---------------------------------------------------------------------------
# wa30 per-level move-bar divisor D (Schema world_model_v5.py _BAR, level 0-based)
WA30_BAR_D = {0: 3, 1: 1, 2: 3, 3: 3, 4: 2, 5: 7, 6: 2, 7: 7, 8: 1}
WA30_DEFAULT_D = 2   # Schema _bar_params default for unknown level
# tr87 K per level (Schema world_model_v5.py lines 262-265): K=2 lv0-4, K=4 lv5
def tr87_K(level0):
    return 2 if (level0 is None or level0 < 5) else 4


def harness_level_to_schema(level):
    """Our events carry 1-based display level; Schema CURRENT_LEVEL is 0-based."""
    if level is None:
        return None
    return level - 1


def extract_fixed(events, stream_id, game):
    """Walk one trace -> transition dicts carrying the FIXED Schema key `phase`.

    Counters reset at each RESET and at each LEVEL change (Schema `n`/fuse are
    per-level). `prev`/`act`/`out` identical to the sealed audit so the pooled
    determinism is directly comparable.
    """
    trans = []
    prev = None
    prev_lvl = None
    n = 0            # actions-since-level-start (ALL actions), Schema `n`
    moves = 0        # non-click, non-RESET moves since level start (ka59 fuse)
    for ev in events:
        et = ev.get("type")
        if et == "initial":
            prev = board_digest(ev["board"])
            prev_lvl = ev.get("level")
            n = 0
            moves = 0
            continue
        if et != "action":
            continue
        if "board" not in ev or prev is None:
            continue
        lvl = prev_lvl  # PRE-action level (key semantics match audit: level is pre-action)
        out = board_digest(ev["board"])
        ak = ev.get("action_display") or ev.get("action_name") or "<unk>"
        aname = ev.get("action_name") or ""
        is_reset = (aname == "RESET" or ak == "RESET")
        is_click = (aname == "ACTION6")

        # fixed phase per game (computed from PRE-action counter state)
        s_lvl0 = harness_level_to_schema(lvl)
        if game == "wa30":
            D = WA30_BAR_D.get(s_lvl0, WA30_DEFAULT_D)
            phase = n % D
        elif game == "tr87":
            K = tr87_K(s_lvl0)
            phase = n % K
        elif game == "ka59":
            phase = moves % 2
        else:
            raise ValueError(game)

        trans.append({
            "stream": stream_id, "prev": prev, "act": ak, "out": out,
            "phase": phase, "lvl": lvl,
        })

        # advance counters
        cur_lvl = ev.get("level")
        if is_reset:
            n = 0
            moves = 0
        else:
            # level change resets the per-level counters (Schema n is per-level)
            if cur_lvl is not None and cur_lvl != prev_lvl:
                n = 0
                moves = 0
            n += 1
            if not is_click:
                moves += 1
        prev = out
        prev_lvl = cur_lvl
    return trans


def load_streams(game, include_post_report=False):
    files = discover(game, include_post_report)
    trans = []
    for pull, fp in files:
        events = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        trans.extend(extract_fixed(events, f"{pull}/{BENCH_VERSION[game].split('-')[0]}", game))
    return trans, [p for p, _ in files]


def phase_key(r):
    return r["phase"]


def _diag_keys(game, events, stream_id):
    """Record alternative keys for transparency (NOT the fixed law): the
    click-INCLUSIVE parity for ka59, and the in-sample-picked modulus for wa30 —
    to document why the literal external law can differ from the in-sample pick."""
    trans = []
    prev = None
    prev_lvl = None
    n = 0
    for ev in events:
        et = ev.get("type")
        if et == "initial":
            prev = board_digest(ev["board"]); prev_lvl = ev.get("level"); n = 0
            continue
        if et != "action" or "board" not in ev or prev is None:
            continue
        out = board_digest(ev["board"])
        ak = ev.get("action_display") or ev.get("action_name") or "<unk>"
        aname = ev.get("action_name") or ""
        is_reset = (aname == "RESET" or ak == "RESET")
        trans.append({"stream": stream_id, "prev": prev, "act": ak, "out": out,
                      "n_all": n})
        cur = ev.get("level")
        if is_reset:
            n = 0
        else:
            if cur is not None and cur != prev_lvl:
                n = 0
            n += 1
        prev = out; prev_lvl = cur
    return trans


def diagnostics(game):
    """Alternative-key determinism for transparency (not the certificate)."""
    files = discover(game, False)
    trans = []
    for pull, fp in files:
        events = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        trans.extend(_diag_keys(game, events, f"{pull}"))
    out = {}
    variants = {"ka59": {"click_inclusive_n%2": lambda r: r["n_all"] % 2},
                "wa30": {"in_sample_pick_n%4": lambda r: r["n_all"] % 4,
                         "literal_law_n%3": lambda r: r["n_all"] % 3}}
    for name, fn in variants.get(game, {}).items():
        st = alias_stats(trans, extra=fn)
        out[name] = {"determinism": st["determinism"],
                     "repeat_visits": st["repeat_visits"],
                     "wilson_lb": wilson_lb(st["modal_hits"], st["repeat_visits"])}
    return out


def verify(game, include_post_report=False):
    trans, pulls = load_streams(game, include_post_report)
    base = alias_stats(trans)                       # unaugmented (frame,action)
    aug = alias_stats(trans, extra=phase_key)       # + fixed Schema phase
    lb_base = wilson_lb(base["modal_hits"], base["repeat_visits"])
    lb_aug = wilson_lb(aug["modal_hits"], aug["repeat_visits"])
    det = aug["determinism"]
    ok = (det is not None and det >= HOLDOUT_DET
          and lb_aug is not None and lb_aug >= HOLDOUT_WILSON_LB)
    return {
        "game": game,
        "engine_version": BENCH_VERSION[game],
        "n_streams": len(pulls),
        "pulls": pulls,
        "include_post_report": include_post_report,
        "base": {
            "determinism": base["determinism"],
            "repeat_visits": base["repeat_visits"],
            "aliased_keys": base["aliased_keys"],
            "wilson_lb": lb_base,
        },
        "fixed_augmented": {
            "determinism": det,
            "repeat_visits": aug["repeat_visits"],
            "modal_hits": aug["modal_hits"],
            "aliased_keys": aug["aliased_keys"],
            "wilson_lb": lb_aug,
        },
        "cert_bar": {"det": HOLDOUT_DET, "wilson_lb": HOLDOUT_WILSON_LB},
        "verdict": "RE-ENTERS-CERTIFIED-RESOLVER-SET" if ok
                   else "STAYS-ALIASED-UNRESOLVED",
        "certified": ok,
    }


LAWS = {
    "wa30": "mod-rate: bar filled=(mult*n+off)//D, per-level D table "
            "(world_model_v5.py L98-111); fixed key = n mod D(level), all actions tick.",
    "ka59": "parity-inverted: bar zeros=round(64*n/budget) (L31/L51); fuse ticks "
            "one per MOVE, clicks (ACTION6) do NOT tick (L20-22); fixed key = "
            "move_count mod 2 (clicks excluded).",
    "tr87": "floor(n/2): row63 bar=floor(n_actions/2), K=2 lv0-4 / K=4 lv5 "
            "(L19,L262-265); fixed key = n mod K(level), all actions tick.",
}


def main():
    results = {"mandate": "R17 OBJ-I / prog-synthesis R3; C6-legal fixed hypothesis",
               "source": "kaggle-data/schema_traces/.../world_model_v5.py (backtest-certified)",
               "cert_bar": {"det": HOLDOUT_DET, "wilson_lb": HOLDOUT_WILSON_LB},
               "note": "fixed external hypotheses; NO fitting, NO selection budget "
                       "(C6-legal). Streams reproduce holdout_report.json benchmark "
                       "engine-version set (sentinel_eval_v1 + sidecar excluded).",
               "laws": LAWS, "primary": {}, "appendix_with_sentinel_eval": {},
               "diagnostics_alt_keys": {}}
    print("=== OBJ-I fixed-resolver verification (primary: report stream set) ===")
    for g in ("wa30", "ka59", "tr87"):
        r = verify(g, include_post_report=False)
        results["primary"][g] = r
        print(f"{g}: base det={_f(r['base']['determinism'])} "
              f"-> fixed-aug det={_f(r['fixed_augmented']['determinism'])} "
              f"(visits={r['fixed_augmented']['repeat_visits']}, "
              f"LB={_f(r['fixed_augmented']['wilson_lb'])}) "
              f"[{r['n_streams']} streams] -> {r['verdict']}")
    print("\n=== appendix: same laws + post-report sentinel_eval_v1 stream ===")
    for g in ("wa30", "ka59", "tr87"):
        r = verify(g, include_post_report=True)
        results["appendix_with_sentinel_eval"][g] = r
        print(f"{g}: fixed-aug det={_f(r['fixed_augmented']['determinism'])} "
              f"(visits={r['fixed_augmented']['repeat_visits']}, "
              f"LB={_f(r['fixed_augmented']['wilson_lb'])}) "
              f"[{r['n_streams']} streams] -> {r['verdict']}")
    for g in ("wa30", "ka59"):
        results["diagnostics_alt_keys"][g] = diagnostics(g)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT / 'results.json'}")
    return results


def _f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else ("-" if x is None else str(x))


if __name__ == "__main__":
    main()
