"""Latent-state audit — how much hidden state does the observable frame miss?

Panel R15 (5/5): state-aliasing is one root cause behind (1) predict-metric
recurrence acc 0.465, (2) EWM step-0 plan aborts, (3) the N5 prune_trace bug.
This audit is the BLOCKING prereq for EWM Stage-1 and any banking/replay build.

Protocol: learnings/war_room/latent_state_audit_protocol.md
Data:     runs/kernel_pulls/*/artifacts/*_events.jsonl (+ runs/phase1_ab/seed1)
Cross:    runs/ewm_dryrun/raw.json (sim fidelity per game)
Output:   runs/latent_state_audit/report.md + report.json

Usage:  uv run python scripts/latent_state_audit.py            # selftests + audit
        uv run python scripts/latent_state_audit.py --selftest # selftests only
        uv run python scripts/latent_state_audit.py --holdout  # held-out resolver
                                                               # validation (R16 C6)

Holdout mode (R16 C6 / R17 checklist item 2): for each ALIASED-RESOLVABLE game,
the resolver is fit/selected on a TRAIN subset of streams (alternating 4/4 split
of the benchmark-engine-version streams; engine versions are NEVER pooled) and
certified on the HELD-OUT streams: held-out augmented determinism >= 0.99 AND
Wilson 95% lower bound >= 0.95, else the game drops to UNRESOLVED.
Output: runs/latent_state_audit/holdout_report.md + holdout_report.json.

Stdlib-only, CPU, offline, $0.
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "latent_state_audit"

DET_THRESHOLD = 0.99          # >= this weighted determinism = resolved/clean
MIN_REPEAT_VISITS = 20        # below this: LOW-SUPPORT (no verdict earned)
SUPPORT_FLOOR = 10            # augmentation eligibility floor ...
SUPPORT_FRAC = 0.20           # ... and >= 20% of base repeat-visit mass

# Holdout certification (R16 C6): held-out det >= 0.99 AND Wilson LB >= 0.95
HOLDOUT_DET = 0.99
HOLDOUT_WILSON_LB = 0.95
Z95 = 1.959963984540054       # two-sided 95% normal quantile


def wilson_lb(k: int, n: int, z: float = Z95):
    """Wilson score interval lower bound for k successes in n trials."""
    if not n:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - rad) / denom

# ---------------------------------------------------------------------------
# transition extraction
# ---------------------------------------------------------------------------

def board_digest(board) -> str:
    return hashlib.blake2b(json.dumps(board, separators=(",", ":")).encode(),
                           digest_size=8).hexdigest()


def extract_transitions(events, stream_id: str):
    """Walk one recorded game trace -> list of transition dicts.

    t = actions since last RESET outcome (counts no-ops: N5 proved no-ops tick
    hidden state). hist = last 3 action keys. level/score are PRE-action.
    """
    trans = []
    prev = None
    prev_lvl = prev_score = None
    prev_bc = None
    t = 0
    hist = []
    analysis_drift = 0

    for ev in events:
        et = ev.get("type")
        if et == "initial":
            prev = board_digest(ev["board"])
            prev_lvl = ev.get("level")
            prev_score = ev.get("score")
            t = 0
            hist = []
            prev_bc = None
            continue
        if et != "action":
            # analysis frames: no action taken; verify they don't move the board
            if et == "analysis" and prev is not None and "board" in ev:
                if board_digest(ev["board"]) != prev:
                    analysis_drift += 1
            continue
        if "board" not in ev or prev is None:
            continue
        out = board_digest(ev["board"])
        ak = ev.get("action_display") or ev.get("action_name") or "<unk>"
        trans.append({
            "stream": stream_id,
            "prev": prev,
            "act": ak,
            "out": out,
            "t": t,
            "bc": bool(ev.get("board_changed", prev != out)),
            "prev_bc": prev_bc,
            "lvl": prev_lvl,
            "score": prev_score,
            "hist": tuple(hist[-3:]),
        })
        # advance
        if ev.get("action_name") == "RESET" or ak == "RESET":
            t = 0
        else:
            t += 1
        hist.append(ak)
        prev = out
        prev_bc = bool(ev.get("board_changed", False))
        prev_lvl = ev.get("level")
        prev_score = ev.get("score")

    return trans, analysis_drift


# ---------------------------------------------------------------------------
# aliasing metrics
# ---------------------------------------------------------------------------

# (name, key-fn, class); order = resolution-complexity rank (minimal wins)
CANDIDATES = [
    ("level",       lambda r: r["lvl"],                          "observable-meta"),
    ("score",       lambda r: r["score"],                        "observable-meta"),
    ("meta",        lambda r: (r["lvl"], r["score"]),            "observable-meta"),
    ("parity",      lambda r: r["t"] % 2,                        "hidden-phase"),
    ("mod3",        lambda r: r["t"] % 3,                        "hidden-phase"),
    ("mod4",        lambda r: r["t"] % 4,                        "hidden-phase"),
    ("mod5",        lambda r: r["t"] % 5,                        "hidden-phase"),
    ("prev_bc",     lambda r: r["prev_bc"],                      "hidden-history"),
    ("hist1",       lambda r: r["hist"][-1:],                    "hidden-history"),
    ("hist2",       lambda r: r["hist"][-2:],                    "hidden-history"),
    ("hist3",       lambda r: r["hist"],                         "hidden-history"),
    ("meta_parity", lambda r: (r["lvl"], r["score"], r["t"] % 2), "compound"),
    ("meta_hist1",  lambda r: (r["lvl"], r["score"], r["hist"][-1:]), "compound"),
]
DIAGNOSTIC = [("tcount", lambda r: r["t"], "diagnostic")]
CAND_FN = {n: f for n, f, _c in CANDIDATES}
CAND_CLASS = {n: c for n, _f, c in CANDIDATES}


def alias_stats(trans, extra=None, per_stream=False):
    """Outcome-distribution stats over keys with >= 2 visits.

    extra: optional fn(record) -> hashable augmentation.
    per_stream: scope keys to a single trace (within-stream aliasing).
    """
    buckets = defaultdict(Counter)
    for r in trans:
        key = (r["prev"], r["act"])
        if extra is not None:
            key = key + (extra(r),)
        if per_stream:
            key = key + (r["stream"],)
        buckets[key][r["out"]] += 1

    repeat_keys = aliased_keys = 0
    visits = maxhits = 0
    ent_w = 0.0
    noeff_aliased = 0  # aliased keys where one outcome == the pre-state (no-effect)
    for key, dist in buckets.items():
        n = sum(dist.values())
        if n < 2:
            continue
        repeat_keys += 1
        visits += n
        maxhits += max(dist.values())
        if len(dist) > 1:
            aliased_keys += 1
            pre = key[0]
            if pre in dist:
                noeff_aliased += 1
        ent = -sum((c / n) * math.log2(c / n) for c in dist.values())
        ent_w += ent * n

    return {
        "n_transitions": len(trans),
        "n_keys": len(buckets),
        "repeat_keys": repeat_keys,
        "repeat_visits": visits,
        "modal_hits": maxhits,
        "aliased_keys": aliased_keys,
        "determinism": (maxhits / visits) if visits else None,
        "entropy_bits": (ent_w / visits) if visits else None,
        "aliased_key_rate": (aliased_keys / repeat_keys) if repeat_keys else None,
        "noeff_involved_aliased_keys": noeff_aliased,
    }


def audit_game(trans):
    """Full audit for one versioned game -> dict with base/candidates/verdict."""
    base = alias_stats(trans)
    within = alias_stats(trans, per_stream=True)
    res = {"base": base, "within_stream": within, "candidates": {}}

    base_visits = base["repeat_visits"]
    resolver = None
    if base_visits < MIN_REPEAT_VISITS:
        verdict = "LOW-SUPPORT"
    elif base["determinism"] is not None and base["determinism"] >= DET_THRESHOLD:
        verdict = "CLEAN"
    else:
        verdict = "ALIASED-UNRESOLVED"
        floor = max(SUPPORT_FLOOR, SUPPORT_FRAC * base_visits)
        for name, fn, cls in CANDIDATES:
            st = alias_stats(trans, extra=fn)
            st["class"] = cls
            st["eligible"] = bool(st["repeat_visits"] >= floor)
            st["resolves"] = bool(st["eligible"] and st["determinism"] is not None
                                  and st["determinism"] >= DET_THRESHOLD)
            res["candidates"][name] = st
            if resolver is None and st["resolves"]:
                resolver = name
                verdict = ("CLEAN-META" if cls == "observable-meta"
                           else f"ALIASED-RESOLVABLE({name})")
        for name, fn, cls in DIAGNOSTIC:
            st = alias_stats(trans, extra=fn)
            st["class"] = cls
            st["eligible"] = False
            st["resolves"] = False
            res["candidates"][name] = st

    # near-miss: candidate fully deterministic on surviving support but the
    # augmentation shattered too much repeat mass to claim resolution
    res["near_miss"] = sorted(
        n for n, st in res["candidates"].items()
        if st["class"] != "diagnostic" and not st["resolves"]
        and st["determinism"] is not None and st["determinism"] >= DET_THRESHOLD
        and not st["eligible"] and st["repeat_visits"] > 0
    ) if verdict == "ALIASED-UNRESOLVED" else []

    res["verdict"] = verdict
    res["resolver"] = resolver
    if resolver:
        res["resolver_class"] = dict((n, c) for n, _f, c in CANDIDATES)[resolver]
        res["determinism_resolved"] = res["candidates"][resolver]["determinism"]
    else:
        res["resolver_class"] = None
        res["determinism_resolved"] = None
    return res


def consumer_flags(verdict, resolver_class):
    """Answer the two consumers directly from the verdict."""
    clean = verdict in ("CLEAN", "CLEAN-META")
    phase = resolver_class == "hidden-phase"
    unresolved = verdict == "ALIASED-UNRESOLVED"
    return {
        # EWM Stage-1: can a frame-conditioned sim be a faithful carrier?
        "ewm_carrier": "SAFE" if clean else ("PHASE-AUGMENT" if phase else
                       ("HISTORY-AUGMENT" if verdict.startswith("ALIASED-RESOLVABLE")
                        else ("NO" if unresolved else "N/A"))),
        # resync-before-abort: drifting phase => resync works; unresolved => no
        "resync_viable": "YES" if phase else ("NOT-NEEDED" if clean else
                         ("NO" if unresolved else "N/A")),
        # banking: prefix-splice/prune only safe when the frame is Markov;
        # N5 proved full-replay-from-RESET survives everywhere.
        "banking": "PREFIX-SAFE" if clean else
                   ("FULL-REPLAY-ONLY" if not verdict.startswith("LOW") else "N/A"),
    }


# ---------------------------------------------------------------------------
# held-out resolver validation (R16 C6 / R17 checklist item 2)
# ---------------------------------------------------------------------------

def split_streams(trans, train_stride=2):
    """Deterministic alternating stream split (sorted ids; even index -> TRAIN).

    train_stride=2 -> 4/4 on 8 streams (the split named by the panel);
    train_stride=3 -> 3/5 (sensitivity only, non-binding).
    Streams MUST already be from a single engine version (caller's contract) —
    engine-version drift must never be pooled into a resolver certificate.
    """
    ids = sorted(set(r["stream"] for r in trans))
    train_ids = set(ids[0::train_stride])
    hold_ids = [i for i in ids if i not in train_ids]
    return ids, sorted(train_ids), hold_ids


def _stream_rows(trans, ids, train_ids, fn):
    """Per-stream (within-stream keys) det + Wilson LB under augmentation fn."""
    rows = []
    tset = set(train_ids)
    for sid in ids:
        sub = [r for r in trans if r["stream"] == sid]
        st = alias_stats(sub, extra=fn)
        rows.append({
            "stream": sid,
            "split": "TRAIN" if sid in tset else "HOLDOUT",
            "actions": len(sub),
            "repeat_visits": st["repeat_visits"],
            "determinism": st["determinism"],
            "wilson_lb": wilson_lb(st["modal_hits"], st["repeat_visits"]),
        })
    return rows


def holdout_validate(trans, train_stride=2):
    """Fit/select resolver on TRAIN streams, certify on HELD-OUT streams.

    Returns status:
      KEEP            — train-selected resolver certifies on holdout
                        (det >= HOLDOUT_DET and Wilson LB >= HOLDOUT_WILSON_LB)
      DROP-HOLDOUT    — a resolver was selected on train but fails the holdout
                        certificate -> game drops to UNRESOLVED
      DROP-FIT-FAILED — train subset selects no resolver (no peeking at holdout
                        to rescue selection) -> UNRESOLVED
      NO-SPLIT        — < 2 streams; cannot validate -> UNRESOLVED
    """
    ids, train_ids, hold_ids = split_streams(trans, train_stride)
    out = {"streams": ids, "train_streams": train_ids, "holdout_streams": hold_ids,
           "thresholds": {"det": HOLDOUT_DET, "wilson_lb": HOLDOUT_WILSON_LB}}
    if not train_ids or not hold_ids:
        out.update(status="NO-SPLIT", validated=False, resolver=None,
                   per_stream=_stream_rows(trans, ids, train_ids, None) if ids else [])
        return out

    train = [r for r in trans if r["stream"] in set(train_ids)]
    hold = [r for r in trans if r["stream"] in set(hold_ids)]

    fit = audit_game(train)
    out["train_verdict"] = fit["verdict"]
    resolver = fit["resolver"]
    out["resolver"] = resolver
    out["resolver_class"] = fit["resolver_class"]
    out["train_det_resolved"] = fit["determinism_resolved"]

    if resolver is None:
        # strict: selection must come from TRAIN alone; no fallback to the
        # in-sample (full-data) resolver — that is exactly the leak C6 bans.
        out.update(status="DROP-FIT-FAILED", validated=False,
                   per_stream=_stream_rows(trans, ids, train_ids, None))
        return out

    fn = CAND_FN[resolver]
    hb = alias_stats(hold)                    # holdout base (unaugmented)
    hs = alias_stats(hold, extra=fn)          # holdout augmented
    lb = wilson_lb(hs["modal_hits"], hs["repeat_visits"])
    out["holdout"] = {
        "base_determinism": hb["determinism"],
        "base_repeat_visits": hb["repeat_visits"],
        "determinism": hs["determinism"],
        "repeat_visits": hs["repeat_visits"],
        "modal_hits": hs["modal_hits"],
        "wilson_lb": lb,
    }
    ok = (hs["determinism"] is not None and hs["determinism"] >= HOLDOUT_DET
          and lb is not None and lb >= HOLDOUT_WILSON_LB)
    out["status"] = "KEEP" if ok else "DROP-HOLDOUT"
    out["validated"] = ok
    out["per_stream"] = _stream_rows(trans, ids, train_ids, fn)
    return out


def clean_certificate(trans):
    """Pooled Wilson certificate for a CLEAN game's base determinism.

    No resolver was *selected* for CLEAN games, so there is no selection leak
    and no split is required — but Q5 makes prefix-splice legality inherit the
    held-out/Wilson standard, so the pooled LB is published per game/stream.
    """
    st = alias_stats(trans)
    lb = wilson_lb(st["modal_hits"], st["repeat_visits"])
    ids = sorted(set(r["stream"] for r in trans))
    return {
        "determinism": st["determinism"],
        "repeat_visits": st["repeat_visits"],
        "wilson_lb": lb,
        "confirmed": bool(st["determinism"] is not None
                          and st["determinism"] >= HOLDOUT_DET
                          and lb is not None and lb >= HOLDOUT_WILSON_LB),
        "per_stream": _stream_rows(trans, ids, [], None),
    }


# ---------------------------------------------------------------------------
# selftests
# ---------------------------------------------------------------------------

def _mk(stream, prev, act, out, t, hist, lvl=1, score=0, prev_bc=None):
    return {"stream": stream, "prev": prev, "act": act, "out": out, "t": t,
            "bc": prev != out, "prev_bc": prev_bc, "lvl": lvl, "score": score,
            "hist": tuple(hist[-3:])}


def synth_stream(kind, stream_id, n, rng):
    """Synthetic 1-frame-per-position games; frame shows position only."""
    trans = []
    pos, counter = 0, 0
    hist = []
    prev_bc = None
    for t in range(n):
        act = rng.choice(["ADV", "STAY"])
        prev = f"pos{pos}"
        if kind == "mod3":                     # hidden counter: ADV fires iff c%3==0
            newpos = (pos + 1) % 10 if (act == "ADV" and counter % 3 == 0) else pos
        elif kind == "clean":                  # frame-Markov
            newpos = (pos + 1) % 10 if act == "ADV" else pos
        elif kind == "coin":                   # truly stochastic
            newpos = (pos + 1) % 10 if (act == "ADV" and rng.random() < 0.5) else pos
        else:
            raise ValueError(kind)
        out = f"pos{newpos}"
        trans.append(_mk(stream_id, prev, act, out, t, hist, prev_bc=prev_bc))
        hist.append(act)
        prev_bc = prev != out
        pos, counter = newpos, counter + 1
    return trans


def selftest():
    rng = random.Random(20260720)
    failures = []

    # 1. hidden mod-3 counter -> must recover mod3 as minimal resolver
    tr = [x for s in range(3) for x in synth_stream("mod3", f"s{s}", 400, rng)]
    r = audit_game(tr)
    ok = (r["verdict"] == "ALIASED-RESOLVABLE(mod3)" and r["resolver"] == "mod3"
          and r["determinism_resolved"] is not None
          and r["determinism_resolved"] >= DET_THRESHOLD
          and r["base"]["determinism"] < DET_THRESHOLD)
    if not ok:
        failures.append(f"mod3 recovery: verdict={r['verdict']} resolver={r['resolver']} "
                        f"base_det={r['base']['determinism']}")

    # 2. clean Markov walk -> CLEAN, zero entropy
    tr = [x for s in range(3) for x in synth_stream("clean", f"s{s}", 400, rng)]
    r = audit_game(tr)
    if not (r["verdict"] == "CLEAN" and r["base"]["entropy_bits"] == 0.0):
        failures.append(f"clean: verdict={r['verdict']} H={r['base']['entropy_bits']}")

    # 3. coin-flip transitions -> ALIASED-UNRESOLVED (no candidate may claim it)
    tr = [x for s in range(3) for x in synth_stream("coin", f"s{s}", 400, rng)]
    r = audit_game(tr)
    if r["verdict"] != "ALIASED-UNRESOLVED":
        failures.append(f"coin: verdict={r['verdict']} resolver={r['resolver']}")

    for f in failures:
        print(f"SELFTEST FAIL: {f}")
    status = "PASS" if not failures else "FAIL"
    print(f"selftest: {status} (3 synthetic games: hidden-mod3 recovered, "
          f"clean=CLEAN, coin=UNRESOLVED)")
    return status


def selftest_holdout():
    """Holdout-mode selftests (R16 C6):
    1. synthetic hidden-mod3 game, 6 streams -> resolver fit on 3 TRAIN streams
       must certify on 3 HELD-OUT streams (det >= 0.99, Wilson LB >= 0.95): KEEP.
    2. synthetic coin-flip noise -> must end UNRESOLVED (no resolver survives).
    3. synthetic hidden-mod3 with tiny support -> train selects the resolver but
       the holdout Wilson LB < 0.95: must DROP to UNRESOLVED (the dc22 case).
    """
    rng = random.Random(20260722)
    failures = []

    tr = [x for s in range(6) for x in synth_stream("mod3", f"s{s}", 400, rng)]
    v = holdout_validate(tr)
    if not (v["status"] == "KEEP" and v["validated"] and v["resolver"] == "mod3"
            and v["holdout"]["determinism"] >= HOLDOUT_DET
            and v["holdout"]["wilson_lb"] >= HOLDOUT_WILSON_LB):
        failures.append(f"holdout mod3 KEEP: status={v['status']} "
                        f"resolver={v.get('resolver')} hold={v.get('holdout')}")

    tr = [x for s in range(6) for x in synth_stream("coin", f"s{s}", 400, rng)]
    v = holdout_validate(tr)
    if v["validated"] or v["status"] == "KEEP":
        failures.append(f"holdout coin must be UNRESOLVED: status={v['status']} "
                        f"resolver={v.get('resolver')}")

    tr = [x for s in range(6) for x in synth_stream("mod3", f"s{s}", 15, rng)]
    v = holdout_validate(tr)
    if not (v["status"] in ("DROP-HOLDOUT", "DROP-FIT-FAILED")
            and not v["validated"]):
        failures.append(f"holdout low-support mod3 must DROP: status={v['status']} "
                        f"hold={v.get('holdout')}")

    for f in failures:
        print(f"HOLDOUT SELFTEST FAIL: {f}")
    status = "PASS" if not failures else "FAIL"
    print(f"holdout selftest: {status} (mod3 6x400 -> KEEP w/ Wilson LB >= 0.95; "
          f"coin -> UNRESOLVED; mod3 6x15 low-support -> DROP)")
    return status


# ---------------------------------------------------------------------------
# real-data drivers
# ---------------------------------------------------------------------------

def discover_streams():
    """-> {versioned_game_id: [(stream_id, filepath)]}"""
    pats = [str(ROOT / "runs" / "kernel_pulls" / "*" / "artifacts" / "*_events.jsonl"),
            str(ROOT / "runs" / "phase1_ab" / "*" / "artifacts" / "*_events.jsonl")]
    files = sorted(set(f for p in pats for f in glob.glob(p)))
    streams = defaultdict(list)
    for fp in files:
        p = Path(fp)
        gid = p.name.split("_")[0]                     # e.g. ls20-9607627b
        pull = p.parents[1].name                       # e.g. war_eval_v1
        streams[gid].append((pull, fp))
    return streams


def load_ewm_crossref():
    """Pool ewm_dryrun sim fidelity per 4-char game across sources."""
    fp = ROOT / "runs" / "ewm_dryrun" / "raw.json"
    if not fp.exists():
        return {}
    raw = json.loads(fp.read_text(encoding="utf-8"))
    pooled = defaultdict(lambda: {"plans": 0, "steps": 0, "matches": 0,
                                  "aborts": 0, "step0_aborts": 0})
    for src in raw.get("sources", {}).values():
        for g, st in src.get("aggregate", {}).get("games", {}).items():
            p = pooled[g]
            p["plans"] += st.get("plans", 0)
            p["steps"] += st.get("steps", 0)
            p["matches"] += st.get("matches", 0)
            p["aborts"] += st.get("aborts", 0)
            p["step0_aborts"] += sum(1 for s in st.get("abort_steps", []) if s == 0)
    out = {}
    for g, p in pooled.items():
        out[g] = {
            **p,
            "step_acc": (p["matches"] / p["steps"]) if p["steps"] else None,
            "step0_abort_share": (p["step0_aborts"] / p["plans"]) if p["plans"] else None,
        }
    return out


VERDICT_RANK = {"LOW-SUPPORT": 0, "CLEAN": 1, "CLEAN-META": 2}  # else 3/4 below


def verdict_severity(v):
    if v in VERDICT_RANK:
        return VERDICT_RANK[v]
    return 4 if v == "ALIASED-UNRESOLVED" else 3


def _load_per_version(streams):
    """-> {versioned_gid: {"trans": [...], "pulls": [...], "n_streams": int}}"""
    out = {}
    for gid, lst in sorted(streams.items()):
        trans = []
        for pull, fp in lst:
            events = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
            tr, _drift = extract_transitions(events, f"{pull}/{gid}")
            trans.extend(tr)
        out[gid] = {"trans": trans, "pulls": [p for p, _ in lst], "n_streams": len(lst)}
    return out


def run_holdout(selftest_status):
    """Held-out resolver validation over the real streams (R17 item 2)."""
    streams = discover_streams()
    if not streams:
        print("no *_events.jsonl streams found — nothing to validate")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pv = _load_per_version(streams)
    audits = {gid: audit_game(d["trans"]) for gid, d in pv.items()}

    games = defaultdict(list)
    for gid in pv:
        games[gid.split("-")[0]].append(gid)

    rows = []
    for g, gids in sorted(games.items()):
        # benchmark engine version = most streams (same rule as the main audit);
        # minority engine versions are EXCLUDED from the certificate — resolver
        # fit/validation never pools across engine versions (cn04/ka59 drift).
        bench = max(gids, key=lambda k: (pv[k]["n_streams"],
                                         verdict_severity(audits[k]["verdict"])))
        excluded = sorted(k for k in gids if k != bench)
        r = audits[bench]
        trans = pv[bench]["trans"]
        row = {
            "game": g,
            "engine_version": bench,
            "excluded_versions": excluded,
            "n_streams": pv[bench]["n_streams"],
            "insample_verdict": r["verdict"],
            "insample_resolver": r["resolver"],
        }
        v = r["verdict"]
        if v.startswith("ALIASED-RESOLVABLE"):
            hv = holdout_validate(trans)                     # 4/4 (binding)
            hv35 = holdout_validate(trans, train_stride=3)   # 3/5 (sensitivity)
            keep = hv["status"] == "KEEP"
            validated = (f"ALIASED-RESOLVABLE({hv['resolver']})" if keep
                         else "ALIASED-UNRESOLVED")
            rcls = hv["resolver_class"] if keep else None
            row.update(mode="resolver", holdout=hv,
                       sensitivity_35={"status": hv35["status"],
                                       "resolver": hv35.get("resolver"),
                                       "holdout": hv35.get("holdout")},
                       validated_verdict=validated,
                       changed=not keep,
                       **consumer_flags(validated, rcls))
        elif v in ("CLEAN", "CLEAN-META"):
            cert = clean_certificate(trans)
            row.update(mode="clean-certificate", clean_cert=cert,
                       validated_verdict=v, changed=False,
                       **consumer_flags(v, None))
            if not cert["confirmed"]:
                row["banking"] = "PREFIX-SAFE (LB<0.95 — flagged)"
        else:
            row.update(mode="unchanged", validated_verdict=v, changed=False,
                       **consumer_flags(v, r["resolver_class"]))
        rows.append(row)
        print(f"  {g} ({bench}): {row['insample_verdict']} -> "
              f"{row['validated_verdict']}"
              + (f" [{row['holdout']['status']}, hold det="
                 f"{_f(row['holdout'].get('holdout', {}).get('determinism'))}, "
                 f"LB={_f(row['holdout'].get('holdout', {}).get('wilson_lb'))}]"
                 if row["mode"] == "resolver" else ""))

    report = {
        "protocol": "learnings/war_room/latent_state_audit_protocol.md",
        "mandate": "R16 C6 (rl-planning MAJOR#2, prog-synthesis N2); "
                   "R17 checklist item 2",
        "selftest": selftest_status,
        "split": "sorted stream ids, alternating; even index -> TRAIN (4/4 on "
                 "8 streams); engine versions never pooled",
        "thresholds": {"holdout_det": HOLDOUT_DET,
                       "holdout_wilson_lb": HOLDOUT_WILSON_LB,
                       "fit_det": DET_THRESHOLD,
                       "support_floor": SUPPORT_FLOOR,
                       "support_frac": SUPPORT_FRAC},
        "rows": rows,
    }
    (OUT_DIR / "holdout_report.json").write_text(json.dumps(report, indent=1),
                                                 encoding="utf-8")
    _write_holdout_md(report)
    kept = [r["game"] for r in rows if r.get("mode") == "resolver" and not r["changed"]]
    dropped = [r["game"] for r in rows if r["changed"]]
    print(f"\nwrote {OUT_DIR / 'holdout_report.md'} and holdout_report.json")
    print(f"resolver KEEP: {', '.join(kept) or 'NONE'}")
    print(f"dropped to UNRESOLVED: {', '.join(dropped) or 'NONE'}")


def _write_holdout_md(rep):
    L = []
    L.append("# Held-out resolver validation — latent-state audit (R17 item 2)\n")
    L.append(f"Mandate: {rep['mandate']}. Protocol: `{rep['protocol']}` + this "
             "holdout extension. Selftest: **" + rep["selftest"] + "** "
             "(synthetic hidden-mod3 game: resolver fit on TRAIN certifies on "
             "HELD-OUT streams with Wilson LB >= 0.95 -> KEEP; synthetic "
             "coin-flip noise -> UNRESOLVED; synthetic low-support mod3 -> "
             "DROP despite held-out det 1.0).\n")
    th = rep["thresholds"]
    L.append(f"Method: for each in-sample ALIASED-RESOLVABLE game, streams of "
             f"the **benchmark engine version only** (versions are never "
             f"pooled — cn04/ka59 drift) are split {rep['split']}. The "
             f"resolver is fit/selected on TRAIN streams alone (same minimal-"
             f"candidate rule + support guard as the main audit, det >= "
             f"{th['fit_det']}); it certifies iff held-out pooled augmented "
             f"determinism >= {th['holdout_det']} AND its Wilson 95% lower "
             f"bound >= {th['holdout_wilson_lb']}. Any failure (fit fails on "
             f"TRAIN, or holdout certificate fails) -> **ALIASED-UNRESOLVED**. "
             f"No fallback to the in-sample resolver — that is the selection "
             f"leak C6 bans.\n")

    rows = rep["rows"]
    res_rows = [r for r in rows if r["mode"] == "resolver"]
    clean_rows = [r for r in rows if r["mode"] == "clean-certificate"]
    other_rows = [r for r in rows if r["mode"] == "unchanged"]

    L.append("## Per-game validation (in-sample ALIASED-RESOLVABLE games)\n")
    L.append("| game | engine version | streams (tr/ho) | in-sample verdict | "
             "train resolver | hold det | hold visits | Wilson LB | status | "
             "validated verdict | EWM carrier | resync | banking |")
    L.append("|---|---|---|---|---|---:|---:|---:|---|---|---|---|---|")
    for r in res_rows:
        hv = r["holdout"]
        h = hv.get("holdout", {})
        L.append("| {g} | {ev} | {tr}/{ho} | {iv} | {res} | {det} | {vis} | "
                 "{lb} | {st} | **{vv}** | {ew} | {rs} | {bk} |".format(
            g=r["game"], ev=r["engine_version"],
            tr=len(hv["train_streams"]), ho=len(hv["holdout_streams"]),
            iv=r["insample_verdict"], res=hv.get("resolver") or "-",
            det=_f(h.get("determinism")), vis=h.get("repeat_visits", "-"),
            lb=_f(h.get("wilson_lb")), st=hv["status"],
            vv=r["validated_verdict"], ew=r["ewm_carrier"],
            rs=r["resync_viable"], bk=r["banking"]))
    L.append("")
    changed = [r for r in res_rows if r["changed"]]
    kept = [r for r in res_rows if not r["changed"]]
    L.append(f"**Verdict changes: {len(changed)}/{len(res_rows)} in-sample "
             f"RESOLVABLE games drop to UNRESOLVED** "
             f"({', '.join(r['game'] for r in changed) or 'none'}); "
             f"{', '.join(r['game'] for r in kept) or 'none'} keep RESOLVABLE "
             "with a held-out certificate.\n")

    L.append("### 3/5-split sensitivity (non-binding)\n")
    L.append("Fit on 3 streams / certify on 5 (the directive's 'fit on <=6' "
             "variant maximises holdout support). Published for transparency; "
             "the 4/4 split above is the binding certificate. Any KEEP that "
             "flips across splits is split-sensitive and should be treated as "
             "fragile by consumers.\n")
    L.append("| game | status(3/5) | resolver | hold det | hold visits | Wilson LB |")
    L.append("|---|---|---|---:|---:|---:|")
    for r in res_rows:
        s = r["sensitivity_35"]
        h = s.get("holdout") or {}
        L.append(f"| {r['game']} | {s['status']} | {s.get('resolver') or '-'} | "
                 f"{_f(h.get('determinism'))} | {h.get('repeat_visits', '-')} | "
                 f"{_f(h.get('wilson_lb'))} |")
    L.append("")

    L.append("## Per-stream resolver table\n")
    L.append("Per-stream determinism under the TRAIN-selected resolver key "
             "(within-stream repeat visits); CLEAN games appear with the base "
             "(unaugmented) key and no split (nothing was fit).\n")
    L.append("| game | stream | engine version | split | visits | det | "
             "Wilson LB | game verdict (validated) |")
    L.append("|---|---|---|---|---:|---:|---:|---|")
    for r in res_rows:
        for s in r["holdout"].get("per_stream", []):
            L.append(f"| {r['game']} | {s['stream'].split('/')[0]} | "
                     f"{r['engine_version']} | {s['split']} | "
                     f"{s['repeat_visits']} | {_f(s['determinism'])} | "
                     f"{_f(s['wilson_lb'])} | {r['validated_verdict']} |")
    for r in clean_rows:
        for s in r["clean_cert"]["per_stream"]:
            L.append(f"| {r['game']} | {s['stream'].split('/')[0]} | "
                     f"{r['engine_version']} | - | {s['repeat_visits']} | "
                     f"{_f(s['determinism'])} | {_f(s['wilson_lb'])} | "
                     f"{r['validated_verdict']} |")
    L.append("")

    L.append("## CLEAN-game pooled certificates (Q5: splice legality inherits "
             "the Wilson standard)\n")
    L.append("No resolver was selected for CLEAN games (no selection leak), so "
             "the pooled base determinism carries the certificate.\n")
    L.append("| game | engine version | det | rep.visits | Wilson LB | "
             "prefix-splice |")
    L.append("|---|---|---:|---:|---:|---|")
    for r in clean_rows:
        c = r["clean_cert"]
        L.append(f"| {r['game']} | {r['engine_version']} | "
                 f"{_f(c['determinism'])} | {c['repeat_visits']} | "
                 f"{_f(c['wilson_lb'])} | "
                 f"{'CONFIRMED' if c['confirmed'] else 'LB<0.95 — flagged'} |")
    L.append("")
    if other_rows:
        L.append("Unchanged (already UNRESOLVED or LOW-SUPPORT in-sample): "
                 + ", ".join(f"{r['game']} ({r['insample_verdict']})"
                             for r in other_rows) + ".\n")

    L.append("## Updated consumer answers (held-out numbers are now the "
             "binding ones)\n")
    safe = [r["game"] for r in rows if r["ewm_carrier"] == "SAFE"]
    phase = [r["game"] for r in rows if r["resync_viable"] == "YES"]
    hist = [r["game"] for r in rows if r["ewm_carrier"] == "HISTORY-AUGMENT"]
    nogo = [r["game"] for r in rows if r["ewm_carrier"] == "NO"]
    pfx = [r["game"] + (" (flagged)" if "flagged" in str(r["banking"]) else "")
           for r in rows if str(r["banking"]).startswith("PREFIX-SAFE")]
    L.append(f"- **EWM Stage-1 safe carriers** (unchanged): {', '.join(safe) or 'NONE'}")
    L.append(f"- **PHASE-AUGMENT / resync-viable** (held-out certified only): "
             f"{', '.join(phase) or 'NONE'}")
    L.append(f"- **HISTORY-AUGMENT** (held-out certified history resolver; "
             f"resync-before-abort NOT implied): {', '.join(hist) or 'NONE'}")
    L.append(f"- **EWM no-go** (unresolved, incl. holdout drops): "
             f"{', '.join(nogo) or 'NONE'}")
    L.append(f"- **Banking prefix-splice** (CLEAN only; see pooled "
             f"certificates): {', '.join(pfx) or 'NONE'}; everything else "
             "FULL-REPLAY-ONLY from RESET, zero pruning.")
    L.append("- Downstream consumers (EWM phase-augment, banking keying, "
             "resurrection prong (i)) must re-point at THESE numbers per R17 "
             "checklist item 2.")
    L.append("")
    (OUT_DIR / "holdout_report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    args = sys.argv[1:]
    st = selftest()
    st_h = selftest_holdout()
    if st != "PASS" or st_h != "PASS":
        sys.exit(1)
    if "--selftest" in args:
        return
    if "--holdout" in args:
        run_holdout(f"{st} (base) / {st_h} (holdout)")
        return

    streams = discover_streams()
    if not streams:
        print("no *_events.jsonl streams found — nothing to audit")
        sys.exit(1)

    ewm = load_ewm_crossref()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_version = {}
    total_actions = 0
    total_drift = 0
    for gid, lst in sorted(streams.items()):
        trans = []
        for pull, fp in lst:
            events = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
            tr, drift = extract_transitions(events, f"{pull}/{gid}")
            trans.extend(tr)
            total_drift += drift
        total_actions += len(trans)
        r = audit_game(trans)
        r["n_streams"] = len(lst)
        r["pulls"] = [p for p, _ in lst]
        per_version[gid] = r
        print(f"  {gid}: {len(trans)} actions, {len(lst)} streams, "
              f"det={_f(r['base']['determinism'])} -> {r['verdict']}")

    # collapse versioned ids to 4-char game rows (worst verdict wins)
    games = defaultdict(list)
    for gid, r in per_version.items():
        games[gid.split("-")[0]].append((gid, r))
    rows = []
    for g, lst in sorted(games.items()):
        # The row reflects the CURRENT BENCHMARK engine version = the one with
        # the most streams (kernel pulls); minority versions (older engines,
        # e.g. phase1_ab/seed1) are reported as drift notes, never merged —
        # engine-version drift must not masquerade as hidden state (protocol §1).
        gid, r = max(lst, key=lambda kv: (kv[1]["n_streams"],
                                          verdict_severity(kv[1]["verdict"])))
        drift = [f"{k}:{v['verdict']}" for k, v in lst
                 if k != gid and v["verdict"] != r["verdict"]
                 and v["verdict"] != "LOW-SUPPORT"]
        flags = consumer_flags(r["verdict"], r["resolver_class"])
        e = ewm.get(g, {})
        rows.append({
            "game": g,
            "benchmark_version": gid,
            "version_drift": drift,
            "near_miss": r.get("near_miss", []),
            "versions": [k for k, _ in lst],
            "n_streams": sum(x[1]["n_streams"] for x in lst),
            "actions": r["base"]["n_transitions"],
            "repeat_visits": r["base"]["repeat_visits"],
            "aliased_keys": r["base"]["aliased_keys"],
            "repeat_keys": r["base"]["repeat_keys"],
            "determinism": r["base"]["determinism"],
            "entropy_bits": r["base"]["entropy_bits"],
            "within_determinism": r["within_stream"]["determinism"],
            "within_repeat_visits": r["within_stream"]["repeat_visits"],
            "resolver": r["resolver"],
            "determinism_resolved": r["determinism_resolved"],
            "verdict": r["verdict"],
            "ewm_step_acc": e.get("step_acc"),
            "ewm_step0_abort_share": e.get("step0_abort_share"),
            **flags,
        })

    report = {
        "protocol": "learnings/war_room/latent_state_audit_protocol.md",
        "selftest": st,
        "thresholds": {"determinism": DET_THRESHOLD,
                       "min_repeat_visits": MIN_REPEAT_VISITS,
                       "support_floor": SUPPORT_FLOOR, "support_frac": SUPPORT_FRAC},
        "coverage": {"versioned_games": len(per_version),
                     "streams": sum(len(v) for v in streams.values()),
                     "actions": total_actions,
                     "analysis_frame_drift": total_drift},
        "verdict_table": rows,
        "per_version": per_version,
        "ewm_crossref": ewm,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    _write_md(report)
    print(f"\nwrote {OUT_DIR / 'report.md'} and report.json "
          f"({report['coverage']['streams']} streams, {total_actions} actions)")

    counts = Counter(r["verdict"].split("(")[0] for r in rows)
    print("verdicts:", dict(counts))


def _f(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, float) else ("-" if x is None else str(x))


def _write_md(rep):
    L = []
    L.append("# Latent-state audit — per-game hidden-state quantification\n")
    L.append(f"Protocol: `{rep['protocol']}`. Selftest: **{rep['selftest']}** "
             "(synthetic hidden mod-3 counter recovered; clean stream = CLEAN; "
             "coin-flip stream = UNRESOLVED).\n")
    c = rep["coverage"]
    L.append(f"Coverage: {c['versioned_games']} versioned games, {c['streams']} "
             f"streams, {c['actions']} actions; analysis-frame drift events: "
             f"{c['analysis_frame_drift']}.\n")
    L.append("Determinism = P(modal next frame | frame, action) over keys seen "
             ">= 2x, visit-weighted (pooled across streams of the same engine "
             "version). Entropy = mean outcome entropy (bits). 'within' = keys "
             "scoped to a single stream (strongest hidden-state evidence). "
             f"Resolved = augmented determinism >= {rep['thresholds']['determinism']}.\n")

    L.append("## Verdict table\n")
    L.append("| game | streams | actions | rep.visits | det | H bits | "
             "within det | resolver | det.res | verdict | EWM step_acc | "
             "step0-abort/plan | EWM carrier | resync | banking |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---|---|---|")
    for r in rep["verdict_table"]:
        L.append("| {game} | {ns} | {ac} | {rv} | {det} | {ent} | {wd} | {res} | "
                 "{dr} | **{v}** | {sa} | {s0} | {ew} | {rs} | {bk} |".format(
            game=r["game"], ns=r["n_streams"], ac=r["actions"],
            rv=r["repeat_visits"], det=_f(r["determinism"]),
            ent=_f(r["entropy_bits"]), wd=_f(r["within_determinism"]),
            res=r["resolver"] or "-", dr=_f(r["determinism_resolved"]),
            v=r["verdict"], sa=_f(r["ewm_step_acc"]),
            s0=_f(r["ewm_step0_abort_share"]), ew=r["ewm_carrier"],
            rs=r["resync_viable"], bk=r["banking"]))
    L.append("")
    L.append("Rows reflect the benchmark engine version (most streams). "
             "Minority-version drift and near-misses:\n")
    for r in rep["verdict_table"]:
        if r["version_drift"]:
            L.append(f"- **{r['game']}**: older engine version(s) disagree — "
                     f"{'; '.join(r['version_drift'])} (phase1_ab/seed1 era); "
                     "engine-version drift, NOT merged into the verdict.")
        if r["near_miss"]:
            L.append(f"- **{r['game']}**: candidate(s) {', '.join(r['near_miss'])} "
                     "reach >= 0.99 determinism on the repeat support that "
                     "survives augmentation, but fail the support guard "
                     "(SUPPORT-COLLAPSED) — plausibly resolvable with more "
                     "data; treated as UNRESOLVED until then.")
    L.append("")

    L.append("## Candidate breakdown (aliased games only)\n")
    for r in rep["verdict_table"]:
        if not r["verdict"].startswith("ALIASED"):
            continue
        gid = r["benchmark_version"]
        pv = rep["per_version"][gid]
        L.append(f"### {r['game']} ({gid}) — base det {_f(pv['base']['determinism'])}, "
                 f"{pv['base']['aliased_keys']}/{pv['base']['repeat_keys']} keys aliased, "
                 f"{pv['base']['noeff_involved_aliased_keys']} involve a no-effect outcome\n")
        L.append("| candidate | class | det | rep.visits | eligible | resolves |")
        L.append("|---|---|---:|---:|---|---|")
        for name, st in pv["candidates"].items():
            L.append(f"| {name} | {st['class']} | {_f(st['determinism'])} | "
                     f"{st['repeat_visits']} | {'y' if st['eligible'] else 'n'} | "
                     f"{'YES' if st['resolves'] else '-'} |")
        L.append("")

    L.append("## Findings (ties to the three R15 failures)\n")
    rows = rep["verdict_table"]
    aliased = [r for r in rows if r["verdict"].startswith("ALIASED")]
    phase_res = [r for r in aliased if r["resolver"] in
                 ("parity", "mod3", "mod4", "mod5")]
    L.append(f"1. **Hidden phase counters are the dominant aliasing mechanism**: "
             f"{len(phase_res)}/{len(aliased)} aliased benchmark games are fully "
             "resolved (det -> ~1.000) by a small modular counter of "
             "actions-since-RESET (parity or mod 3/4/5) — an invisible blink/"
             "tick phase. Observable metadata (level/score) resolves NOTHING: "
             "the hidden variable is truly outside the observation.")
    L.append("2. **This is the predict-metric 0.465 mechanism**: in the aliased "
             "games, most aliased (frame,action) keys have a no-effect outcome "
             "on one phase and an effect on the other (see "
             "'involve a no-effect outcome' counts). A no-effect FACT keyed on "
             "(frame,action) alone is wrong whenever the phase differs on "
             "recurrence — exactly the ~54% flip rate R14 measured.")
    L.append("3. **This is the N5 prune_trace mechanism**: no-op actions still "
             "advance the phase counter; dropping leading no-ops desyncs the "
             "phase and the first replayed action lands on a different frame "
             "(step-0 frame_divergence on sc25/m0r0 — sc25 is mod5-aliased "
             "here; m0r0 is the worst unresolved game, det 0.618).")
    clean_bad_sim = [r["game"] for r in rows if r["verdict"] == "CLEAN"
                     and r["ewm_step_acc"] is not None and r["ewm_step_acc"] < 0.6]
    L.append("4. **EWM step-0 aborts split into two causes**: on ALIASED games "
             "(s5i5, sb26, vc33, tr87) low sim step_acc co-occurs with phase "
             "aliasing — the sim is phase-blind, and resync/phase-augmentation "
             "fixes it. But " + (", ".join(clean_bad_sim) or "(none)") +
             " are frame-Markov CLEAN yet still have step_acc < 0.6 — those "
             "sims are just wrong (sim bugs / engine-version drift), and NO "
             "amount of state augmentation or resync will save them; they need "
             "sim fixes, not aliasing work.")
    L.append("")
    L.append("## Consumer answers\n")
    safe = [r["game"] for r in rep["verdict_table"] if r["ewm_carrier"] == "SAFE"]
    phase = [r["game"] for r in rep["verdict_table"] if r["resync_viable"] == "YES"]
    nogo = [r["game"] for r in rep["verdict_table"] if r["ewm_carrier"] == "NO"]
    pfx = [r["game"] for r in rep["verdict_table"] if r["banking"] == "PREFIX-SAFE"]
    L.append(f"- **EWM Stage-1 safe carriers** (frame(+meta) is Markov): {', '.join(safe) or 'NONE'}")
    L.append(f"- **Resync-before-abort viable** (phase variable drifts, reality "
             f"deterministic): {', '.join(phase) or 'NONE'}")
    L.append(f"- **EWM no-go** (unresolved aliasing — abort-and-fallback is "
             f"correct): {', '.join(nogo) or 'NONE'}")
    L.append(f"- **Banking prefix-splice safe**: {', '.join(pfx) or 'NONE'}; all "
             "other audited games are FULL-REPLAY-ONLY from RESET with ZERO "
             "pruning (N5: full unpruned replay survives on all 25; the "
             "prune_trace bug dropped hidden-state-mutating no-ops).")
    L.append("")
    (OUT_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
