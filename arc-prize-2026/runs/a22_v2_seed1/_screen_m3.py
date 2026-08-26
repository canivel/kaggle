# Throwaway: M3 refuted-hypothesis re-proposal forensics for A22 v2 seed-1 screen.
# Re-implementation of the v1 procedure (learnings/sweeps/a22_seed1_screen_2026-08-03.md §3):
# [THINKING] blocks per analysis_step; sentence-split; PROPOSAL/REFUTATION cue tagging;
# stop-word-stripped token-set Jaccard; reprop J>=0.75 vs earlier PROPOSAL >=2 turns back;
# refuted-reprop J>=theta vs REFUTATION >=2 turns earlier, theta in {0.35,0.45,0.60}.
# Runs on v2 arm + war baseline (the binding paired comparison) and on the v1 arm as a
# calibration check against the recorded v1 JSON numbers.
import json, glob, os, re, sys

ROOT = r"F:\kaggle\arc-prize-2026"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from phase1_gate import signflip_p_exact

ARMS = {
    "v2":  os.path.join(ROOT, "runs", "a22_v2_seed1", "transcripts"),
    "war": os.path.join(ROOT, "runs", "kernel_pulls", "war_eval_v1", "transcripts"),
    "v1":  os.path.join(ROOT, "runs", "a22_compaction_v1", "transcripts"),
}

PROP_CUES = ["maybe", "what if", "the goal is", "hypothesis", "i think",
             "let me try", "new theory"]
REF_CUES = ["no effect", "didn't work", "ruled out", "impossible",
            "unchanged", "contradicts"]
STOP = set("""a an the and or but if then else for of to in on at by with from as is are was
were be been being it its this that these those i you we they he she them his her their my
our your me us so not no yes do does did done have has had will would can could should may
might must let s t re ve ll d m don didn doesn isn aren wasn weren won shouldn couldn
there here what which who whom when where why how all any both each few more most other
some such only own same than too very just now also again further once about into through
during before after above below up down out off over under""".split())

STEP_RE = re.compile(r"^--- analysis_step=(\d+)")
TAG_RE = re.compile(r"^\[[A-Z][A-Z >_-]*\]")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-z0-9']+")

def thinking_sentences(path):
    """-> list of (turn, sentence) over the whole transcript."""
    out = []
    turn = 0
    in_think = False
    buf = []
    def flush():
        nonlocal buf
        if buf:
            text = " ".join(buf)
            for s in SENT_RE.split(text):
                s = s.strip()
                if s:
                    out.append((cur_turn_of_buf, s))
        buf = []
    cur_turn_of_buf = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.match(line)
            if m:
                flush()
                in_think = False
                turn = int(m.group(1))
                continue
            if TAG_RE.match(line.strip()) and line.strip().startswith("["):
                flush()
                in_think = (line.strip() == "[THINKING]")
                cur_turn_of_buf = turn
                continue
            if in_think:
                buf.append(line.strip())
    flush()
    return out

def toks(s):
    return frozenset(w for w in WORD_RE.findall(s.lower()) if w not in STOP)

def jac(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / (len(a) + len(b) - inter)

def analyse_game(path):
    sents = thinking_sentences(path)
    props, refs = [], []   # (turn, tokset)
    for turn, s in sents:
        low = s.lower()
        ts = toks(s)
        if len(ts) < 3:
            continue
        if any(c in low for c in PROP_CUES):
            props.append((turn, ts))
        if any(c in low for c in REF_CUES):
            refs.append((turn, ts))
    n = len(props)
    if n == 0:
        return dict(n_props=0, n_refs=len(refs), reprop=0.0,
                    refrep={0.35: 0.0, 0.45: 0.0, 0.60: 0.0})
    reprop = 0
    refrep = {0.35: 0, 0.45: 0, 0.60: 0}
    for i, (t, ts) in enumerate(props):
        if any(jac(ts, ts2) >= 0.75 for t2, ts2 in props[:i] if t2 <= t - 2):
            reprop += 1
        for th in refrep:
            if any(jac(ts, rs) >= th for rt, rs in refs if rt <= t - 2):
                refrep[th] += 1
    return dict(n_props=n, n_refs=len(refs), reprop=reprop / n,
                refrep={th: c / n for th, c in refrep.items()})

results = {}
for arm, tdir in ARMS.items():
    per = {}
    for f in sorted(glob.glob(os.path.join(tdir, "*_p0.txt"))):
        g = os.path.basename(f).split("-")[0]
        per[g] = analyse_game(f)
    results[arm] = per
    print(f"[{arm}] done: {len(per)} games")

json.dump(results, open(os.path.join(ROOT, "runs", "a22_v2_seed1", "_m3_raw.json"), "w"), indent=1)

# ---- calibration vs recorded v1 numbers ----
v1json = json.load(open(os.path.join(ROOT, "runs", "a22_compaction_v1", "m1m2m3_screen.json")))
rec = {r["game"]: r for r in v1json["per_game"]}
print("\ncalibration (my impl vs recorded v1 screen):")
print(f"{'game':6}{'my_v1_rp':>9}{'rec_v1_rp':>10}{'my_war_rp':>10}{'rec_war_rp':>11}")
d1 = []; d2 = []
for g in sorted(rec):
    a = results['v1'][g]['reprop']; b = rec[g]['a22_reprop_rate']
    c = results['war'][g]['reprop']; e = rec[g]['war_reprop_rate']
    d1.append(a-b); d2.append(c-e)
    print(f"{g:6}{a:>9.3f}{b:>10.3f}{c:>10.3f}{e:>11.3f}")
print("mean abs dev v1-arm %.4f, war-arm %.4f" % (sum(map(abs,d1))/len(d1), sum(map(abs,d2))/len(d2)))

# ---- paired v2 vs war ----
games = sorted(results["v2"])
def paired(metric):
    deltas = []
    for g in games:
        a = results["v2"][g]
        w = results["war"][g]
        av = a["reprop"] if metric == "reprop" else a["refrep"][metric]
        wv = w["reprop"] if metric == "reprop" else w["refrep"][metric]
        deltas.append(av - wv)
    mean = sum(deltas) / len(deltas)
    nz = [d for d in deltas if d != 0]
    if nz:
        obs = sum(nz)
        p = min(1.0, 2 * signflip_p_exact(nz, abs(obs))[0])
    else:
        p = 1.0
    worse = sum(1 for d in deltas if d > 0)
    better = sum(1 for d in deltas if d < 0)
    return mean, p, worse, better

M3 = {}
for m, key in [("reprop", "reprop"), (0.35, "refrep035"), (0.45, "refrep045"), (0.60, "refrep060")]:
    mean, p, worse, better = paired(m)
    M3[key] = dict(mean_delta_pp=100*mean, p=p, a22_worse=worse, war_worse=better)
    print(f"M3 {key}: mean d {100*mean:+.2f} pp, p={p:.4f}, v2 worse in {worse}, war worse in {better}")
json.dump(M3, open(os.path.join(ROOT, "runs", "a22_v2_seed1", "_m3_summary.json"), "w"), indent=1)
