"""Public submission blending - replicating Yusuf's 0.91727 approach.

Download top public notebook outputs and blend with our best using
cascaded blending + positional weighting.
"""

import sys
sys.path.insert(0, "../kaggle-agent/src" if sys.platform == "win32" else "/app/kaggle-agent/src")

import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from kaggle_agent.pipeline.submission import generate_submission


def download_notebook_output(notebook_slug, output_dir="public_subs"):
    """Download a public notebook's output submission."""
    Path(output_dir).mkdir(exist_ok=True)
    safe_name = notebook_slug.replace("/", "_")
    out_path = Path(output_dir) / f"{safe_name}.csv"

    if out_path.exists():
        print(f"  Already have: {safe_name}", flush=True)
        return out_path

    print(f"  Downloading: {notebook_slug}...", flush=True)
    result = subprocess.run(
        ["kaggle", "kernels", "output", notebook_slug, "-p", output_dir],
        capture_output=True, text=True,
    )
    # Rename to safe name
    for f in Path(output_dir).glob("submission*.csv"):
        if f.name != out_path.name:
            f.rename(out_path)
            break

    if out_path.exists():
        return out_path
    # Try alternate name
    for f in Path(output_dir).glob("*.csv"):
        if safe_name not in f.name:
            f.rename(out_path)
            return out_path

    print(f"  FAILED to download {notebook_slug}", flush=True)
    return None


def cascaded_blend(base, others, alpha=0.95):
    """Cascaded blending: base dominates, others mixed in at (1-alpha) each."""
    result = base.copy()
    for other in others:
        result = result * alpha + other * (1 - alpha)
    return result


def h_blend_simple(submissions, weights, subwts=[0.11, -0.01, -0.03, -0.07],
                   asc_weight=0.30, desc_weight=0.70):
    """Simplified h_blend with positional weighting.

    For each row:
    - Sort submission values
    - Assign positional bonuses (subwts) based on rank
    - Combine with main weights
    """
    n_subs = len(submissions)
    n_rows = len(submissions[0])
    result = np.zeros(n_rows)

    # Ensure we have enough subwts
    while len(subwts) < n_subs:
        subwts.append(0.0)

    for i in range(n_rows):
        vals = np.array([s[i] for s in submissions])

        # Descending sort
        desc_order = np.argsort(-vals)
        score_desc = 0
        for j in range(n_subs):
            pos = np.where(desc_order == j)[0][0]  # position of submission j
            score_desc += vals[j] * (weights[j] + subwts[pos])

        # Ascending sort
        asc_order = np.argsort(vals)
        score_asc = 0
        for j in range(n_subs):
            pos = np.where(asc_order == j)[0][0]
            score_asc += vals[j] * (weights[j] + subwts[pos])

        result[i] = desc_weight * score_desc + asc_weight * score_asc

    return result


def rank_calibrate(pred_values, pred_ranks, weight_values=0.99, weight_adjust=0.01):
    """Artem's rank calibration: reorder pred_values by pred_ranks ordering."""
    rank_new = rankdata(pred_ranks) * weight_values
    # Add small random perturbation for tie-breaking
    rank_new += np.random.RandomState(42).randn(len(rank_new)) * weight_adjust

    # Map pred_values onto the new rank ordering
    df = pd.DataFrame({"rank": rank_new, "pred": pred_values})
    df_me = df.groupby("rank")["pred"].mean()
    df_me.loc[:] = np.sort(df_me.values)
    # Ensure monotonicity
    for i in range(1, len(df_me)):
        if df_me.iloc[i] <= df_me.iloc[i-1]:
            df_me.iloc[i] = df_me.iloc[i-1] + 1e-6 / len(df)
    result = df.join(df_me, on="rank", rsuffix="_new")
    return result["pred_new"].values


def main():
    print("=" * 70, flush=True)
    print("PUBLIC SUBMISSION BLENDING", flush=True)
    print("=" * 70, flush=True)

    test_ids = pd.read_csv("data/test.csv")["id"]

    # Try to download top public notebook outputs
    notebooks = [
        "artemevstafyev/cv-auc-0-91930-xgb-cb-blend",
        "anthonytherrien/predict-customer-churn-blend",
        "blamerx/s6e3-ridge-xgb-n-gram-0-91927-cv",
        "datasciencegrad/s6e3-detail-eda-baseline-xgb-auc-0-91808",
    ]

    public_subs = {}
    for nb in notebooks:
        path = download_notebook_output(nb)
        if path and path.exists():
            df = pd.read_csv(path)
            if "Churn" in df.columns:
                public_subs[nb.split("/")[0]] = df["Churn"].values
                print(f"  Loaded {nb.split('/')[0]}: [{df['Churn'].min():.4f}, {df['Churn'].max():.4f}]", flush=True)

    # Load our best
    our_best = pd.read_csv("submissions/iter6_blamerx.csv")["Churn"].values
    print(f"  Our iter6: [{our_best.min():.4f}, {our_best.max():.4f}]", flush=True)

    if not public_subs:
        print("\nNo public submissions downloaded. Using our own variants.", flush=True)
        # Blend our own submissions instead
        variants = {}
        for fname in ["iter6_blamerx", "iter9_xgb", "iter11_xgb", "iter5_tree"]:
            path = f"submissions/{fname}.csv"
            try:
                variants[fname] = pd.read_csv(path)["Churn"].values
            except FileNotFoundError:
                pass

        if len(variants) >= 2:
            names = list(variants.keys())
            # Cascaded blend with iter6 as base
            base = variants["iter6_blamerx"]
            others = [variants[k] for k in names if k != "iter6_blamerx"]
            cascaded = cascaded_blend(base, others, alpha=0.95)
            generate_submission(test_ids, cascaded, "id", "Churn", "submissions/cascaded_blend.csv")
            print(f"  Saved cascaded_blend.csv", flush=True)

            # Rank calibration: iter6 values with iter9 ranks
            if "iter9_xgb" in variants:
                rc = rank_calibrate(variants["iter6_blamerx"], variants["iter9_xgb"])
                generate_submission(test_ids, rc, "id", "Churn", "submissions/rank_calibrated_6_9.csv")
                print(f"  Saved rank_calibrated_6_9.csv", flush=True)

        return

    # === BLEND WITH PUBLIC SUBMISSIONS ===
    print("\n[Blending with public submissions]", flush=True)

    # 1. Cascaded blend (Yusuf's approach)
    base = list(public_subs.values())[0] if public_subs else our_best
    others = list(public_subs.values())[1:] + [our_best]
    cascaded = cascaded_blend(base, others, alpha=0.95)
    generate_submission(test_ids, cascaded, "id", "Churn", "submissions/public_cascaded.csv")
    print(f"  Saved public_cascaded.csv", flush=True)

    # 2. Simple average with top public
    if "artemevstafyev" in public_subs:
        avg = (public_subs["artemevstafyev"] + our_best) / 2
        generate_submission(test_ids, avg, "id", "Churn", "submissions/public_artem_avg.csv")
        print(f"  Saved public_artem_avg.csv", flush=True)

        # 3. Rank calibration (Artem's approach)
        rc = rank_calibrate(our_best, public_subs["artemevstafyev"])
        generate_submission(test_ids, rc, "id", "Churn", "submissions/public_rank_cal.csv")
        print(f"  Saved public_rank_cal.csv", flush=True)

    # 4. h_blend if we have 4+ submissions
    all_subs = list(public_subs.values()) + [our_best]
    if len(all_subs) >= 4:
        subs_4 = all_subs[:4]
        weights = [0.40, 0.30, 0.20, 0.10]
        hb = h_blend_simple(subs_4, weights)
        generate_submission(test_ids, hb, "id", "Churn", "submissions/public_hblend.csv")
        print(f"  Saved public_hblend.csv", flush=True)

    print("\nDONE!", flush=True)


if __name__ == "__main__":
    main()
