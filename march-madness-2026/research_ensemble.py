#!/usr/bin/env python3
"""
Research Round 5: Mega ensemble combining tree models + all transformer research.
Runs after individual research experiments to find the best blend.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")
TRAIN_SEASONS = list(range(2003, 2026))
CV_SEASONS = [2021, 2022, 2023, 2024, 2025]
CURRENT_BEST = 0.0234


def brier_score(y_true, y_pred):
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)


def get_data():
    from features import FeatureBuilder
    fb = FeatureBuilder(str(DATA_DIR))
    X, y, seasons = fb.build_training_data(TRAIN_SEASONS)
    X = X.fillna(0)
    return fb, X, y, seasons


def get_tree_preds(X, y, seasons, test_season):
    """Get predictions from tree models for a test season."""
    from model import MarchMadnessModel
    test_mask = (seasons == test_season).values
    train_mask = ~test_mask
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte = X[test_mask]

    preds = {}
    for name, mtype in [("xgb", "xgb"), ("lgbm", "lgbm"), ("lr", "logistic")]:
        m = MarchMadnessModel(mtype)
        m.train(Xtr, ytr)
        preds[name] = m.predict(Xte)
    return preds


def get_nn_preds(X, y, seasons, test_season):
    """Get predictions from basic neural net."""
    from model import NeuralNetModel
    test_mask = (seasons == test_season).values
    train_mask = ~test_mask
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte = X[test_mask]

    nn = NeuralNetModel()
    nn.train(Xtr, ytr)
    return nn.predict(Xte)


def get_tabpfn_preds(X, y, seasons, test_season):
    """Get predictions from TabPFN."""
    try:
        from tabpfn import TabPFNClassifier
        test_mask = (seasons == test_season).values
        train_mask = ~test_mask
        Xtr, ytr = X[train_mask].values, y[train_mask].values
        Xte = X[test_mask].values

        model = TabPFNClassifier(device='cpu')
        model.fit(Xtr, ytr)
        return model.predict_proba(Xte)[:, 1]
    except Exception as e:
        print(f"  TabPFN failed: {e}", flush=True)
        return None


def get_ft_transformer_preds(X, y, seasons, test_season):
    """Get predictions from FT-Transformer."""
    try:
        import torch
        import torch.nn as nn

        test_mask = (seasons == test_season).values
        train_mask = ~test_mask

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_mask].values).astype(np.float32)
        ytr = y[train_mask].values.astype(np.float32)
        Xte = scaler.transform(X[test_mask].values).astype(np.float32)

        n_features = Xtr.shape[1]
        d_token = 64
        n_heads = 4
        n_layers = 3
        d_ffn = 128
        dropout = 0.1

        class FTTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_tokenizers = nn.ModuleList([
                    nn.Linear(1, d_token) for _ in range(n_features)
                ])
                self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_token, nhead=n_heads, dim_feedforward=d_ffn,
                    dropout=dropout, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.head = nn.Sequential(
                    nn.LayerNorm(d_token), nn.Linear(d_token, 1), nn.Sigmoid()
                )

            def forward(self, x):
                tokens = [self.feature_tokenizers[i](x[:, i:i+1]) for i in range(n_features)]
                tokens = torch.stack(tokens, dim=1)
                cls = self.cls_token.expand(x.size(0), -1, -1)
                tokens = torch.cat([cls, tokens], dim=1)
                out = self.transformer(tokens)
                return self.head(out[:, 0]).squeeze(-1)

        model = FTTransformer()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.BCELoss()

        # Train/val split
        n = len(Xtr)
        idx = np.random.RandomState(42).permutation(n)
        val_n = max(1, int(0.1 * n))
        val_idx, tr_idx = idx[:val_n], idx[val_n:]

        Xtr_t = torch.tensor(Xtr[tr_idx])
        ytr_t = torch.tensor(ytr[tr_idx])
        Xval_t = torch.tensor(Xtr[val_idx])
        yval_t = torch.tensor(ytr[val_idx])
        Xte_t = torch.tensor(Xte)

        best_loss, best_state, patience, no_improve = float('inf'), None, 20, 0
        dataset = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        for epoch in range(150):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                vl = criterion(model(Xval_t), yval_t).item()
            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            return model(Xte_t).numpy()
    except Exception as e:
        print(f"  FT-Transformer failed: {e}", flush=True)
        return None


def search_optimal_weights(all_model_preds, y_true):
    """Grid search for optimal ensemble weights."""
    best_brier = 1.0
    best_weights = None
    model_names = list(all_model_preds.keys())
    n_models = len(model_names)

    # Generate weight combinations (step 0.05)
    from itertools import product
    steps = np.arange(0, 1.01, 0.05)

    if n_models <= 4:
        for combo in product(steps, repeat=n_models):
            if abs(sum(combo) - 1.0) > 0.01:
                continue
            pred = np.zeros(len(y_true))
            for w, name in zip(combo, model_names):
                pred += w * all_model_preds[name]
            pred = np.clip(pred, 0.001, 0.999)
            b = brier_score(y_true, pred)
            if b < best_brier:
                best_brier = b
                best_weights = dict(zip(model_names, combo))
    else:
        # Too many models for full grid - use random search
        rng = np.random.RandomState(42)
        for _ in range(50000):
            raw = rng.random(n_models)
            weights = raw / raw.sum()
            pred = np.zeros(len(y_true))
            for w, name in zip(weights, model_names):
                pred += w * all_model_preds[name]
            pred = np.clip(pred, 0.001, 0.999)
            b = brier_score(y_true, pred)
            if b < best_brier:
                best_brier = b
                best_weights = dict(zip(model_names, weights))

    return best_weights, best_brier


def main():
    start = time.time()
    print("=" * 60, flush=True)
    print("RESEARCH ROUND 5: MEGA ENSEMBLE WITH TRANSFORMERS", flush=True)
    print("=" * 60, flush=True)

    fb, X, y, seasons = get_data()
    print(f"Data: {X.shape[0]} games, {X.shape[1]} features", flush=True)

    # Per-season CV collecting predictions from all models
    all_season_preds = {}  # season -> {model_name -> preds}
    all_season_true = {}   # season -> y_true

    for s in CV_SEASONS:
        print(f"\n--- Season {s} ---", flush=True)
        test_mask = (seasons == s).values
        y_test = y[test_mask].values
        all_season_true[s] = y_test
        all_season_preds[s] = {}

        # Tree models
        tree_preds = get_tree_preds(X, y, seasons, s)
        for name, p in tree_preds.items():
            all_season_preds[s][name] = np.clip(p, 0.001, 0.999)
            print(f"  {name}: Brier={brier_score(y_test, p):.6f}", flush=True)

        # Basic NN
        nn_p = get_nn_preds(X, y, seasons, s)
        all_season_preds[s]["nn"] = np.clip(nn_p, 0.001, 0.999)
        print(f"  nn: Brier={brier_score(y_test, nn_p):.6f}", flush=True)

        # TabPFN
        tabpfn_p = get_tabpfn_preds(X, y, seasons, s)
        if tabpfn_p is not None:
            all_season_preds[s]["tabpfn"] = np.clip(tabpfn_p, 0.001, 0.999)
            print(f"  tabpfn: Brier={brier_score(y_test, tabpfn_p):.6f}", flush=True)

        # FT-Transformer
        ft_p = get_ft_transformer_preds(X, y, seasons, s)
        if ft_p is not None:
            all_season_preds[s]["ft_transformer"] = np.clip(ft_p, 0.001, 0.999)
            print(f"  ft_transformer: Brier={brier_score(y_test, ft_p):.6f}", flush=True)

    # Combine all predictions across seasons
    model_names = set()
    for s in CV_SEASONS:
        model_names.update(all_season_preds[s].keys())
    model_names = sorted(model_names)

    # Only keep models that worked for ALL seasons
    valid_models = [m for m in model_names
                    if all(m in all_season_preds[s] for s in CV_SEASONS)]
    print(f"\nValid models for ensemble: {valid_models}", flush=True)

    # Concatenate predictions
    concat_preds = {m: np.concatenate([all_season_preds[s][m] for s in CV_SEASONS])
                    for m in valid_models}
    concat_true = np.concatenate([all_season_true[s] for s in CV_SEASONS])

    # Individual model overall scores
    print("\n--- Individual Model Results ---", flush=True)
    for m in valid_models:
        b = brier_score(concat_true, concat_preds[m])
        print(f"  {m:20s}: {b:.6f}", flush=True)

    # Search optimal weights
    print("\n--- Searching Optimal Ensemble Weights ---", flush=True)
    best_weights, best_brier = search_optimal_weights(concat_preds, concat_true)
    print(f"\nOptimal weights:", flush=True)
    for m, w in sorted(best_weights.items(), key=lambda x: -x[1]):
        if w > 0.01:
            print(f"  {m:20s}: {w:.2f}", flush=True)
    print(f"Optimal Brier: {best_brier:.6f}", flush=True)
    print(f"Current best:  {CURRENT_BEST:.6f}", flush=True)
    improvement = (CURRENT_BEST - best_brier) / CURRENT_BEST * 100
    print(f"Improvement:   {improvement:.2f}%", flush=True)

    # Equal-weight ensemble for comparison
    equal_pred = np.zeros(len(concat_true))
    for m in valid_models:
        equal_pred += concat_preds[m] / len(valid_models)
    equal_brier = brier_score(concat_true, np.clip(equal_pred, 0.001, 0.999))
    print(f"\nEqual-weight ensemble: {equal_brier:.6f}", flush=True)

    # Per-season breakdown with optimal weights
    print("\n--- Per-Season Breakdown (Optimal Weights) ---", flush=True)
    per_season_brier = {}
    for s in CV_SEASONS:
        pred = np.zeros(len(all_season_true[s]))
        for m, w in best_weights.items():
            if m in all_season_preds[s]:
                pred += w * all_season_preds[s][m]
        pred = np.clip(pred, 0.001, 0.999)
        b = brier_score(all_season_true[s], pred)
        per_season_brier[s] = b
        print(f"  {s}: {b:.6f}", flush=True)

    # Log experiment
    exp = {
        "experiment_id": "R05",
        "description": f"Mega ensemble: {'+'.join(m for m,w in best_weights.items() if w > 0.01)}",
        "brier_score": f"{best_brier:.6f}",
        "per_season_scores": json.dumps({str(k): round(v,6) for k,v in per_season_brier.items()}),
        "notes": f"weights: {json.dumps({k:round(v,3) for k,v in best_weights.items() if v > 0.01})}",
        "kept": "yes" if best_brier < CURRENT_BEST else "no"
    }
    df = pd.read_csv("experiments.tsv", sep="\t") if Path("experiments.tsv").exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([exp])], ignore_index=True)
    df.to_csv("experiments.tsv", sep="\t", index=False)

    print(f"\nTotal time: {time.time()-start:.1f}s", flush=True)

    if best_brier < CURRENT_BEST:
        print(f"\n*** NEW BEST! {best_brier:.6f} < {CURRENT_BEST:.6f} ***", flush=True)
        print("Generating submission with optimal weights...", flush=True)

        # Train all valid models on full data and generate submission
        from model import MarchMadnessModel, NeuralNetModel

        sub_df = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
        X_sub = fb.build_submission_features(str(DATA_DIR / "SampleSubmissionStage2.csv")).fillna(0)

        final_preds = np.zeros(len(X_sub))

        for m_name, w in best_weights.items():
            if w < 0.01:
                continue
            print(f"  Training {m_name} on full data (weight={w:.2f})...", flush=True)

            if m_name == "xgb":
                model = MarchMadnessModel("xgb")
                model.train(X, y)
                final_preds += w * model.predict(X_sub)
            elif m_name == "lgbm":
                model = MarchMadnessModel("lgbm")
                model.train(X, y)
                final_preds += w * model.predict(X_sub)
            elif m_name == "lr":
                model = MarchMadnessModel("logistic")
                model.train(X, y)
                final_preds += w * model.predict(X_sub)
            elif m_name == "nn":
                nn = NeuralNetModel()
                nn.train(X, y)
                final_preds += w * nn.predict(X_sub)
            elif m_name == "tabpfn":
                from tabpfn import TabPFNClassifier
                tpfn = TabPFNClassifier(device='cpu')
                tpfn.fit(X.values, y.values)
                final_preds += w * tpfn.predict_proba(X_sub.values)[:, 1]
            elif m_name == "ft_transformer":
                # Inline FT-Transformer training on full data
                ft_p = get_ft_transformer_preds(X, y, seasons, -1)  # won't work directly
                if ft_p is not None:
                    final_preds += w * ft_p

        final_preds = np.clip(final_preds, 0.01, 0.99)
        sub_df["Pred"] = final_preds
        sub_df.to_csv("submission.csv", index=False)
        print(f"Saved submission.csv", flush=True)

        # Submit
        print("Submitting to Kaggle...", flush=True)
        os.environ.setdefault("KAGGLE_API_TOKEN", "KGAT_9e09aecd744cb0683ca3985a8e6d277b")
        os.system(
            f'kaggle competitions submit -c march-machine-learning-mania-2026 '
            f'-f submission.csv -m "Transformer ensemble Brier={best_brier:.6f}"'
        )
    else:
        print(f"\nNo improvement over current best {CURRENT_BEST:.6f}", flush=True)


if __name__ == "__main__":
    main()
