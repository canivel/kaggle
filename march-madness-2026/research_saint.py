"""
Research experiment: SAINT-style transformer for March Madness prediction.

SAINT = Self-Attention and Intersample Attention Transformer.
Each layer alternates:
  1. Self-attention across feature tokens (captures feature interactions)
  2. Intersample attention across batch samples (captures sample similarities)

Usage: uv run python research_saint.py
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from features import FeatureBuilder

# ---------------------------------------------------------------------------
# SAINT model
# ---------------------------------------------------------------------------

class SAINTLayer(nn.Module):
    """One SAINT layer: self-attention over features, then intersample attention."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        # Self-attention across feature tokens
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        # Intersample attention across batch dimension
        self.inter_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )

    def forward(self, x):
        # x: (B, N_feat, D)
        # 1) Self-attention over features
        x = self.self_attn(x)
        # 2) Intersample attention: for each feature position attend across samples
        B, N, D = x.shape
        x = x.permute(1, 0, 2)  # (N_feat, B, D)
        x = self.inter_attn(x)
        x = x.permute(1, 0, 2)  # (B, N_feat, D)
        return x


class SAINTModel(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Embed each scalar feature into d_model dimensions
        self.feature_embed = nn.Linear(1, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional embedding for feature tokens + CLS
        self.pos_embed = nn.Parameter(torch.randn(1, n_features + 1, d_model) * 0.02)

        # SAINT layers
        self.layers = nn.ModuleList([
            SAINTLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Classification head on CLS token
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: (B, n_features)
        B = x.size(0)

        # Each feature becomes a token: (B, n_features, 1) -> (B, n_features, d_model)
        x = x.unsqueeze(-1)
        x = self.feature_embed(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_features+1, d_model)

        # Add positional embedding
        x = x + self.pos_embed

        # SAINT layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # CLS token output
        cls_out = x[:, 0, :]
        logits = self.head(cls_out).squeeze(-1)
        return logits  # raw logits; apply sigmoid outside


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, X_val, y_val, config, verbose=False):
    """Train a SAINT model and return best val predictions and loss."""
    n_features = X_train.shape[1]
    model = SAINTModel(
        n_features=n_features,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.1),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 128), shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_preds = None
    patience_counter = 0
    patience = config.get("patience", 20)
    epochs = config.get("epochs", 150)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = torch.sigmoid(val_preds).clamp(1e-7, 1-1e-7).numpy().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1:3d}  train_loss={train_loss/n_batches:.4f}  val_loss={val_loss:.4f}", flush=True)

        if patience_counter >= patience:
            if verbose:
                print(f"    Early stop at epoch {epoch+1}", flush=True)
            break

    return best_preds, best_val_loss


def brier_score(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ---------------------------------------------------------------------------
# Evaluation: leave-one-season-out CV
# ---------------------------------------------------------------------------

def evaluate_config(X, y, seasons, config, label):
    print(f"\n{'='*60}", flush=True)
    print(f"Config: {label}", flush=True)
    print(f"  d_model={config['d_model']}, n_heads={config['n_heads']}, "
          f"n_layers={config['n_layers']}, batch_size={config.get('batch_size', 128)}", flush=True)
    print(f"{'='*60}", flush=True)

    test_seasons = [2021, 2022, 2023, 2024, 2025]
    season_briers = {}

    for test_season in test_seasons:
        test_mask = seasons == test_season
        train_mask = ~test_mask

        X_tr = X[train_mask].values if hasattr(X, "values") else X[train_mask]
        y_tr = y[train_mask] if isinstance(y, np.ndarray) else y[train_mask].values
        X_te = X[test_mask].values if hasattr(X, "values") else X[test_mask]
        y_te = y[test_mask] if isinstance(y, np.ndarray) else y[test_mask].values

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # 90/10 train/val split
        n_val = max(1, int(len(X_tr_s) * 0.1))
        indices = np.random.permutation(len(X_tr_s))
        val_idx = indices[:n_val]
        tr_idx = indices[n_val:]

        X_fit = X_tr_s[tr_idx]
        y_fit = y_tr[tr_idx]
        X_val = X_tr_s[val_idx]
        y_val = y_tr[val_idx]

        print(f"\n  Season {test_season}: train={len(X_fit)}, val={len(X_val)}, test={len(X_te_s)}", flush=True)

        # Train model
        _, _ = train_model(X_fit, y_fit, X_val, y_val, config, verbose=True)

        # Retrain on full training data using val split for early stopping
        # (use same split for stopping criterion)
        preds, _ = train_model(X_tr_s[tr_idx], y_tr[tr_idx], X_tr_s[val_idx], y_tr[val_idx], config, verbose=False)

        # Actually we need test predictions - train on full train, evaluate on test
        # Use the val split for early stopping, then predict on test set
        model = SAINTModel(
            n_features=X_tr_s.shape[1],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            dropout=config.get("dropout", 0.1),
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )
        criterion = nn.BCEWithLogitsLoss()

        train_ds = TensorDataset(
            torch.tensor(X_fit, dtype=torch.float32),
            torch.tensor(y_fit, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 128), shuffle=True)

        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        X_te_t = torch.tensor(X_te_s, dtype=torch.float32)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience = config.get("patience", 20)

        for epoch in range(config.get("epochs", 150)):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                vl = criterion(model(X_val_t), y_val_t).item()

            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # Load best model and predict on test
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_preds = torch.sigmoid(model(X_te_t)).clamp(1e-7, 1-1e-7).numpy()

        test_preds = np.clip(test_preds, 0.001, 0.999)
        bs = brier_score(y_te, test_preds)
        season_briers[test_season] = bs
        print(f"  Season {test_season} Brier: {bs:.4f}", flush=True)

    overall = np.mean(list(season_briers.values()))
    print(f"\n  Overall mean Brier: {overall:.4f}", flush=True)
    return overall, season_briers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("SAINT-style Transformer for March Madness", flush=True)
    print("=" * 60, flush=True)

    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("\nLoading data...", flush=True)
    fb = FeatureBuilder()
    X, y, seasons = fb.build_training_data(list(range(2015, 2026)))
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
    print(f"Seasons: {sorted(seasons.unique())}", flush=True)

    y_arr = y.values if hasattr(y, "values") else np.array(y)
    s_arr = seasons.values if hasattr(seasons, "values") else np.array(seasons)

    BASELINE = 0.0234

    # Config 1: Default
    config_default = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.1,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "epochs": 150,
        "patience": 20,
    }

    # Config 2: Larger
    config_larger = {
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 3,
        "dropout": 0.1,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "epochs": 150,
        "patience": 20,
    }

    results = {}

    try:
        overall, per_season = evaluate_config(X, y_arr, s_arr, config_default, "Default (d64, h4, L2)")
        results["Default"] = overall
    except Exception as e:
        print(f"\nERROR in default config: {e}", flush=True)
        import traceback
        traceback.print_exc()
        results["Default"] = None

    try:
        overall, per_season = evaluate_config(X, y_arr, s_arr, config_larger, "Larger (d128, h8, L3)")
        results["Larger"] = overall
    except Exception as e:
        print(f"\nERROR in larger config: {e}", flush=True)
        import traceback
        traceback.print_exc()
        results["Larger"] = None

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Baseline Brier:  {BASELINE:.4f}", flush=True)
    for name, score in results.items():
        if score is not None:
            delta = score - BASELINE
            flag = "BETTER" if delta < 0 else "WORSE"
            print(f"  {name:20s}  Brier={score:.4f}  delta={delta:+.4f}  {flag}", flush=True)
        else:
            print(f"  {name:20s}  FAILED", flush=True)

    best_name = min((k for k, v in results.items() if v is not None), key=lambda k: results[k], default=None)
    if best_name:
        print(f"\nBest config: {best_name} with Brier={results[best_name]:.4f}", flush=True)
        if results[best_name] < BASELINE:
            print("Result: IMPROVEMENT over baseline!", flush=True)
        else:
            print("Result: Does not beat baseline.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
