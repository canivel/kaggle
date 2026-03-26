"""
Research experiment: FT-Transformer (Feature Tokenizer Transformer) for March Madness prediction.

Tests a from-scratch PyTorch FT-Transformer on tabular features with leave-one-season-out CV.
Runnable with: uv run python research_ft_transformer.py
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from features import FeatureBuilder

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """Embeds each numerical feature independently into a d_token-dimensional space."""

    def __init__(self, n_features, d_token):
        super().__init__()
        # Each feature gets its own linear projection from scalar -> d_token
        self.weights = nn.Parameter(torch.empty(n_features, d_token))
        self.biases = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.biases)

    def forward(self, x):
        # x: (batch, n_features)
        # out: (batch, n_features, d_token)
        # Each feature_i: x[:, i] * weights[i] + biases[i]
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for tabular binary classification."""

    def __init__(self, n_features, d_token=64, n_heads=4, n_layers=3, d_ffn=128, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        # Feature tokenizer: each feature -> d_token embedding
        self.feature_tokenizer = FeatureTokenizer(n_features, d_token)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm before head
        self.ln = nn.LayerNorm(d_token)

        # Output head: CLS token -> probability
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x):
        # x: (batch, n_features)
        batch_size = x.size(0)

        # Tokenize features: (batch, n_features, d_token)
        tokens = self.feature_tokenizer(x)

        # Prepend [CLS] token: (batch, 1 + n_features, d_token)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Transformer encoder
        tokens = self.transformer(tokens)

        # Extract [CLS] output
        cls_out = tokens[:, 0, :]
        cls_out = self.ln(cls_out)

        # Predict
        logit = self.head(cls_out).squeeze(-1)
        return logit  # raw logit, apply sigmoid externally


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, X_val, y_val, config, device, verbose=False):
    """Train an FT-Transformer with early stopping. Returns trained model."""
    n_features = X_train.shape[1]
    model = FTTransformer(
        n_features=n_features,
        d_token=config['d_token'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ffn=config.get('d_ffn', config['d_token'] * 2),
        dropout=config.get('dropout', 0.1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    batch_size = 256
    n_samples = X_train_t.size(0)
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(200):
        model.train()
        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            xb = X_train_t[i:i + batch_size]
            yb = y_train_t[i:i + batch_size]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d} | train_loss={epoch_loss/n_batches:.4f} | val_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}", flush=True)
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, X, device):
    """Return predicted probabilities."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 70, flush=True)
    print("FT-Transformer Research Experiment", flush=True)
    print("=" * 70, flush=True)

    device = torch.device('cpu')
    print(f"Device: {device}", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)

    # Load data
    print("\nLoading features...", flush=True)
    fb = FeatureBuilder()
    seasons = list(range(2021, 2026))
    X, y, season_labels = fb.build_training_data(seasons)
    print(f"Data shape: X={X.shape}, y={y.shape}", flush=True)
    print(f"Seasons: {sorted(pd.Series(season_labels).unique())}", flush=True)

    # Fill NaN
    X = X.fillna(0)
    feature_names = X.columns.tolist()
    X_np = X.values.astype(np.float64)
    y_np = y.values.astype(np.float64) if hasattr(y, 'values') else np.array(y, dtype=np.float64)
    season_np = np.array(season_labels)

    # Configs to test
    configs = {
        'A (d=64, h=4, L=3)': {'d_token': 64, 'n_heads': 4, 'n_layers': 3, 'd_ffn': 128, 'dropout': 0.1},
        'B (d=32, h=4, L=2)': {'d_token': 32, 'n_heads': 4, 'n_layers': 2, 'd_ffn': 64, 'dropout': 0.1},
        'C (d=128, h=8, L=4)': {'d_token': 128, 'n_heads': 8, 'n_layers': 4, 'd_ffn': 256, 'dropout': 0.1},
    }

    baseline_brier = 0.0234
    results = {}

    for config_name, config in configs.items():
        print(f"\n{'=' * 70}", flush=True)
        print(f"Config {config_name}", flush=True)
        print(f"{'=' * 70}", flush=True)

        season_briers = {}
        all_preds = []
        all_trues = []

        for test_season in seasons:
            try:
                t0 = time.time()
                print(f"\n  Season {test_season} (test) ...", flush=True)

                # Split
                train_mask = season_np != test_season
                test_mask = season_np == test_season

                X_train_raw = X_np[train_mask]
                y_train_raw = y_np[train_mask]
                X_test_raw = X_np[test_mask]
                y_test_raw = y_np[test_mask]

                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_raw)
                X_test_scaled = scaler.transform(X_test_raw)

                # 90/10 train/val split within training fold (deterministic)
                n_train = len(X_train_scaled)
                n_val = max(1, int(n_train * 0.1))
                rng = np.random.RandomState(42 + test_season)
                indices = rng.permutation(n_train)
                val_idx = indices[:n_val]
                trn_idx = indices[n_val:]

                X_trn = X_train_scaled[trn_idx]
                y_trn = y_train_raw[trn_idx]
                X_val = X_train_scaled[val_idx]
                y_val = y_train_raw[val_idx]

                print(f"    Train: {len(X_trn)}, Val: {len(X_val)}, Test: {len(X_test_scaled)}", flush=True)

                # Set seeds for reproducibility
                torch.manual_seed(42)
                np.random.seed(42)

                # Train
                model = train_model(X_trn, y_trn, X_val, y_val, config, device, verbose=True)

                # Predict
                preds = predict(model, X_test_scaled, device)
                preds = np.clip(preds, 0.001, 0.999)

                # Brier score
                brier = np.mean((y_test_raw - preds) ** 2)
                season_briers[test_season] = brier
                all_preds.extend(preds.tolist())
                all_trues.extend(y_test_raw.tolist())

                elapsed = time.time() - t0
                print(f"    Brier={brier:.4f}  ({elapsed:.1f}s)", flush=True)

            except Exception as e:
                print(f"    ERROR in season {test_season}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                season_briers[test_season] = float('nan')

        # Overall
        overall_brier = np.mean((np.array(all_trues) - np.array(all_preds)) ** 2) if all_preds else float('nan')
        results[config_name] = {
            'season_briers': season_briers,
            'overall_brier': overall_brier,
        }

        print(f"\n  Config {config_name} results:", flush=True)
        for s, b in sorted(season_briers.items()):
            print(f"    {s}: {b:.4f}", flush=True)
        print(f"    Overall: {overall_brier:.4f}", flush=True)
        diff = overall_brier - baseline_brier
        print(f"    vs baseline (0.0234): {diff:+.4f} ({'better' if diff < 0 else 'worse'})", flush=True)

    # Summary
    print(f"\n{'=' * 70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"{'Config':<30s} {'Overall Brier':>14s} {'vs 0.0234':>10s}", flush=True)
    print(f"{'-' * 56}", flush=True)
    print(f"{'Baseline (LightGBM)':<30s} {baseline_brier:>14.4f} {'':>10s}", flush=True)
    for config_name, res in results.items():
        b = res['overall_brier']
        diff = b - baseline_brier
        marker = 'BETTER' if diff < 0 else 'worse'
        print(f"{config_name:<30s} {b:>14.4f} {diff:>+10.4f} {marker}", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    best_config = min(results, key=lambda k: results[k]['overall_brier'])
    best_brier = results[best_config]['overall_brier']
    print(f"Best config: {best_config} with Brier={best_brier:.4f}", flush=True)
    if best_brier < baseline_brier:
        print(f"FT-Transformer BEATS baseline by {baseline_brier - best_brier:.4f}!", flush=True)
    else:
        print(f"FT-Transformer does not beat baseline (gap: {best_brier - baseline_brier:.4f})", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    try:
        run_experiment()
    except Exception as e:
        print(f"FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
