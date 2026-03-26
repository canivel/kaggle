# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "numpy",
#   "torch",
#   "scikit-learn",
# ]
# ///
"""
Research experiment: Temporal/sequence approach for March Madness prediction.
Treats each team's season as a sequence of games, encodes with a Transformer,
then predicts tournament matchup outcomes.

Current best CV Brier on 2021-2025: 0.0234
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "data"
CV_SEASONS = [2021, 2022, 2023, 2024, 2025]

# -- Per-game feature columns we extract --------------------------------------
# For each game a team plays, we build a feature vector from these raw stats.
GAME_FEATURE_NAMES = [
    "score_diff",
    "fg_pct", "fg3_pct", "ft_pct",
    "opp_fg_pct", "opp_fg3_pct", "opp_ft_pct",
    "off_reb", "def_reb", "assists", "turnovers", "steals", "blocks",
    "opp_off_reb", "opp_def_reb", "opp_assists", "opp_turnovers", "opp_steals", "opp_blocks",
    "win", "home", "away", "neutral",
    "day_num",
    "elo_after",
]
N_GAME_FEATURES = len(GAME_FEATURE_NAMES)

MAX_SEQ_LEN = 40  # pad/truncate to this length


# -----------------------------------------------------------------------------
# 1. Data loading
# -----------------------------------------------------------------------------

def load_detailed_results():
    """Load men's and women's regular season detailed results."""
    dfs = []
    for prefix in ["M", "W"]:
        path = f"{DATA_DIR}/{prefix}RegularSeasonDetailedResults.csv"
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"  Loaded {path}: {len(df)} games, seasons {df['Season'].min()}-{df['Season'].max()}", flush=True)
        except FileNotFoundError:
            print(f"  {path} not found, skipping", flush=True)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_tourney_results():
    """Load men's and women's tournament compact results."""
    dfs = []
    for prefix in ["M", "W"]:
        path = f"{DATA_DIR}/{prefix}NCAATourneyCompactResults.csv"
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except FileNotFoundError:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_seeds():
    """Load tournament seeds."""
    dfs = []
    for prefix in ["M", "W"]:
        path = f"{DATA_DIR}/{prefix}NCAATourneySeeds.csv"
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except FileNotFoundError:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# -----------------------------------------------------------------------------
# 2. Per-game feature extraction with running Elo
# -----------------------------------------------------------------------------

def safe_pct(made, att):
    """Compute shooting percentage, returning 0 when attempts is 0."""
    return made / att if att > 0 else 0.0


def build_team_game_sequences(detailed_df):
    """
    For each (season, team), build a chronologically-ordered list of per-game
    feature vectors.  Also computes a running Elo for each team within season.

    Returns: dict[(season, team_id)] -> np.ndarray of shape (n_games, N_GAME_FEATURES)
    """
    # Initialise Elo ratings: carry over across seasons with regression to mean
    elo = {}
    K = 20
    HOME_ADV = 100

    sequences = {}

    # Process season by season, day by day
    seasons = sorted(detailed_df["Season"].unique())
    for season in seasons:
        # Regress all existing Elos toward 1500
        for tid in list(elo.keys()):
            elo[tid] = 1500 + 0.75 * (elo.get(tid, 1500) - 1500)

        season_df = detailed_df[detailed_df["Season"] == season].sort_values("DayNum")

        # Temporary storage: list of feature dicts per team
        season_games = {}  # team_id -> list of feature vectors

        for _, row in season_df.iterrows():
            w_id = int(row["WTeamID"])
            l_id = int(row["LTeamID"])
            day = int(row["DayNum"])

            # Current Elo before game
            w_elo = elo.get(w_id, 1500)
            l_elo = elo.get(l_id, 1500)

            # Update Elo
            loc = row["WLoc"]
            w_adj = w_elo + (HOME_ADV if loc == "H" else (-HOME_ADV if loc == "A" else 0))
            expected_w = 1.0 / (1.0 + 10 ** ((l_elo - w_adj) / 400))
            elo[w_id] = w_elo + K * (1.0 - expected_w)
            elo[l_id] = l_elo + K * (0.0 - (1.0 - expected_w))

            # -- Winner's feature vector --
            w_features = [
                row["WScore"] - row["LScore"],                       # score_diff
                safe_pct(row["WFGM"], row["WFGA"]),                  # fg_pct
                safe_pct(row["WFGM3"], row["WFGA3"]),                # fg3_pct
                safe_pct(row["WFTM"], row["WFTA"]),                  # ft_pct
                safe_pct(row["LFGM"], row["LFGA"]),                  # opp_fg_pct
                safe_pct(row["LFGM3"], row["LFGA3"]),                # opp_fg3_pct
                safe_pct(row["LFTM"], row["LFTA"]),                  # opp_ft_pct
                row["WOR"], row["WDR"],                              # off_reb, def_reb
                row["WAst"], row["WTO"], row["WStl"], row["WBlk"],   # assists, TO, steals, blocks
                row["LOR"], row["LDR"],                              # opp off_reb, def_reb
                row["LAst"], row["LTO"], row["LStl"], row["LBlk"],   # opp assists, TO, steals, blocks
                1.0,                                                 # win
                1.0 if loc == "H" else 0.0,                         # home
                1.0 if loc == "A" else 0.0,                         # away (winner played away)
                1.0 if loc == "N" else 0.0,                         # neutral
                day,                                                 # day_num
                elo[w_id],                                           # elo after game
            ]

            # -- Loser's feature vector --
            l_loc_map = {"H": "A", "A": "H", "N": "N"}
            l_loc = l_loc_map.get(loc, "N")
            l_features = [
                row["LScore"] - row["WScore"],                       # score_diff (negative)
                safe_pct(row["LFGM"], row["LFGA"]),                  # fg_pct
                safe_pct(row["LFGM3"], row["LFGA3"]),                # fg3_pct
                safe_pct(row["LFTM"], row["LFTA"]),                  # ft_pct
                safe_pct(row["WFGM"], row["WFGA"]),                  # opp_fg_pct
                safe_pct(row["WFGM3"], row["WFGA3"]),                # opp_fg3_pct
                safe_pct(row["WFTM"], row["WFTA"]),                  # opp_ft_pct
                row["LOR"], row["LDR"],
                row["LAst"], row["LTO"], row["LStl"], row["LBlk"],
                row["WOR"], row["WDR"],
                row["WAst"], row["WTO"], row["WStl"], row["WBlk"],
                0.0,                                                 # loss
                1.0 if l_loc == "H" else 0.0,
                1.0 if l_loc == "A" else 0.0,
                1.0 if l_loc == "N" else 0.0,
                day,
                elo[l_id],
            ]

            season_games.setdefault(w_id, []).append(w_features)
            season_games.setdefault(l_id, []).append(l_features)

        # Store as numpy arrays keyed by (season, team)
        for tid, games in season_games.items():
            sequences[(season, tid)] = np.array(games, dtype=np.float32)

    return sequences


# -----------------------------------------------------------------------------
# 3. Normalisation & padding
# -----------------------------------------------------------------------------

def fit_scaler(sequences, exclude_seasons=None):
    """Fit a StandardScaler on all game features, optionally excluding seasons."""
    all_feats = []
    for (season, _), arr in sequences.items():
        if exclude_seasons and season in exclude_seasons:
            continue
        all_feats.append(arr)
    if not all_feats:
        return StandardScaler()
    stacked = np.concatenate(all_feats, axis=0)
    scaler = StandardScaler()
    scaler.fit(stacked)
    return scaler


def pad_sequence(arr, max_len, n_features):
    """Pad or truncate a (n_games, n_features) array to (max_len, n_features).
    Returns (padded_array, mask) where mask is 1 for real games, 0 for padding."""
    n = arr.shape[0]
    if n >= max_len:
        # Take the last max_len games (most recent)
        return arr[-max_len:], np.ones(max_len, dtype=np.float32)
    else:
        padded = np.zeros((max_len, n_features), dtype=np.float32)
        mask = np.zeros(max_len, dtype=np.float32)
        padded[:n] = arr
        mask[:n] = 1.0
        return padded, mask


# -----------------------------------------------------------------------------
# 4. Dataset
# -----------------------------------------------------------------------------

class TourneyMatchupDataset(Dataset):
    """
    Each item is a tournament game: two team sequences + label (1 if team1 wins).
    We always put the lower-ID team as team1 for consistency.
    """

    def __init__(self, tourney_df, sequences, scaler, seasons):
        self.items = []
        missing = 0

        for _, row in tourney_df.iterrows():
            season = int(row["Season"])
            if season not in seasons:
                continue

            w_id = int(row["WTeamID"])
            l_id = int(row["LTeamID"])

            # Canonical ordering: lower ID = team1
            if w_id < l_id:
                t1, t2, label = w_id, l_id, 1.0
            else:
                t1, t2, label = l_id, w_id, 0.0

            key1 = (season, t1)
            key2 = (season, t2)

            if key1 not in sequences or key2 not in sequences:
                missing += 1
                continue

            # Normalise
            seq1 = scaler.transform(sequences[key1])
            seq2 = scaler.transform(sequences[key2])

            # Pad
            seq1_pad, mask1 = pad_sequence(seq1, MAX_SEQ_LEN, N_GAME_FEATURES)
            seq2_pad, mask2 = pad_sequence(seq2, MAX_SEQ_LEN, N_GAME_FEATURES)

            self.items.append((
                torch.tensor(seq1_pad, dtype=torch.float32),
                torch.tensor(mask1, dtype=torch.float32),
                torch.tensor(seq2_pad, dtype=torch.float32),
                torch.tensor(mask2, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
            ))

        if missing > 0:
            print(f"    Warning: {missing} tourney games skipped (missing sequences)", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# -----------------------------------------------------------------------------
# 5. Model
# -----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TemporalTeamEncoder(nn.Module):
    """
    Per-game features -> Linear projection -> Positional Encoding ->
    TransformerEncoder -> masked mean pool -> team embedding
    """

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, embed_dim=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=MAX_SEQ_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x: (batch, seq_len, input_dim)
        mask: (batch, seq_len) - 1 for real, 0 for padding
        Returns: (batch, embed_dim)
        """
        # Project input features to d_model
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = self.pos_enc(h)

        # Transformer expects key_padding_mask: True = ignore
        pad_mask = (mask == 0)  # (batch, seq_len)
        h = self.transformer(h, src_key_padding_mask=pad_mask)  # (batch, seq_len, d_model)

        # Masked mean pooling
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # (batch, d_model)

        h = self.dropout(h)
        h = self.output_proj(h)  # (batch, embed_dim)
        return h


class TemporalMatchupPredictor(nn.Module):
    """
    team1_embedding, team2_embedding -> concat + diff + product -> MLP -> sigmoid
    """

    def __init__(self, embed_dim=32, hidden_dim=64, dropout=0.1):
        super().__init__()
        # Input: concat (2*embed), diff (embed), element-wise product (embed) = 4*embed
        self.mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, emb1, emb2):
        """
        emb1, emb2: (batch, embed_dim)
        Returns: (batch,) probabilities
        """
        combined = torch.cat([emb1, emb2, emb1 - emb2, emb1 * emb2], dim=-1)
        logit = self.mlp(combined).squeeze(-1)
        return torch.sigmoid(logit).clamp(1e-7, 1 - 1e-7)


class TemporalModel(nn.Module):
    """Full model: encoder + predictor."""

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, embed_dim=32, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = TemporalTeamEncoder(
            input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, embed_dim,
        )
        self.predictor = TemporalMatchupPredictor(embed_dim, hidden_dim, dropout)

    def forward(self, seq1, mask1, seq2, mask2):
        emb1 = self.encoder(seq1, mask1)
        emb2 = self.encoder(seq2, mask2)
        return self.predictor(emb1, emb2)


# -----------------------------------------------------------------------------
# 6. Training loop
# -----------------------------------------------------------------------------

def train_model(model, train_dataset, val_dataset, epochs=100, lr=1e-3, batch_size=64):
    """Train model and return validation predictions + labels."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # -- Train --
        model.train()
        train_loss = 0.0
        n_train = 0
        for seq1, mask1, seq2, mask2, labels in train_loader:
            optimizer.zero_grad()
            preds = model(seq1, mask1, seq2, mask2)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)
            n_train += len(labels)

        scheduler.step()

        # -- Validate --
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            for seq1, mask1, seq2, mask2, labels in val_loader:
                preds = model(seq1, mask1, seq2, mask2)
                loss = criterion(preds, labels)
                val_loss += loss.item() * len(labels)
                n_val += len(labels)

        avg_train = train_loss / max(n_train, 1)
        avg_val = val_loss / max(n_val, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1:3d}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  best={best_val_loss:.4f}", flush=True)

        if patience_counter >= patience:
            print(f"      Early stop at epoch {epoch+1}", flush=True)
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Get final val predictions
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        for seq1, mask1, seq2, mask2, labels in val_loader:
            preds = model(seq1, mask1, seq2, mask2)
            all_preds.append(preds.numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# -----------------------------------------------------------------------------
# 7. Leave-one-season-out CV
# -----------------------------------------------------------------------------

def run_cv():
    print("=" * 70, flush=True)
    print("TEMPORAL/SEQUENCE MODEL FOR MARCH MADNESS PREDICTION", flush=True)
    print("=" * 70, flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    detailed_df = load_detailed_results()
    tourney_df = load_tourney_results()

    if detailed_df.empty or tourney_df.empty:
        print("ERROR: Could not load required data files.", flush=True)
        return

    print(f"\nTotal detailed regular season games: {len(detailed_df)}", flush=True)
    print(f"Total tournament games: {len(tourney_df)}", flush=True)

    # Build game sequences with running Elo
    print("\nBuilding per-game sequences with running Elo...", flush=True)
    sequences = build_team_game_sequences(detailed_df)
    print(f"  Built sequences for {len(sequences)} (season, team) pairs", flush=True)

    # Show sequence length stats
    seq_lens = [arr.shape[0] for arr in sequences.values()]
    print(f"  Sequence lengths: min={min(seq_lens)}, median={int(np.median(seq_lens))}, "
          f"max={max(seq_lens)}, mean={np.mean(seq_lens):.1f}", flush=True)

    # Determine which seasons have detailed results
    detailed_seasons = set(detailed_df["Season"].unique())
    tourney_seasons_avail = set(tourney_df["Season"].unique()) & detailed_seasons
    print(f"  Seasons with both detailed results and tourney data: "
          f"{min(tourney_seasons_avail)}-{max(tourney_seasons_avail)}", flush=True)

    # Filter to seasons where we have detailed results
    valid_cv_seasons = [s for s in CV_SEASONS if s in tourney_seasons_avail]
    print(f"\nCV seasons: {valid_cv_seasons}", flush=True)

    if not valid_cv_seasons:
        print("ERROR: No valid CV seasons found.", flush=True)
        return

    # All training seasons (with detailed results, minus current year placeholder)
    all_train_seasons = sorted(tourney_seasons_avail - set(valid_cv_seasons))
    # We'll also use non-held-out CV seasons as training data in each fold
    print(f"Base training seasons: {min(all_train_seasons)}-{max(all_train_seasons)} ({len(all_train_seasons)} seasons)", flush=True)

    # Leave-one-season-out CV
    all_cv_preds = []
    all_cv_labels = []
    season_results = {}

    for held_out in valid_cv_seasons:
        print(f"\n{'-' * 50}", flush=True)
        print(f"  Fold: held-out season = {held_out}", flush=True)

        # Train on everything except held-out
        train_seasons = sorted((tourney_seasons_avail - {held_out}))
        val_seasons = [held_out]

        # Fit scaler on training data only
        scaler = fit_scaler(sequences, exclude_seasons=set(val_seasons))

        # Build datasets
        train_dataset = TourneyMatchupDataset(tourney_df, sequences, scaler, train_seasons)
        val_dataset = TourneyMatchupDataset(tourney_df, sequences, scaler, val_seasons)

        print(f"    Train games: {len(train_dataset)}, Val games: {len(val_dataset)}", flush=True)

        if len(val_dataset) == 0:
            print(f"    Skipping {held_out}: no validation games", flush=True)
            continue

        # Create model
        model = TemporalModel(
            input_dim=N_GAME_FEATURES,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            embed_dim=32,
            hidden_dim=64,
            dropout=0.15,
        )

        # Train
        preds, labels = train_model(model, train_dataset, val_dataset, epochs=100, lr=1e-3, batch_size=64)

        # Clip predictions for numerical stability
        preds = np.clip(preds, 0.01, 0.99)

        # Brier score
        brier = np.mean((preds - labels) ** 2)
        log_loss_val = -np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))
        accuracy = np.mean((preds > 0.5) == labels)

        print(f"    {held_out} => Brier: {brier:.4f}  LogLoss: {log_loss_val:.4f}  "
              f"Acc: {accuracy:.3f}  ({len(labels)} games)", flush=True)

        season_results[held_out] = {
            "brier": brier, "logloss": log_loss_val,
            "accuracy": accuracy, "n_games": len(labels),
        }
        all_cv_preds.extend(preds.tolist())
        all_cv_labels.extend(labels.tolist())

    # -- Overall results --
    if all_cv_preds:
        all_cv_preds = np.array(all_cv_preds)
        all_cv_labels = np.array(all_cv_labels)

        overall_brier = np.mean((all_cv_preds - all_cv_labels) ** 2)
        overall_logloss = -np.mean(
            all_cv_labels * np.log(all_cv_preds) + (1 - all_cv_labels) * np.log(1 - all_cv_preds)
        )
        overall_acc = np.mean((all_cv_preds > 0.5) == all_cv_labels)

        print(f"\n{'=' * 70}", flush=True)
        print(f"OVERALL CV RESULTS (2021-2025)", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"  Brier Score:  {overall_brier:.4f}", flush=True)
        print(f"  Log Loss:     {overall_logloss:.4f}", flush=True)
        print(f"  Accuracy:     {overall_acc:.3f}", flush=True)
        print(f"  Total games:  {len(all_cv_labels)}", flush=True)

        print(f"\nPer-season breakdown:", flush=True)
        for s in sorted(season_results.keys()):
            r = season_results[s]
            print(f"  {s}: Brier={r['brier']:.4f}  LogLoss={r['logloss']:.4f}  "
                  f"Acc={r['accuracy']:.3f}  N={r['n_games']}", flush=True)

        print(f"\nBaseline comparison:", flush=True)
        print(f"  Current best CV Brier: 0.0234", flush=True)
        print(f"  This model CV Brier:   {overall_brier:.4f}", flush=True)
        diff = overall_brier - 0.0234
        direction = "worse" if diff > 0 else "better"
        print(f"  Difference:            {diff:+.4f} ({direction})", flush=True)

        # Prediction distribution
        print(f"\nPrediction distribution:", flush=True)
        print(f"  Mean: {all_cv_preds.mean():.3f}  Std: {all_cv_preds.std():.3f}", flush=True)
        print(f"  Min:  {all_cv_preds.min():.3f}  Max: {all_cv_preds.max():.3f}", flush=True)
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pct = (all_cv_preds < thresh).mean() * 100
            print(f"    < {thresh}: {pct:5.1f}%", flush=True)


if __name__ == "__main__":
    run_cv()
