"""JEPA world model for ARC-AGI-3 — Day-1 skeleton.

Spec: jepa_wm/DESIGN.md
Pattern: V-JEPA-2-AC (arXiv:2506.09985) action-conditioned latent predictor.

Components:
  ARCTokenizer        per-cell 16-way embed + 4x4 patch -> 256 tokens
  ActionEncoder       7 action types + 64x64 ACTION6 coords (factored)
  ViT                 12-layer encoder (E_theta) / target-EMA (E_bar)
  Predictor           8-layer narrower transformer (P_phi)
  RewardHead          3-way classifier {0, +step, +level-complete}
  DoneHead            binary
  ValueHead           scalar return-to-win (MCTS bootstrap)
  JEPAWorldModel      glues everything; supports .step(z, a) for MCTS rollouts

Anti-collapse safeguards (per Risk #1):
  - target encoder is EMA copy (no grad)
  - predictor much smaller than encoder
  - VICReg-style variance regularizer hook (compute_variance_reg)
  - per-dim variance monitor (latent_variance method)

NOT YET WIRED:
  - Training loop (next file)
  - 2D RoPE (placeholder uses sinusoidal; swap before training)
  - Color/D4 augmentations (data pipeline file)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Config
# =============================================================================
@dataclass
class JEPAConfig:
    # Grid
    grid_h: int = 64
    grid_w: int = 64
    n_colors: int = 16
    patch: int = 4

    # Model
    d_model: int = 256
    enc_layers: int = 12
    enc_heads: int = 8
    pred_layers: int = 8
    pred_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Actions
    n_simple_actions: int = 7  # ACTION1-5 + ACTION6 + ACTION7
    n_coord_bins: int = 64  # 64-way categorical x, y

    # Heads
    n_reward_classes: int = 3  # {0, +step, +level-complete}

    # Training
    ema_m_start: float = 0.996
    ema_m_end: float = 1.000


# =============================================================================
# Tokenizer: 64x64 categorical grid -> 256 tokens
# =============================================================================
class ARCTokenizer(nn.Module):
    """Per-cell color embedding + 4x4 patch concat -> 256 tokens of d_model."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        # Per-cell color embedding
        self.color_emb = nn.Embedding(cfg.n_colors, cfg.d_model // (cfg.patch * cfg.patch))
        # After 4x4 patch flatten we get d_model dims per token
        self.n_tokens = (cfg.grid_h // cfg.patch) * (cfg.grid_w // cfg.patch)
        # Learned 2D positional embedding (TODO: replace with 2D RoPE before training)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_tokens, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: (B, H, W) long in [0, n_colors)
        B, H, W = grid.shape
        p = self.cfg.patch
        # Per-cell embed
        x = self.color_emb(grid)  # (B, H, W, d_cell)
        # Patchify: (B, H/p, p, W/p, p, d_cell) -> (B, H/p, W/p, p*p*d_cell)
        x = x.reshape(B, H // p, p, W // p, p, -1).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, (H // p) * (W // p), -1)  # (B, n_tokens, d_model)
        x = x + self.pos_emb
        return x


# =============================================================================
# Action encoder: action_id (0-6) + optional (x, y) coord bins
# =============================================================================
class ActionEncoder(nn.Module):
    """Encodes an action (action_id, x, y) into a sequence of action tokens.

    For ACTION6 (click): emits [ACT6_emb, x_emb, y_emb] = 3 tokens.
    For simple actions: emits [act_emb, 0, 0] (pad with zero-tokens for fixed length).
    """

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.act_emb = nn.Embedding(cfg.n_simple_actions, cfg.d_model)
        self.x_emb = nn.Embedding(cfg.n_coord_bins, cfg.d_model)
        self.y_emb = nn.Embedding(cfg.n_coord_bins, cfg.d_model)
        # Action segment positional bias
        self.pos_emb = nn.Parameter(torch.zeros(1, 3, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, action_id: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # action_id: (B,) long ; x, y: (B,) long in [0, 64) (0 if not ACTION6)
        a = self.act_emb(action_id)  # (B, d)
        xe = self.x_emb(x)
        ye = self.y_emb(y)
        tok = torch.stack([a, xe, ye], dim=1)  # (B, 3, d)
        tok = tok + self.pos_emb
        return tok


# =============================================================================
# Transformer building block
# =============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


# =============================================================================
# Encoder (E_theta) and target encoder (E_bar)
# =============================================================================
class ViTEncoder(nn.Module):
    def __init__(self, cfg: JEPAConfig, n_layers: int, n_heads: int):
        super().__init__()
        self.tokenizer = ARCTokenizer(cfg)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: (B, H, W) -> z: (B, n_tokens, d)
        z = self.tokenizer(grid)
        for blk in self.blocks:
            z = blk(z)
        z = self.ln(z)
        return z


# =============================================================================
# Predictor: state_tokens + action_tokens -> next_state_token_predictions
# =============================================================================
class Predictor(nn.Module):
    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.action_encoder = ActionEncoder(cfg)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.pred_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.pred_layers)
        ])
        self.ln = nn.LayerNorm(cfg.d_model)
        # Segment-type embeddings: 0=state, 1=action, 2=next-state-mask
        self.seg_emb = nn.Embedding(3, cfg.d_model)

    def forward(
        self,
        z_state: torch.Tensor,
        action_id: torch.Tensor,
        action_x: torch.Tensor,
        action_y: torch.Tensor,
    ) -> torch.Tensor:
        # z_state: (B, T_state, d) ; returns (B, T_state, d) predicted z_{t+1}
        B, T, d = z_state.shape
        a_tokens = self.action_encoder(action_id, action_x, action_y)  # (B, 3, d)
        mask_tokens = self.mask_token.expand(B, T, d)
        # Concat: [state ; action ; mask] and add segment embeddings
        seg = torch.cat([
            torch.zeros(B, T, dtype=torch.long, device=z_state.device),
            torch.full((B, 3), 1, dtype=torch.long, device=z_state.device),
            torch.full((B, T), 2, dtype=torch.long, device=z_state.device),
        ], dim=1)
        seq = torch.cat([z_state, a_tokens, mask_tokens], dim=1) + self.seg_emb(seg)
        # Causal mask between segments, bidirectional within
        L = T + 3 + T
        attn_mask = torch.zeros(L, L, device=z_state.device, dtype=torch.bool)
        # mask out: state tokens can attend to state only; action can attend to state+action;
        # next-state-mask tokens can attend to state+action+next-mask.
        # nn.MultiheadAttention uses True = "do NOT attend". Build accordingly.
        # Index ranges:
        s_lo, s_hi = 0, T
        a_lo, a_hi = T, T + 3
        n_lo, n_hi = T + 3, L
        # state q -> can attend state only
        attn_mask[s_lo:s_hi, a_lo:n_hi] = True
        # action q -> state + action
        attn_mask[a_lo:a_hi, n_lo:n_hi] = True
        # next-mask q -> all (no mask needed) — leave False
        for blk in self.blocks:
            seq = blk(seq, attn_mask=attn_mask)
        seq = self.ln(seq)
        # Return only the next-state predictions
        return seq[:, n_lo:n_hi, :]


# =============================================================================
# Reward / done / value heads
# =============================================================================
class Heads(nn.Module):
    """Pool predicted-next-state embedding and emit reward/done/value."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.pool = nn.Sequential(nn.LayerNorm(cfg.d_model))
        self.r_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.n_reward_classes),
        )
        self.done_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )
        self.v_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, z: torch.Tensor):
        # z: (B, T, d) predicted next state
        pooled = self.pool(z.mean(dim=1))  # mean-pool over tokens
        r_logits = self.r_head(pooled)
        done_logit = self.done_head(pooled).squeeze(-1)
        v = self.v_head(pooled).squeeze(-1)
        return r_logits, done_logit, v


# =============================================================================
# Full JEPA World Model
# =============================================================================
class JEPAWorldModel(nn.Module):
    """Pulls everything together.

    forward(grid_t, action) returns (z_pred, r_logits, done_logit, v).
    target_encode(grid_t1) returns z_target (no grad).
    """

    def __init__(self, cfg: JEPAConfig | None = None):
        super().__init__()
        self.cfg = cfg or JEPAConfig()
        self.encoder = ViTEncoder(self.cfg, self.cfg.enc_layers, self.cfg.enc_heads)
        self.target_encoder = ViTEncoder(self.cfg, self.cfg.enc_layers, self.cfg.enc_heads)
        # Initialize target = context, freeze gradients
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(self.cfg)
        self.heads = Heads(self.cfg)

    def step(self, grid_t: torch.Tensor, action_id: torch.Tensor,
             action_x: torch.Tensor, action_y: torch.Tensor):
        """One forward step. For training AND MCTS rollouts."""
        z_t = self.encoder(grid_t)
        z_pred = self.predictor(z_t, action_id, action_x, action_y)
        r_logits, done_logit, v = self.heads(z_pred)
        return z_t, z_pred, r_logits, done_logit, v

    @torch.no_grad()
    def target_encode(self, grid_t1: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(grid_t1)

    @torch.no_grad()
    def update_target(self, m: float):
        """EMA update of target encoder."""
        for p_ctx, p_tgt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.mul_(m).add_(p_ctx.detach(), alpha=1 - m)

    @torch.no_grad()
    def latent_variance(self, grid_batch: torch.Tensor) -> torch.Tensor:
        """Per-dim variance of encoder output — collapse monitor."""
        z = self.encoder(grid_batch)  # (B, T, d)
        z_flat = z.reshape(-1, z.shape[-1])
        return z_flat.var(dim=0)  # (d,)

    def compute_variance_reg(self, z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """VICReg-style variance regularizer (hinge: enforce std > 1)."""
        z_flat = z.reshape(-1, z.shape[-1])
        std = torch.sqrt(z_flat.var(dim=0) + eps)
        return F.relu(1.0 - std).mean()

    def n_params(self) -> dict:
        def cnt(m):
            return sum(p.numel() for p in m.parameters())
        return dict(
            encoder=cnt(self.encoder),
            target_encoder=cnt(self.target_encoder),
            predictor=cnt(self.predictor),
            heads=cnt(self.heads),
            total=cnt(self),
        )


# =============================================================================
# Smoke test (run as: uv run python -m jepa_wm.models.jepa)
# =============================================================================
if __name__ == "__main__":
    cfg = JEPAConfig()
    wm = JEPAWorldModel(cfg)
    params = wm.n_params()
    print("Param count:", {k: f"{v/1e6:.2f}M" for k, v in params.items()})

    B = 4
    grid_t = torch.randint(0, cfg.n_colors, (B, cfg.grid_h, cfg.grid_w))
    grid_t1 = torch.randint(0, cfg.n_colors, (B, cfg.grid_h, cfg.grid_w))
    action_id = torch.randint(0, cfg.n_simple_actions, (B,))
    action_x = torch.randint(0, cfg.n_coord_bins, (B,))
    action_y = torch.randint(0, cfg.n_coord_bins, (B,))

    z_t, z_pred, r_logits, done_logit, v = wm.step(grid_t, action_id, action_x, action_y)
    z_target = wm.target_encode(grid_t1)
    print("Shapes:", dict(
        z_t=tuple(z_t.shape), z_pred=tuple(z_pred.shape),
        z_target=tuple(z_target.shape),
        r_logits=tuple(r_logits.shape), done=tuple(done_logit.shape), v=tuple(v.shape),
    ))

    # Collapse check
    var = wm.latent_variance(grid_t)
    print(f"Latent per-dim variance: min={var.min().item():.4f} "
          f"mean={var.mean().item():.4f} max={var.max().item():.4f}")

    # JEPA loss + variance reg
    loss_jepa = F.l1_loss(z_pred, z_target)
    var_reg = wm.compute_variance_reg(z_t)
    print(f"L_jepa (random init): {loss_jepa.item():.4f}  var_reg: {var_reg.item():.4f}")

    # EMA update sanity
    wm.update_target(0.996)
    print("EMA update OK")
