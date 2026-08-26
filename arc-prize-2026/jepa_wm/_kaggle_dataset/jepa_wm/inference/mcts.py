"""MCTS-in-latent for JEPA world model.

Pattern: MuZero/UniZero closed-loop MPC.
- Encode current REAL frame to z_0 (once per real step).
- Run N simulations from z_0; each sim expands a path via P_phi (no env interaction).
- Score with PUCT. At leaves, use v_head as bootstrap.
- After N sims, pick the root action by visit count.
- Execute on the REAL env, re-encode the resulting frame, repeat.

Latent dynamics are queried in BATCHES at each depth for speed.

NOT YET WIRED:
  - The factored x/y action head for ACTION6 — current MCTS branches over
    a flat enumerated action set (simple actions + a top-K click candidates
    set you pass in). The K-prior trick (per design risk #3) lives in the agent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class MCTSConfig:
    n_simulations: int = 64
    max_depth: int = 12
    c_puct: float = 1.25
    discount: float = 0.997
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    value_temp: float = 1.0


@dataclass
class ActionSpec:
    """An action as a (action_id, x, y) triple. x,y are 0 for non-click."""
    action_id: int
    x: int = 0
    y: int = 0

    def as_tensors(self, device):
        return (
            torch.tensor([self.action_id], device=device, dtype=torch.long),
            torch.tensor([self.x], device=device, dtype=torch.long),
            torch.tensor([self.y], device=device, dtype=torch.long),
        )


@dataclass
class Node:
    z: torch.Tensor  # (1, T, d) latent state
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    children: dict = field(default_factory=dict)  # action_index -> Node

    @property
    def value(self):
        return self.value_sum / max(1, self.visit_count)

    def expanded(self):
        return len(self.children) > 0


class LatentMCTS:
    """MCTS over latent rollouts of a JEPAWorldModel."""

    def __init__(self, world_model, cfg: MCTSConfig | None = None, device=None):
        self.wm = world_model
        self.wm.eval()
        self.cfg = cfg or MCTSConfig()
        self.device = device or next(world_model.parameters()).device

    @torch.no_grad()
    def _expand(self, node: Node, actions: list[ActionSpec], priors: torch.Tensor | None = None):
        """Expand `node` with one child per action. priors: (n_actions,) or None."""
        n = len(actions)
        if priors is None:
            priors = torch.ones(n, device=self.device) / n
        for i, a in enumerate(actions):
            node.children[i] = Node(z=node.z, prior=float(priors[i]))  # z filled at first visit
        # Mark expanded by leaving children dict non-empty

    @torch.no_grad()
    def _batched_dynamics(self, z_batch: torch.Tensor, actions: list[ActionSpec]):
        """Given a batch of latents and the SAME action template per element,
        return (z_pred, r_pred_class, done_prob, v_pred). Used during simulation."""
        # Build action tensors from the per-batch action selection upstream.
        # Caller passes parallel lists — we just stack here.
        B = z_batch.shape[0]
        a_id = torch.tensor([a.action_id for a in actions], device=self.device, dtype=torch.long)
        a_x = torch.tensor([a.x for a in actions], device=self.device, dtype=torch.long)
        a_y = torch.tensor([a.y for a in actions], device=self.device, dtype=torch.long)
        # The predictor expects (B, T, d) for state — we already have z_batch
        z_pred = self.wm.predictor(z_batch, a_id, a_x, a_y)
        r_logits, done_logit, v = self.wm.heads(z_pred)
        r_probs = torch.softmax(r_logits, dim=-1)
        # Expected reward over 3-way class: weights {0, +1, +5} (level-up gets 5)
        weights = torch.tensor([0.0, 1.0, 5.0], device=self.device)
        r_pred = (r_probs * weights).sum(dim=-1)
        done = torch.sigmoid(done_logit)
        return z_pred, r_pred, done, v

    def _ucb_select(self, node: Node) -> int:
        total = sum(c.visit_count for c in node.children.values())
        best_idx, best_score = -1, -float("inf")
        sqrt_total = math.sqrt(max(1, total))
        for i, c in node.children.items():
            u = self.cfg.c_puct * c.prior * sqrt_total / (1 + c.visit_count)
            q = c.value
            s = q + u
            if s > best_score:
                best_score = s
                best_idx = i
        return best_idx

    @torch.no_grad()
    def search(
        self,
        z_root: torch.Tensor,
        actions: list[ActionSpec],
        priors: Optional[torch.Tensor] = None,
        add_root_dirichlet: bool = True,
    ) -> tuple[int, dict]:
        """Run N simulations from z_root over the given action set; return
        the most-visited root action index + a stats dict."""
        root = Node(z=z_root, prior=1.0)
        self._expand(root, actions, priors)

        # Root-Dirichlet for exploration
        if add_root_dirichlet:
            alpha = torch.tensor([self.cfg.dirichlet_alpha] * len(actions))
            noise = torch.distributions.Dirichlet(alpha).sample().to(self.device)
            for i, c in root.children.items():
                c.prior = (1 - self.cfg.dirichlet_eps) * c.prior + self.cfg.dirichlet_eps * float(noise[i])

        for _ in range(self.cfg.n_simulations):
            node = root
            path = [node]
            depth = 0
            # Selection
            while node.expanded() and depth < self.cfg.max_depth:
                idx = self._ucb_select(node)
                node = node.children[idx]
                path.append((idx, node))
                depth += 1
                if node.visit_count == 0:
                    break  # Reached an unevaluated leaf
            # Expansion / evaluation
            if depth == 0:
                # Root only; skip
                value = 0.0
            else:
                # Use the parent's z to predict this leaf
                parent_node = path[-2] if len(path) >= 2 else root
                if isinstance(parent_node, tuple):
                    parent_node = parent_node[1]
                action_idx_from_parent = path[-1][0]
                action = actions[action_idx_from_parent]
                z_pred, r_pred, done, v = self._batched_dynamics(parent_node.z, [action])
                leaf = path[-1][1]
                leaf.z = z_pred  # set latent for future expansions from here
                leaf.reward = float(r_pred.item())
                value = 0.0 if float(done.item()) > 0.5 else float(v.item())
                # Pre-expand its children with uniform prior so the next sim can descend
                self._expand(leaf, actions, priors=None)
            # Backup
            for entry in reversed(path):
                if isinstance(entry, tuple):
                    _, n = entry
                else:
                    n = entry
                n.value_sum += value
                n.visit_count += 1
                value = n.reward + self.cfg.discount * value

        # Pick action with highest visit count
        best_idx, best_visits = -1, -1
        visits = {}
        for i, c in root.children.items():
            visits[i] = c.visit_count
            if c.visit_count > best_visits:
                best_visits = c.visit_count
                best_idx = i
        stats = dict(visits=visits, root_value=root.value)
        return best_idx, stats


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    from jepa_wm.models.jepa import JEPAConfig, JEPAWorldModel
    cfg = JEPAConfig()
    wm = JEPAWorldModel(cfg).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = wm.to(device)

    # Encode a fake current frame
    grid = torch.randint(0, cfg.n_colors, (1, cfg.grid_h, cfg.grid_w), device=device)
    with torch.no_grad():
        z_root = wm.encoder(grid)

    # Action set: simple ACTION1-5 + 3 candidate clicks
    actions = [
        ActionSpec(1), ActionSpec(2), ActionSpec(3), ActionSpec(4), ActionSpec(5),
        ActionSpec(6, x=32, y=32), ActionSpec(6, x=16, y=48), ActionSpec(6, x=48, y=16),
    ]

    mcts = LatentMCTS(wm, MCTSConfig(n_simulations=16, max_depth=4), device=device)
    best, stats = mcts.search(z_root, actions, add_root_dirichlet=True)
    print("Best action idx:", best)
    print("Root value:", stats["root_value"])
    print("Visits:", stats["visits"])
    print("MCTS-in-latent smoke OK")
