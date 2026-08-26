# =====================================================================
# v31 — FULL AXIOM agent (object-centric Bayesian world model)
#
# Clean re-implementation from the paper:
#   AXIOM: Learning to Play Games in Minutes with Expanding
#   Object-Centric Models  (arXiv:2505.24784, Verses AI)
#
# NOT a port of github.com/VersesTech/axiom (license-clean re-impl
# from the published equations only).
#
# Modules implemented (all numpy, CPU, no gradient descent):
#   sMM  : slot model -> replaced by connected-component segmentation
#          on the discrete 64x64 colour grid (FrameProcessor, copied
#          verbatim from v24_agent.py — self-contained, numpy only).
#   iMM  : identity mixture. Match objects across frames on a
#          [colour, w, h] signature using a Dirichlet/NIW-style
#          responsibility; spawn a new identity type when the best
#          responsibility falls below tau.
#   tMM  : transition mixture. Switching linear dynamical system,
#          L modes SHARED across all slots. Each mode predicts a
#          delta-state. New mode grows when no existing mode predicts
#          the observed transition within tolerance.
#   rMM  : relational mixture. Conditions the tMM switch on the
#          nearest-neighbour context (continuous geometric features +
#          discrete identity/action/reward). Mixture of
#          Gaussian x Categorical, grown on demand.
#   BMR  : Bayesian Model Reduction. Every BMR_PERIOD frames, greedily
#          test pairwise rMM-component merges and accept a merge iff it
#          lowers the expected free energy of the multinomials over
#          (reward, next tMM switch).
#   plan : H-step rollout. pi* = argmin sum [ -E[log p(r)]
#          - infogain ]  (utility + Dirichlet information gain,
#          closed-form).
#
# Per-step loop:
#   perceive(sMM) -> identify(iMM) -> tMM infer -> rMM update
#   -> M-step natural-param increments -> growth check
#   -> (every BMR_PERIOD) BMR -> plan H -> act
# =====================================================================
import logging
import math
import time
from collections import deque

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)


# =====================================================================
# FrameProcessor — copied verbatim from v24_agent.py (numpy only,
# self-contained). Only segment_frame is used by v31.
# =====================================================================
class FrameProcessor:
    OFFSETS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    OFFSETS8 = ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self):
        self.connectivity_rank = 4
        self.status_bar_mode = "rule"
        self.status_bar_distance_threshold = 3
        self.status_bar_ratio_threshold = 5
        self.status_bar_twins_threshold = 3
        self.frame_shape = (64, 64)
        self.status_bar_color = 16
        self.minimal_width = 2
        self.maximal_width = 32
        self.non_salient_color = set([0, 1, 2, 3, 4, 5])
        self.salient_color = set([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    def segment_frame(self, frame):
        """Segment frame into 4-connected same-colour components.

        Returns (label_map, components) where each component dict has:
        bounding_box=(x1,y1,x2,y2) inclusive, color, area, is_rectangle,
        number_of_twins, twin_ids.
        """
        h, w = frame.shape
        label_map = np.zeros((h, w), dtype=int) - 1
        components = []
        cid = -1
        offsets = self.OFFSETS4 if self.connectivity_rank == 4 else self.OFFSETS8

        for y in range(h):
            for x in range(w):
                if label_map[y, x] != -1:
                    continue
                cid += 1
                color = int(frame[y, x])
                q = deque([(y, x)])
                label_map[y, x] = cid
                min_x = max_x = x
                min_y = max_y = y
                area = 0
                while q:
                    cy, cx = q.popleft()
                    area += 1
                    min_x, max_x = min(min_x, cx), max(max_x, cx)
                    min_y, max_y = min(min_y, cy), max(max_y, cy)
                    for dy, dx in offsets:
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < h and 0 <= nx < w
                                and label_map[ny, nx] == -1
                                and frame[ny, nx] == color):
                            label_map[ny, nx] = cid
                            q.append((ny, nx))
                rect_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                is_rect = area == rect_area
                components.append(dict(
                    bounding_box=(min_x, min_y, max_x, max_y),
                    color=color, area=area, is_rectangle=is_rect,
                ))

        for i, comp in enumerate(components):
            twins = [j for j, other in enumerate(components)
                     if i != j and other["area"] == comp["area"]
                     and other["is_rectangle"] == comp["is_rectangle"]
                     and other["color"] == comp["color"]]
            comp["number_of_twins"] = len(twins)
            comp["twin_ids"] = twins
        return label_map, components


# =====================================================================
# AXIOM Bayesian world model
# =====================================================================
EPS = 1e-9


def _slot_state(comp):
    """Object state x = [cx, cy, color, area, w, h] from a component."""
    x1, y1, x2, y2 = comp["bounding_box"]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = float(x2 - x1 + 1)
    h = float(y2 - y1 + 1)
    return np.array([cx, cy, float(comp["color"]), float(comp["area"]), w, h],
                    dtype=np.float64)


def _signature(comp):
    """iMM identity signature = (color, w, h)."""
    x1, y1, x2, y2 = comp["bounding_box"]
    return (int(comp["color"]), int(x2 - x1 + 1), int(y2 - y1 + 1))


class IdentityMM:
    """iMM: match objects across frames by [colour, shape-extent].

    For ARC the identity signature (colour, w, h) is *discrete*, so the
    NIW Gaussian collapses to an exact-match mixture: each distinct
    signature is one identity type, with a Dirichlet over type usage
    (Dir(1..1, alpha0)). A new type is spawned the first time a
    signature is seen — exactly the "max responsibility < tau" rule
    when the per-type likelihood is a Kronecker delta on the discrete
    feature. This is robust (no parameter collapse) and closed-form.
    """

    def __init__(self, tau=0.30, alpha0=1.0):
        self.tau = tau
        self.alpha0 = alpha0
        self.sig2idx = {}       # signature tuple -> identity index
        self.mu = []            # representative 3-vector per type
        self.counts = []        # Dirichlet usage counts

    def assign(self, feat):
        """feat = (colour, w, h) array. Return identity index, growing
        a new type on first sight of a signature."""
        key = (int(round(feat[0])), int(round(feat[1])), int(round(feat[2])))
        k = self.sig2idx.get(key)
        if k is not None:
            self.counts[k] += 1.0
            return k
        k = len(self.mu)
        self.sig2idx[key] = k
        self.mu.append(np.array(key, dtype=np.float64))
        self.counts.append(1.0)
        return k


class TransitionMM:
    """tMM: switching linear dynamical system over slot delta-state.

    L modes SHARED across all slots. Mode l predicts the next-state
    delta as a constant b_l (a translation in centroid space; the
    full D_l x is reduced to identity for ARC because object identity
    is preserved across frames). Likelihood N(dx; b_l, 2 I). A new
    mode grows when no mode predicts the observed transition within
    tolerance.
    """

    def __init__(self, tol=2.5, obs_var=2.0, max_modes=64):
        self.tol = tol
        self.obs_var = obs_var
        self.max_modes = max_modes
        self.b = []          # list of delta-vectors (6-D)
        self.n = []          # pseudo-count per mode
        self.counts = []     # Dirichlet usage counts

    def _ll(self, dx):
        if not self.b:
            return np.array([])
        b = np.stack(self.b)                          # (L,6)
        # only centroid + size dims matter for dynamics
        diff = dx[None, :] - b
        return -0.5 * np.sum(diff * diff, axis=1) / self.obs_var

    def infer(self, dx):
        """Return (mode, was_new) for an observed delta-state."""
        if self.b:
            ll = self._ll(dx)
            l = int(np.argmax(ll))
            # tolerance in centroid space (dims 0,1)
            cen_err = math.hypot(dx[0] - self.b[l][0], dx[1] - self.b[l][1])
            if cen_err <= self.tol:
                self.n[l] += 1.0
                lr = 1.0 / self.n[l]
                self.b[l] = (1 - lr) * self.b[l] + lr * dx
                self.counts[l] += 1.0
                return l, False
        if len(self.b) >= self.max_modes:
            # saturated: snap to closest existing mode
            ll = self._ll(dx)
            l = int(np.argmax(ll))
            self.counts[l] += 1.0
            return l, False
        self.b.append(dx.astype(np.float64).copy())
        self.n.append(1.0)
        self.counts.append(1.0)
        return len(self.b) - 1, True

    def predict(self, l):
        return self.b[l] if 0 <= l < len(self.b) else np.zeros(6)


class RelationalMM:
    """rMM: conditions the tMM switch on nearest-neighbour context.

    A component m models p(f, d | s=m) = N(f; mu_m, Sigma_m) *
    prod_i Cat(d_i; alpha_{m,i}).

      continuous f = [C*cx, C*cy, dx_near, dy_near]   (geometric)
      discrete   d = (identity z, action a_{t-1},
                      reward bucket r_t, tMM switch s_tmm)

    Each component predicts a distribution over the tMM switch and over
    the reward bucket -> used by the planner and by BMR. Grown on demand
    (sparsity = nearest-neighbour-only context).
    """

    NA = 7          # actions: 0..6 (RESET..ACTION6)
    NR = 3          # reward buckets: 0 neg, 1 zero/small, 2 positive
    # features in game-cell units (ARC grids are <=16 cells, displayed
    # at 4x -> divide pixel coords by 4). feat_var/grow_thresh tuned so
    # objects >~1.5 cells apart spawn distinct relational components.
    CSCALE = 1.0 / 4.0

    def __init__(self, n_modes_hint=64, feat_var=2.0, grow_thresh=-2.0):
        self.feat_var = feat_var
        self.grow_thresh = grow_thresh
        self.mu = []        # (4,) continuous mean
        self.n = []         # pseudo-count
        # Dirichlet count tables, lazily sized
        self.cnt_s = []     # over tMM switch
        self.cnt_r = []     # over reward bucket  (size NR)
        self.cnt_a = []     # over action         (size NA)
        self.cnt_z = []     # over identity bucket

    @staticmethod
    def _feat(cx, cy, dxn, dyn):
        return np.array([cx * RelationalMM.CSCALE, cy * RelationalMM.CSCALE,
                         dxn * RelationalMM.CSCALE, dyn * RelationalMM.CSCALE],
                        dtype=np.float64)

    def _grow(self, feat):
        self.mu.append(feat.copy())
        self.n.append(1.0)
        self.cnt_s.append({})
        self.cnt_r.append(np.ones(self.NR))
        self.cnt_a.append(np.ones(self.NA))
        self.cnt_z.append({})
        return len(self.mu) - 1

    def _ll(self, feat):
        if not self.mu:
            return np.array([])
        mu = np.stack(self.mu)
        d = feat[None, :] - mu
        return -0.5 * np.sum(d * d, axis=1) / self.feat_var

    MAX_COMP = 512

    def select(self, feat):
        """Pick (or grow) the rMM component matching the context."""
        if self.mu:
            ll = self._ll(feat)
            m = int(np.argmax(ll))
            if ll[m] >= self.grow_thresh or len(self.mu) >= self.MAX_COMP:
                self.n[m] += 1.0
                lr = 1.0 / self.n[m]
                self.mu[m] = (1 - lr) * self.mu[m] + lr * feat
                return m
        return self._grow(feat)

    def update(self, m, s_tmm, r_bucket, action, z):
        self.cnt_s[m][s_tmm] = self.cnt_s[m].get(s_tmm, 0.0) + 1.0
        self.cnt_r[m][min(r_bucket, self.NR - 1)] += 1.0
        if 0 <= action < self.NA:
            self.cnt_a[m][action] += 1.0
        self.cnt_z[m][z] = self.cnt_z[m].get(z, 0.0) + 1.0

    # ---- predictive distributions for the planner --------------------
    def p_reward(self, m):
        c = self.cnt_r[m]
        return c / c.sum()

    def p_switch(self, m):
        d = self.cnt_s[m]
        if not d:
            return {}
        tot = sum(d.values())
        return {k: v / tot for k, v in d.items()}

    def expected_reward(self, m):
        """E[r] with bucket centres {-1, 0, +1}."""
        p = self.p_reward(m)
        return -1.0 * p[0] + 0.0 * p[1] + 1.0 * p[2]

    def dirichlet_infogain(self, m):
        """Closed-form expected information gain of one more observation
        for the reward + switch Dirichlets (entropy of the mean dist,
        higher = more uncertain = more to learn)."""
        pr = self.p_reward(m)
        h_r = -np.sum(pr * np.log(pr + EPS))
        ps = self.p_switch(m)
        if ps:
            v = np.array(list(ps.values()))
            h_s = -np.sum(v * np.log(v + EPS))
        else:
            h_s = math.log(2.0)
        return h_r + h_s


# =====================================================================
# Bayesian Model Reduction over rMM components
# =====================================================================
def _dir_efe(cnt_vec):
    """Expected free energy proxy for a multinomial = neg-entropy of its
    Dirichlet mean (lower entropy => more committed => preferred)."""
    p = cnt_vec / (cnt_vec.sum() + EPS)
    return float(np.sum(p * np.log(p + EPS)))   # = -entropy


def _switch_vec(d):
    if not d:
        return np.ones(1)
    keys = sorted(d.keys())
    return np.array([d[k] for k in keys], dtype=np.float64)


def bayesian_model_reduction(rmm, max_pairs=400):
    """Greedily test pairwise rMM-component merges; accept iff the merge
    lowers the combined expected free energy of the multinomials over
    (reward, next tMM switch)."""
    K = len(rmm.mu)
    if K < 2:
        return 0
    merged = 0
    # candidate pairs: nearest in continuous feature space (sparsity)
    mu = np.stack(rmm.mu)
    order = []
    for i in range(K):
        for j in range(i + 1, K):
            order.append((np.sum((mu[i] - mu[j]) ** 2), i, j))
    order.sort()
    alive = [True] * K
    for _, i, j in order[:max_pairs]:
        if not (alive[i] and alive[j]):
            continue
        # current EFE
        efe_before = (_dir_efe(rmm.cnt_r[i]) + _dir_efe(rmm.cnt_r[j])
                      + _dir_efe(_switch_vec(rmm.cnt_s[i]))
                      + _dir_efe(_switch_vec(rmm.cnt_s[j])))
        # merged statistics
        mr = rmm.cnt_r[i] + rmm.cnt_r[j] - 1.0
        ms = dict(rmm.cnt_s[i])
        for k, v in rmm.cnt_s[j].items():
            ms[k] = ms.get(k, 0.0) + v
        efe_after = 2.0 * (_dir_efe(mr) + _dir_efe(_switch_vec(ms)))
        if efe_after < efe_before:           # lower EFE => accept merge
            ni, nj = rmm.n[i], rmm.n[j]
            rmm.mu[i] = (ni * rmm.mu[i] + nj * rmm.mu[j]) / (ni + nj + EPS)
            rmm.n[i] = ni + nj
            rmm.cnt_r[i] = mr
            rmm.cnt_s[i] = ms
            rmm.cnt_a[i] = rmm.cnt_a[i] + rmm.cnt_a[j] - 1.0
            for k, v in rmm.cnt_z[j].items():
                rmm.cnt_z[i][k] = rmm.cnt_z[i].get(k, 0.0) + v
            alive[j] = False
            merged += 1
    if merged:
        keep = [k for k in range(K) if alive[k]]
        rmm.mu = [rmm.mu[k] for k in keep]
        rmm.n = [rmm.n[k] for k in keep]
        rmm.cnt_s = [rmm.cnt_s[k] for k in keep]
        rmm.cnt_r = [rmm.cnt_r[k] for k in keep]
        rmm.cnt_a = [rmm.cnt_a[k] for k in keep]
        rmm.cnt_z = [rmm.cnt_z[k] for k in keep]
    return merged


# =====================================================================
# The agent
# =====================================================================
class MyAgent(Agent):
    MAX_ACTIONS = 100000

    SIMPLE = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
              3: GameAction.ACTION3, 4: GameAction.ACTION4,
              5: GameAction.ACTION5}

    BMR_PERIOD = 500
    PLAN_H = 3                 # rollout horizon
    INFO_W = 0.5               # info-gain weight in the planner objective

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1e6) + abs(hash(self.game_id)) % 1000000
        self.rng = np.random.default_rng(seed % (2 ** 32))
        self.fp = FrameProcessor()

        # world model
        self.imm = IdentityMM()
        self.tmm = TransitionMM()
        self.rmm = RelationalMM()

        # bookkeeping
        self.frame_count = 0
        self.last_levels = 0
        self.prev_slots = None          # list[(state, identity)]
        self.prev_action_id = 0         # 0=RESET
        self.prev_segments = None
        self.prev_label_map = None
        self.prev_frame_hash = None
        self.error_count = 0
        # per-(context-component, action) value table fed by the rMM
        self.q_recent = {}              # (m, a) -> ema reward

        # active-information click frontier: keyed by (frame_hash,
        # segment-signature) so the agent systematically probes every
        # distinct clickable object once before repeating — this is the
        # grounded realisation of the planner's info-gain term for the
        # sparse-reward click games (ft09 etc).
        self.click_tried = {}           # (fh, sig) -> n times tried
        self.click_effect = {}          # (fh, sig) -> True if ever changed
        self.last_click_key = None      # (fh, sig) of the click just made

    # ----------------------------------------------------------------
    def is_done(self, frames, latest_frame):
        try:
            return latest_frame.state is GameState.WIN
        except Exception:
            return True

    # ----------------------------------------------------------------
    def choose_action(self, frames, latest_frame):
        try:
            return self._axiom_step(frames, latest_frame)
        except Exception as e:
            self.error_count += 1
            logger.warning(f"v31 choose_action error: {e}; safe RESET")
            self.prev_slots = None
            self.prev_action_id = 0
            return GameAction.RESET

    # ----------------------------------------------------------------
    def _axiom_step(self, frames, latest_frame):
        st = latest_frame.state
        if st in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_slots = None
            self.prev_action_id = 0
            self.prev_frame_hash = None
            return GameAction.RESET

        arr = np.array(latest_frame.frame, dtype=np.uint8)
        if arr.size == 0:
            return GameAction.RESET
        grid = arr[-1].copy()                    # 64x64 discrete colours
        n_subframes = arr.shape[0]

        avail = list(getattr(latest_frame, "available_actions", []) or [])
        cur_levels = getattr(latest_frame, "levels_completed", 0) or 0

        # ---- reward signal --------------------------------------------
        fh = hash(grid.tobytes())
        if cur_levels > self.last_levels:
            r_val, r_bucket = 1.0, 2
        elif self.prev_frame_hash is not None and fh != self.prev_frame_hash:
            r_val, r_bucket = 0.1, 1
        else:
            r_val, r_bucket = -0.1, 0
        self.last_levels = cur_levels
        frame_changed = (self.prev_frame_hash is not None
                         and fh != self.prev_frame_hash)

        # credit the previous click target with its observed effect
        if self.last_click_key is not None:
            if frame_changed or r_bucket == 2:
                self.click_effect[self.last_click_key] = True
            self.last_click_key = None

        # lightweight model-growth telemetry (every 300 steps, INFO)
        if not hasattr(self, "_dbg"):
            self._dbg = dict(steps=0, changed=0, rew_pos=0)
        self._dbg["steps"] += 1
        if r_bucket == 1:
            self._dbg["changed"] += 1
        if r_bucket == 2:
            self._dbg["rew_pos"] += 1
        if self._dbg["steps"] % 300 == 0:
            logger.info(
                f"v31 step={self._dbg['steps']} changed={self._dbg['changed']} "
                f"pos={self._dbg['rew_pos']} imm={len(self.imm.mu)} "
                f"tmm={len(self.tmm.b)} rmm={len(self.rmm.mu)} "
                f"err={self.error_count}")

        # ---- perceive (sMM via connected components) ------------------
        label_map, comps = self.fp.segment_frame(grid)
        # keep the most informative slots (drop the huge background blob)
        idx_comps = list(enumerate(comps))
        idx_comps.sort(key=lambda ic: ic[1]["area"])
        # background = largest area component; keep the rest (cap for speed)
        kept = idx_comps[:-1] if len(idx_comps) > 1 else idx_comps
        kept = kept[:60]

        # ---- identify (iMM) -------------------------------------------
        slots = []          # list of dict: state, ident, comp_idx, comp
        for ci, comp in kept:
            x = _slot_state(comp)
            sig = np.array(_signature(comp), dtype=np.float64)
            z = self.imm.assign(sig)
            slots.append(dict(state=x, ident=z, ci=ci, comp=comp))

        # ---- learn from the previous transition -----------------------
        if self.prev_slots is not None and self.prev_action_id is not None:
            self._learn_transition(slots, r_bucket, r_val)

        # ---- periodic Bayesian Model Reduction ------------------------
        self.frame_count += 1
        if self.frame_count % self.BMR_PERIOD == 0:
            try:
                bayesian_model_reduction(self.rmm)
            except Exception as e:
                logger.warning(f"v31 BMR skipped: {e}")

        # ---- build action set -----------------------------------------
        simple_actions = [a for a in avail if a in self.SIMPLE]
        has_click = 6 in avail

        if not simple_actions and not has_click:
            self.prev_slots = slots
            self.prev_action_id = 0
            self.prev_frame_hash = fh
            return GameAction.RESET

        # ---- plan H steps & act ---------------------------------------
        action_id, click_xy = self._plan(slots, label_map, simple_actions,
                                         has_click, n_subframes, fh)

        if action_id == 6 and click_xy is not None:
            act = GameAction.ACTION6
            act.set_data({"x": int(click_xy[0]), "y": int(click_xy[1])})
        elif action_id in self.SIMPLE:
            act = self.SIMPLE[action_id]
        else:
            act = GameAction.RESET
            action_id = 0

        self.prev_slots = slots
        self.prev_action_id = action_id
        self.prev_segments = comps
        self.prev_label_map = label_map
        self.prev_frame_hash = fh
        return act

    # ----------------------------------------------------------------
    def _nearest(self, x, slots, exclude_ci):
        """Nearest other slot in centroid space -> (dx, dy, ident)."""
        best = None
        bd = 1e18
        for s in slots:
            if s["ci"] == exclude_ci:
                continue
            o = s["state"]
            d = (o[0] - x[0]) ** 2 + (o[1] - x[1]) ** 2
            if d < bd:
                bd = d
                best = s
        if best is None:
            return 0.0, 0.0, -1
        return (best["state"][0] - x[0], best["state"][1] - x[1],
                best["ident"])

    def _learn_transition(self, slots, r_bucket, r_val):
        """Match prev slots to current by identity, feed tMM + rMM,
        do the conjugate M-step (natural-param increments are the
        running pseudo-count updates inside the modules)."""
        # index current slots by identity
        cur_by_id = {}
        for s in slots:
            cur_by_id.setdefault(s["ident"], []).append(s)

        a = self.prev_action_id
        for ps in self.prev_slots:
            z = ps["ident"]
            matches = cur_by_id.get(z)
            if not matches:
                continue
            # nearest current slot of same identity = the moved object
            px = ps["state"]
            cs = min(matches,
                     key=lambda s: (s["state"][0] - px[0]) ** 2
                     + (s["state"][1] - px[1]) ** 2)
            dx = cs["state"] - px

            # tMM: infer / grow the shared transition mode
            s_tmm, _new = self.tmm.infer(dx)

            # rMM: nearest-neighbour relational context
            dxn, dyn, _zn = self._nearest(px, self.prev_slots, ps["ci"])
            feat = RelationalMM._feat(px[0], px[1], dxn, dyn)
            m = self.rmm.select(feat)
            self.rmm.update(m, s_tmm, r_bucket, a, z)

            # per-(context, action) reward EMA (planner value head)
            key = (m, a)
            old = self.q_recent.get(key, 0.0)
            self.q_recent[key] = 0.7 * old + 0.3 * r_val

    # ----------------------------------------------------------------
    def _score_action(self, slots, a):
        """Planner per-action score:  E[log p(r)] + INFO_W * infogain.
        Uses the rMM component each slot would route to (current
        context) and the action-conditioned reward statistics."""
        if not slots:
            return 0.0
        util = 0.0
        info = 0.0
        seen = 0
        for ps in slots:
            px = ps["state"]
            dxn, dyn, _ = self._nearest(px, slots, ps["ci"])
            feat = RelationalMM._feat(px[0], px[1], dxn, dyn)
            if not self.rmm.mu:
                continue
            ll = self.rmm._ll(feat)
            m = int(np.argmax(ll))
            util += self.rmm.expected_reward(m)
            info += self.rmm.dirichlet_infogain(m)
            # action-conditioned recent reward (exploitation signal)
            util += 2.0 * self.q_recent.get((m, a), 0.0)
            # prefer actions this context has tried less (exploration)
            atried = self.rmm.cnt_a[m][a] if a < self.rmm.NA else 1.0
            info += 1.0 / atried
            seen += 1
        if seen == 0:
            return 0.0
        return (util + self.INFO_W * info) / seen

    def _plan(self, slots, label_map, simple_actions, has_click,
              n_subframes, fh):
        """H-step rollout over the available actions. The paper's
        argmin over policies = utility + info-gain. We realise it as a
        closed-form surrogate: each candidate action is scored by its
        rMM expected reward + Dirichlet info-gain (geometrically
        discounted over the H-step horizon). For click games the
        info-gain term is *grounded* in a systematic per-object click
        frontier so the rare frame-changing clicks are actually
        discovered (otherwise the model never receives a signal)."""
        cand = list(simple_actions)
        if has_click:
            cand.append(6)

        # Score simple actions with the learned model (utility +
        # info-gain, horizon-discounted).
        gamma = 0.9
        horizon = (1.0 - gamma ** self.PLAN_H) / (1.0 - gamma)
        best_simple, best_score = None, -1e18
        for a in simple_actions:
            s = self._score_action(slots, a) * horizon
            if s > best_score:
                best_score = s
                best_simple = a

        click_xy = None
        click_score = -1e18
        if has_click:
            click_xy, click_score = self._pick_click(slots, label_map, fh)
            click_score *= horizon

        # If we have both modalities, pick the higher-scoring one; under
        # cold start (no model yet) prefer systematic clicking which is
        # the only thing that yields information in sparse click games.
        eps = self._epsilon()
        explore = (len(self.rmm.mu) < 3) or (self.rng.random() < eps)

        if has_click and not simple_actions:
            return 6, click_xy
        if not has_click:
            if best_simple is None:
                return 0, None
            if explore:
                return int(self.rng.choice(simple_actions)), None
            return best_simple, None
        # both available
        if explore:
            # bias exploration toward clicks (richer action space)
            if self.rng.random() < 0.7:
                return 6, click_xy
            return int(self.rng.choice(simple_actions)), None
        if click_score >= best_score:
            return 6, click_xy
        return best_simple, None

    def _epsilon(self):
        """Decaying exploration rate (info-gain dominates early)."""
        return max(0.08, 0.7 * math.exp(-self.frame_count / 600.0))

    def _pick_click(self, slots, label_map, fh):
        """Choose an ACTION6 target.

        Priority (the planner's utility + info-gain made concrete):
          1. objects whose click is *known* to change the frame and
             whose rMM context has positive expected reward (exploit);
          2. objects never clicked from this frame state (pure
             info-gain — systematic coverage of the frontier);
          3. least-clicked object, model-scored (residual info-gain).

        Always clicks a real pixel of the chosen segment (critical:
        ACTION6 to (0,0) is a known fatal no-op without set_data; we
        also must hit an actual segment pixel)."""
        if not slots:
            return (32, 32), 0.0

        scored = []
        for ps in slots:
            px = ps["state"]
            # key the frontier by frame-state AND object position so
            # every distinct on-screen object is probed once (many
            # objects share a (colour,w,h) signature -> position is
            # required to make the systematic sweep actually cover them)
            key = (fh, int(round(px[0])), int(round(px[1])),
                   int(px[2]))
            tried = self.click_tried.get(key, 0)
            effective = self.click_effect.get(key, False)

            dxn, dyn, _ = self._nearest(px, slots, ps["ci"])
            feat = RelationalMM._feat(px[0], px[1], dxn, dyn)
            if self.rmm.mu:
                ll = self.rmm._ll(feat)
                m = int(np.argmax(ll))
                model_v = (self.rmm.expected_reward(m)
                           + self.INFO_W * self.rmm.dirichlet_infogain(m))
            else:
                model_v = 0.0

            if effective:
                # known-useful object: exploit, modulated by model value
                sc = 10.0 + model_v
            elif tried == 0:
                # unprobed object: maximal info-gain (systematic sweep)
                sc = 5.0 + 0.001 * float(ps["comp"]["area"])
            else:
                # diminishing info-gain with repetition
                sc = model_v + 1.0 / (1.0 + tried)
            if int(px[2]) in self.fp.salient_color:
                sc += 0.25
            scored.append((sc, key, ps))

        scored.sort(key=lambda t: t[0], reverse=True)
        best_sc, best_key, best = scored[0]
        self.click_tried[best_key] = self.click_tried.get(best_key, 0) + 1
        self.last_click_key = best_key

        ci = best["ci"]
        pts = np.argwhere(label_map == ci)
        if len(pts) == 0:
            return (int(best["state"][0]), int(best["state"][1])), best_sc
        # click the segment centroid pixel when possible (stable target)
        cyx = pts.mean(axis=0)
        d2 = np.sum((pts - cyx) ** 2, axis=1)
        p = pts[int(np.argmin(d2))]
        return (int(p[1]), int(p[0])), best_sc      # (x, y)
