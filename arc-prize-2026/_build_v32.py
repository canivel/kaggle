"""Build v32 = v22 + FrameProcessor status-bar masking in BFS state hash
   + relaxed transient-field detection (>=80% instead of 100%).

Attacks the dominant zero-BFS failure: rendered step-counters in frame
pixels make every state hash unique -> dedup never fires -> A* explodes.
"""
from pathlib import Path

V32 = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v32_agent.py")
src = V32.read_text(encoding="utf-8")
fp_block = Path("/tmp/_fp_block.py").read_text(encoding="utf-8")

# 1) Inject FrameProcessor class just before "class BFSSolver:"
assert "class BFSSolver:" in src
src = src.replace(
    "class BFSSolver:",
    "# ===== FrameProcessor (status-bar detection, from v24b graph-explorer) =====\n"
    + fp_block + "\n\n# v32: shared FrameProcessor instance for status-bar masking\n"
    + "_V32_FP = FrameProcessor()\n\n\nclass BFSSolver:",
    1,
)

# 2) Add status-bar mask init in BFSSolver.__init__ (after self.bfs_timeout = ...)
anchor = "self.bfs_timeout = bfs_timeout"
assert anchor in src
src = src.replace(
    anchor,
    anchor + "\n        self._sb_mask = None  # v32: status-bar pixel mask, computed once",
    1,
)

# 3) Rewrite _state_hash to mask status-bar pixels before md5
old_hash = '''    def _state_hash(self, g, frame, hidden_fields=None, transient_fields=None):
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]'''
new_hash = '''    def _ensure_sb_mask(self, frame):
        """Compute the status-bar pixel mask once from the first frame seen.
        Rendered step-counters/animation digits live here; masking them
        prevents state-hash explosion (every step otherwise looks unique)."""
        if self._sb_mask is not None:
            return
        try:
            seg, segs = _V32_FP.segment_frame(np.asarray(frame, dtype=np.uint8))
            _, mask = _V32_FP.identify_status_bars(seg, segs)
            if mask is not None and mask.shape == frame.shape:
                self._sb_mask = mask.astype(bool)
            else:
                self._sb_mask = np.zeros(frame.shape, dtype=bool)
        except Exception:
            self._sb_mask = np.zeros(np.asarray(frame).shape, dtype=bool)

    def _state_hash(self, g, frame, hidden_fields=None, transient_fields=None):
        self._ensure_sb_mask(frame)
        if self._sb_mask is not None and self._sb_mask.shape == frame.shape and self._sb_mask.any():
            fmask = frame.copy()
            fmask[self._sb_mask] = 0
            fh = hashlib.md5(fmask.tobytes()).hexdigest()[:16]
        else:
            fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]'''
assert old_hash in src, "state_hash anchor not found"
src = src.replace(old_hash, new_hash, 1)

# 4) Relax transient detection: a field changing on >=80% of sampled actions
#    (was: exactly 100%) is a monotone budget counter -> exclude from hash.
old_tr = '''        transient = set()
        for k, cnt in changed_count.items():
            if cnt != n_sampled:
                continue
            v = initial[k]
            if isinstance(v, bool):
                continue  # boolean flags are meaningful state, never transient
            transient.add(k)'''
new_tr = '''        transient = set()
        for k, cnt in changed_count.items():
            v = initial[k]
            if isinstance(v, bool):
                continue  # boolean flags are meaningful state, never transient
            # v32: monotone budget counters change on MOST actions (not always
            # exactly all — some no-op actions skip the tick). >=80% => transient.
            if cnt >= max(1, int(0.8 * n_sampled)):
                transient.add(k)'''
assert old_tr in src, "transient anchor not found"
src = src.replace(old_tr, new_tr, 1)

V32.write_text(src, encoding="utf-8")
print(f"v32 built: {src.count(chr(10))} lines, FrameProcessor={'class FrameProcessor' in src}, "
      f"sb_mask={'_ensure_sb_mask' in src}, relaxed_transient={'>=80%' in src or '0.8 * n_sampled' in src}")
