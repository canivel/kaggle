"""v38 = v37 + GE multi-restart within per-game budget."""
from pathlib import Path

P = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v38_agent.py")
src = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v37_agent.py").read_text(encoding="utf-8")

# 1) init restart state in __init__ — anchor includes the _ge_failed line so it
#    matches ONLY the __init__ occurrence (the other _ge_level=-1 has a comment).
old1 = "        s._ge_level = -1\n        s._ge_failed = False"
new1 = (
    "        s._ge_level = -1\n"
    "        s._ge_failed = False\n"
    "        s._ge_base_seed = int(hashlib.md5(str(s.game_id).encode()).hexdigest()[:8], 16)\n"
    "        s._ge_restart_idx = 0\n"
    "        s._ge_actions_since_restart = 0\n"
    "        s._ge_levels_at_restart = 0\n"
    "        s._GE_RESTART_BUDGET = 2000  # actions w/o level progress -> reseed"
)
assert src.count(old1) == 1, f"anchor1 count={src.count(old1)}"
src = src.replace(old1, new1, 1)

# 2) level-change block in _ge_pick: reset restart bookkeeping + per-tick check.
old2 = (
    "        if lvl != s._ge_level:\n"
    "            s._ge.reset()\n"
    "            s._ge_status_mask = None\n"
    "            s._ge_last_hash = None\n"
    "            s._ge_last_action_id = None\n"
    "            s._ge_level = lvl"
)
new2 = (
    "        if lvl != s._ge_level:\n"
    "            s._ge.reset()\n"
    "            s._ge_status_mask = None\n"
    "            s._ge_last_hash = None\n"
    "            s._ge_last_action_id = None\n"
    "            s._ge_level = lvl\n"
    "            s._ge_restart_idx = 0\n"
    "            s._ge_actions_since_restart = 0\n"
    "            s._ge_levels_at_restart = lvl\n"
    "            random.seed(s._ge_base_seed)\n"
    "            np.random.seed(s._ge_base_seed % (2**32 - 1))\n"
    "\n"
    "        # v38: GE multi-restart. No level progress within budget ->\n"
    "        # deterministic re-seed + explorer reset (best-of-K, reproducible).\n"
    "        s._ge_actions_since_restart += 1\n"
    "        if lvl > s._ge_levels_at_restart:\n"
    "            s._ge_levels_at_restart = lvl\n"
    "            s._ge_actions_since_restart = 0\n"
    "            s._ge_restart_idx = 0\n"
    "        elif s._ge_actions_since_restart >= s._GE_RESTART_BUDGET:\n"
    "            s._ge_restart_idx += 1\n"
    "            s._ge_actions_since_restart = 0\n"
    "            rs = (s._ge_base_seed + s._ge_restart_idx * 7919) % (2**32 - 1)\n"
    "            random.seed(rs)\n"
    "            np.random.seed(rs)\n"
    "            s._ge.reset()\n"
    "            s._ge_status_mask = None\n"
    "            s._ge_last_hash = None\n"
    "            s._ge_last_action_id = None\n"
    "            logger.info(f\"v38 GE restart #{s._ge_restart_idx} seed={rs} L{lvl}\")"
)
assert src.count(old2) == 1, f"anchor2 count={src.count(old2)}"
src = src.replace(old2, new2, 1)

P.write_text(src, encoding="utf-8")
import ast
ast.parse(src)
print(f"v38 built: {src.count(chr(10))} lines, syntax OK, "
      f"restart={'GE restart #' in src}, budget={'_GE_RESTART_BUDGET' in src}")
