"""Build v33 = v32 + expanded action scan (attacks queue-exhaustion games).

Diagnostic showed bp35/g50t/sb26 exhaust BFS with 1-321 unique states because
_scan_actions prunes (a) ACTION7/undo via the `a<=5` filter and (b) every
valid click that has no IMMEDIATE L0 frame effect — but those are exactly the
arm/unlock actions these games require. v33:
  - include ACTION7 (undo) as a bare action when available
  - also keep up to K no-immediate-effect valid clicks (armed actions),
    capped to bound branching factor
"""
from pathlib import Path

P = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v33_agent.py")
src = P.read_text(encoding="utf-8")

# 1) Include ACTION7 (undo) alongside simple actions 1-5
old1 = "        for a in [a for a in avail if a <= 5]:\n            actions.append((a, None))"
new1 = ("        # v33: include ACTION7 (undo) — essential arm/revert action in\n"
        "        # several games; the old `a<=5` filter silently dropped it.\n"
        "        for a in [a for a in avail if a <= 5 or a == 7]:\n"
        "            actions.append((a, None))")
assert old1 in src, "anchor 1 (simple-action loop) not found"
src = src.replace(old1, new1, 1)

# 2) In the _get_valid_actions click loop, collect no-effect clicks too (capped).
#    Insert an armed-click collector + a post-loop append of a capped subset.
old2 = """            if hasattr(game, '_get_valid_actions'):
                try:
                    valid = game._get_valid_actions()
                    for ai_obj in valid:
                        act_id = ai_obj.id._value_ if hasattr(ai_obj.id, '_value_') else int(ai_obj.id)
                        if act_id == 6:
                            g = copy.deepcopy(game)
                            try:
                                r = g.perform_action(ai_obj, raw=True)
                                if r.frame:
                                    f = np.array(r.frame[-1])
                                    diff = np.sum(f0 != f)
                                    if diff > 0:
                                        eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                        if eh not in seen_effects:
                                            seen_effects.add(eh)
                                            actions.append((6, ai_obj.data))
                                        # v19: record productive click into cross-game memory
                                        if self.cgm is not None and isinstance(ai_obj.data, dict):
                                            x, y = ai_obj.data.get('x'), ai_obj.data.get('y')
                                            if x is not None and y is not None:
                                                self.cgm['productive_clicks'].append((x, y))
                                                if len(self.cgm['productive_clicks']) > 200:
                                                    self.cgm['productive_clicks'] = self.cgm['productive_clicks'][-100:]
                            except:
                                pass
                except:
                    pass"""
new2 = """            armed_clicks = []  # v33: valid clicks with NO immediate L0 effect
            ARMED_CAP = 24     # bound branching: keep at most this many
            if hasattr(game, '_get_valid_actions'):
                try:
                    valid = game._get_valid_actions()
                    for ai_obj in valid:
                        act_id = ai_obj.id._value_ if hasattr(ai_obj.id, '_value_') else int(ai_obj.id)
                        if act_id == 6:
                            g = copy.deepcopy(game)
                            try:
                                r = g.perform_action(ai_obj, raw=True)
                                if r.frame:
                                    f = np.array(r.frame[-1])
                                    diff = np.sum(f0 != f)
                                    if diff > 0:
                                        eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                        if eh not in seen_effects:
                                            seen_effects.add(eh)
                                            actions.append((6, ai_obj.data))
                                        if self.cgm is not None and isinstance(ai_obj.data, dict):
                                            x, y = ai_obj.data.get('x'), ai_obj.data.get('y')
                                            if x is not None and y is not None:
                                                self.cgm['productive_clicks'].append((x, y))
                                                if len(self.cgm['productive_clicks']) > 200:
                                                    self.cgm['productive_clicks'] = self.cgm['productive_clicks'][-100:]
                                    elif len(armed_clicks) < ARMED_CAP * 3:
                                        # No immediate effect: candidate ARM action.
                                        armed_clicks.append(ai_obj.data)
                            except:
                                pass
                except:
                    pass
            # v33: if scan found few/no effective clicks, the game likely
            # needs an arm-then-act sequence. Add a capped set of no-effect
            # valid clicks so BFS can discover the unlock.
            if armed_clicks and len(seen_effects) <= 4:
                step = max(1, len(armed_clicks) // ARMED_CAP)
                for d in armed_clicks[::step][:ARMED_CAP]:
                    actions.append((6, d))"""
assert old2 in src, "anchor 2 (valid-actions click loop) not found"
src = src.replace(old2, new2, 1)

P.write_text(src, encoding="utf-8")
print(f"v33 built: {src.count(chr(10))} lines, action7={'a == 7' in src}, "
      f"armed_clicks={'armed_clicks' in src}")
