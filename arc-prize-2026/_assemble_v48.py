"""Assemble v48 = v39 BFS infra + v47 SG ActionModel + new hybrid MyAgent."""
import sys

with open('notebooks/forge_agent/v39_agent.py') as f:
    v39 = f.read()
with open('notebooks/forge_agent/v47_agent.py') as f:
    v47 = f.read()

v39_lines = v39.split('\n')
bfs_block = '\n'.join(v39_lines[:1062])

v47_lines = v47.split('\n')
am_start = next(i for i, l in enumerate(v47_lines) if l.startswith('class ActionModel'))
am_end = next(i for i in range(am_start, len(v47_lines)) if v47_lines[i].startswith('class MyAgent'))
action_model_block = '\n'.join(v47_lines[am_start:am_end])

HEADER = (
    "# =====================================================================\n"
    "# v48 = BFS + SG-CNN hybrid\n"
    "#   - v39 full BFS infra (BFSSolver, _fast_deepcopy, find_game_source_and_class)\n"
    "#   - StochasticGoose ActionModel (16-ch one-hot, 4-conv backbone, 5+4096 action head)\n"
    "#   - choose_action: if BFS solves the level execute BFS plan; else SG-CNN online learning\n"
    "#   - NO GraphExplorer (replaces v35 GE fallback with SG-CNN)\n"
    "# =====================================================================\n"
)

MYAGENT = '''
class MyAgent(Agent):
    """BFS + SG-CNN hybrid."""
    MAX_ACTIONS = float('inf')

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(hashlib.md5(str(s.game_id).encode()).hexdigest()[:8], 16)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed % (2 ** 32 - 1))

        s.start_time = time.time()
        s.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s.grid_size = 64
        s.num_coordinates = 64 * 64
        s.num_colours = 16

        s._bfs = None
        s._bfs_tried = False
        s._bfs_solved_last = False
        s._bfs_solution = None
        s._bfs_step = 0
        s.cl = -1

        s.action_model = None
        s.optimizer = None
        s.experience_buffer = deque(maxlen=200000)
        s.experience_hashes = set()
        s.batch_size = 64
        s.train_frequency = 5
        s.prev_frame = None
        s.prev_action_idx = None
        s.current_score = -1

        s.action_list = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                         GameAction.ACTION4, GameAction.ACTION5]

    def _init_bfs(s):
        src, cls = find_game_source_and_class(s.game_id, s.arc_env if hasattr(s, "arc_env") else None)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180)
            try:
                s._bfs.load()
            except Exception:
                pass

    def _try_bfs_solve(s, level_idx):
        if s._bfs is None or s._bfs.game_cls is None:
            return None
        try:
            sol = s._bfs.solve_level(level_idx, max_states=200000)
        except Exception:
            sol = None
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            s._bfs_solved_last = True
        else:
            s._bfs_solved_last = False
        return sol

    def _frame_to_tensor(s, fd):
        frame = np.array(fd.frame, dtype=np.int64)[-1]
        if frame.shape != (s.grid_size, s.grid_size):
            raise RuntimeError("frame shape " + str(frame.shape))
        frame = np.clip(frame, 0, s.num_colours - 1)
        tensor = torch.zeros(s.num_colours, s.grid_size, s.grid_size, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        return tensor.to(s.device)

    def _experience_hash(s, frame_np, action_idx):
        return hashlib.md5(frame_np.tobytes() + str(action_idx).encode()).hexdigest()

    def _sample_sg(s, combined_logits, available_actions):
        action_logits = combined_logits[:5].clone()
        coord_logits = combined_logits[5:].clone()
        action6_available = False
        action_mask = torch.full_like(action_logits, float("-inf"))
        if available_actions:
            for a in available_actions:
                av = a.value if hasattr(a, "value") else int(a)
                if 1 <= av <= 5:
                    action_mask[av - 1] = 0.0
                elif av == 6:
                    action6_available = True
            action_logits = action_logits + action_mask
            if not action6_available:
                coord_logits = coord_logits + torch.full_like(coord_logits, float("-inf"))
        action_probs = torch.sigmoid(action_logits)
        coord_probs = torch.sigmoid(coord_logits) / s.num_coordinates
        all_probs = torch.cat([action_probs, coord_probs])
        total = all_probs.sum()
        if not torch.isfinite(total) or total <= 0:
            valid = [i for i in range(5) if action_mask[i] == 0.0]
            return (random.choice(valid) if valid else 0), None
        all_probs = all_probs / total
        idx = int(np.random.choice(len(all_probs), p=all_probs.cpu().numpy()))
        if idx < 5:
            return idx, None
        c = idx - 5
        return 5, (c // s.grid_size, c % s.grid_size)

    def _train_sg(s):
        if len(s.experience_buffer) < s.batch_size:
            return
        idxs = np.random.choice(len(s.experience_buffer), s.batch_size, replace=False)
        batch = [s.experience_buffer[i] for i in idxs]
        states = torch.stack([torch.from_numpy(e["state"]).float().to(s.device) for e in batch])
        action_indices = torch.tensor([e["action_idx"] for e in batch], dtype=torch.long, device=s.device)
        rewards = torch.tensor([e["reward"] for e in batch], dtype=torch.float32, device=s.device)
        s.optimizer.zero_grad()
        logits = s.action_model(states)
        selected = logits.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        main_loss = F.binary_cross_entropy_with_logits(selected, rewards)
        all_probs = torch.sigmoid(logits)
        loss = main_loss - 0.0001 * all_probs[:, :5].mean() - 0.00001 * all_probs[:, 5:].mean()
        loss.backward()
        s.optimizer.step()

    def _reset_sg_for_level(s):
        s.experience_buffer.clear()
        s.experience_hashes.clear()
        s.action_model = ActionModel(input_channels=s.num_colours, grid_size=s.grid_size).to(s.device)
        s.optimizer = optim.Adam(s.action_model.parameters(), lr=0.0001)
        s.prev_frame = None
        s.prev_action_idx = None

    def _lvl(s, f):
        return getattr(f, "score", None) or f.levels_completed

    def is_done(s, frames, lf):
        return (lf.state is GameState.WIN
                or (time.time() - s.start_time) >= 8 * 3600 - 300)

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)
            if lvl != s.cl:
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()
                s._bfs_solution = None
                s._bfs_step = 0
                if s._bfs:
                    s._try_bfs_solve(lvl)
                s._reset_sg_for_level()
                s.cl = lvl
                s.current_score = lvl

            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.prev_frame = None
                s.prev_action_idx = None
                a = GameAction.RESET
                a.reasoning = "reset"
                return a

            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                if data:
                    sel.set_data(data)
                sel.reasoning = "bfs:" + str(s._bfs_step) + "/" + str(len(s._bfs_solution))
                return sel

            cur_tensor = s._frame_to_tensor(lf)
            cur_np = cur_tensor.cpu().numpy().astype(bool)
            if s.prev_frame is not None and s.prev_action_idx is not None:
                eh = s._experience_hash(s.prev_frame, s.prev_action_idx)
                if eh not in s.experience_hashes:
                    frame_changed = not np.array_equal(s.prev_frame, cur_np)
                    s.experience_buffer.append({
                        "state": s.prev_frame, "action_idx": s.prev_action_idx,
                        "reward": 1.0 if frame_changed else 0.0})
                    s.experience_hashes.add(eh)
            avail = getattr(lf, "available_actions", None) or []
            with torch.no_grad():
                logits = s.action_model(cur_tensor.unsqueeze(0)).squeeze(0)
            aidx, coords = s._sample_sg(logits, avail)
            if aidx < 5:
                sel = s.action_list[aidx]
                sel.reasoning = "sg:a" + str(aidx + 1)
                unified_idx = aidx
            else:
                sel = GameAction.ACTION6
                y, x = coords
                sel.set_data({"x": int(x), "y": int(y)})
                sel.reasoning = "sg:click(" + str(x) + "," + str(y) + ")"
                unified_idx = 5 + (y * s.grid_size + x)
            s.prev_frame = cur_np
            s.prev_action_idx = unified_idx
            if s.action_counter % s.train_frequency == 0:
                s._train_sg()
            return sel
        except Exception as e:
            traceback.print_exc()
            a = random.choice(s.action_list)
            a.reasoning = "err:" + str(e)[:40]
            return a
'''

out = HEADER + bfs_block + '\n\n' + action_model_block + '\n\n' + MYAGENT
with open('notebooks/forge_agent/v48_agent.py', 'w') as f:
    f.write(out)
print('v48 lines:', out.count('\n'))
