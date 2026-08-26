import json
import numpy as np
import hashlib
import os, sys
from tqdm import tqdm
from glob import glob
import itertools
import random
from visualize import *
from pp import *
from collections import deque
from copy import deepcopy
from collections import defaultdict

def cut_at_token(output, token_id):
    eos_positions = (output==token_id).nonzero()[0]
    return output[:eos_positions[0]] if len(eos_positions) else output

def shuffled(data_list):
    return np.random.permutation(data_list).tolist()

def permute_mod(a, descriptor, invert=False):
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    assert sorted(permutation) == list(range(10))
    a = np.asarray(a)
    
    if a.ndim == 3:
        if not invert:
            permutation = np.argsort(permutation)
        a = a[..., permutation]
    
    else:
        assert a.ndim == 2
        if invert:
            permutation = np.argsort(permutation)
        perm_array = np.asarray(permutation)

        # Only apply permutation to elements not equal to -1
        mask = (a != -1) & (a != 10)
        a[mask] = perm_array[a[mask]]

    return a

def permute_rnd_col_(query):
    permutation = [0]+(1+np.random.permutation(9)).tolist()
    return 'permute' + ''.join(map(str, permutation))

def permute_rnd_all_(query):
    permutation = np.random.permutation(10).tolist()
    return 'permute' + ''.join(map(str, permutation))

def permute_cnt_col_(query):
    elements, frequency = np.unique(np.concatenate([list(range(10))]+[np.array(x['input']).ravel() for x in query['train']]), return_counts=True)
    permutation = [0]+sorted(np.random.permutation(9)+1, key=lambda i: frequency[i], reverse=True)  # randomness as tie breaker
    return 'permute' + ''.join(map(str, permutation))

def permute_cnt_all_(query):
    elements, frequency = np.unique(np.concatenate([list(range(10))]+[np.array(x['input']).ravel() for x in query['train']]), return_counts=True)
    permutation = sorted(np.random.permutation(10), key=lambda i: frequency[i], reverse=True)  # randomness as tie breaker
    return 'permute' + ''.join(map(str, permutation))

permute_rnd_col = (permute_mod, permute_rnd_col_)
permute_rnd_all = (permute_mod, permute_rnd_all_)
permute_cnt_col = (permute_mod, permute_cnt_col_)
permute_cnt_all = (permute_mod, permute_cnt_all_)
permute_None = (np.copy, None)

class ArcDataset(object):
    # ----------------------------
    # OBJECT FINDING
    # ----------------------------
    def find_objects(self,grid,diagonal=True,bg=None):
        """Extract connected components and row-stack them"""
        grid = np.array(grid)
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)

        if diagonal:
            directions = [(1,0), (-1,0), (0,1), (0,-1),
                          (1,1), (1,-1), (-1,1), (-1,-1)]
        else:
            directions = [(1,0), (-1,0), (0,1), (0,-1)]

        objects = []

        def bfs(sr, sc):
            q = deque([(sr, sc)])
            coords = []
            values = []
            v = grid[sr, sc]
            visited[sr, sc] = True
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                values.append(grid[r, c])
                for dr, dc in directions:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == v:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if bg is not None and v==bg: return
            # normalize
            min_r, min_c = min(rr for rr, _ in coords), min(cc for _, cc in coords)
            rel_coords = [(rr-min_r, cc-min_c) for rr, cc in coords]
            obj_h = max(rr for rr, _ in rel_coords) + 1
            obj_w = max(cc for _, cc in rel_coords) + 1
            obj_grid = np.full((obj_h, obj_w), -1, dtype=int)
            for (rr, cc), val in zip(rel_coords, values):
                obj_grid[rr, cc] = val
            return obj_grid

        for r in range(h):
            for c in range(w):
                if not visited[r, c]:
                    obj = bfs(r, c)
                    if bg is not None and grid[r][c]==bg:
                        continue
                    objects.append(obj)
        return objects

    # ----------------------------
    # PER-EXAMPLE BACKGROUND DETECTION
    # ----------------------------
    def detect_background(self, example_item):
        """
        Detect background value for a single example item (dict with 'input'/'output').
        Returns background value if full-size object exists, else None.
        """
        candidate = None
        for gkey in ["input", "output"]:
            if gkey in example_item:
                grid = np.array(example_item[gkey])
                objs = self.find_objects(grid)
                full_objs = [o for o in objs if o.shape == grid.shape]
                if len(full_objs)>1: return None
                if not full_objs:
                    if 'output' in example_item and gkey=='input' and grid.shape!=np.array(example_item['output']).shape: return None
                    continue  # no full-grid object
                val = np.unique(full_objs[0][full_objs[0] != -1])
                if len(val) == 1:
                    row_mask = np.where(np.all(full_objs[0] == val[0], axis=1))[0]
                    uniq_diff = np.unique(np.diff(row_mask))
                    if len(row_mask)>2 and len(uniq_diff)==1 and uniq_diff!=1: return None
                    col_mask = np.where(np.all(full_objs[0] == val[0], axis=0))[0]
                    uniq_diff = np.unique(np.diff(col_mask))
                    if len(col_mask)>2 and len(uniq_diff)==1 and uniq_diff!=1: return None
                    if candidate is None:
                        candidate = val[0]
                        if 'output' not in example_item: return candidate
                        elif gkey=='input' and grid.shape!=np.array(example_item['output']).shape: return candidate
                    elif candidate != val[0]:
                        return None  # inconsistent values
                else:
                    return None  # multiple values in full-grid object
        return candidate

    # ----------------------------
    # PAD OBJECTS VERTICALLY
    # ----------------------------
    def pad_between_objects(self, objects, pad_val):
        """Stack objects vertically with one row of padding between them."""
        if not objects:
            return np.array([[pad_val]])

        max_w = max(obj.shape[1] for obj in objects)
        padded = []

        for i, obj in enumerate(objects):
            # pad width to max_w
            h, w = obj.shape
            obj_padded = np.pad(obj, ((0, 0), (0, max_w - w)), constant_values=pad_val)
            obj_padded[obj_padded==-1] = pad_val
            padded.append(obj_padded)

            # add one row of padding between objects
            if i < len(objects) - 1:
                sep = np.full((1, max_w), pad_val)
                padded.append(sep)

        return np.vstack(padded)

    # ----------------------------
    # CUT BACKGROUND LINES/COLUMNS
    # ----------------------------
    def collapse_bg_lines(self, grid, bg_val=0):
        # --- rows ---
        row_is_bg = np.all(grid == bg_val, axis=1)
        keep_rows = []
        prev_bg = False
        for i, is_bg in enumerate(row_is_bg):
            if not is_bg:
                keep_rows.append(i)  # always keep non-bg row
                prev_bg = False
            else:
                if not prev_bg:
                    keep_rows.append(i)  # keep only first of a bg run
                prev_bg = True
        
        # grid = grid[keep_rows, :]
    
        # --- cols ---
        col_is_bg = np.all(grid == bg_val, axis=0)
        keep_cols = []
        prev_bg = False
        for j, is_bg in enumerate(col_is_bg):
            if not is_bg:
                keep_cols.append(j)
                prev_bg = False
            else:
                if not prev_bg:
                    keep_cols.append(j)
                prev_bg = True
    
        # grid = grid[:, keep_cols]
        return keep_rows, keep_cols

    def cut_background(self, in_grid, out_grid, bg_val):
        """Cut rows/cols consisting only of bg_val according to rules."""
        in_grid, out_grid = np.array(in_grid), np.array(out_grid)

        if in_grid.shape != out_grid.shape:
            keep_rows, keep_cols = self.collapse_bg_lines(in_grid,bg_val)
            in_cut = in_grid[keep_rows,:]
            in_cut = in_cut[:,keep_cols]
            out_cut = out_grid
        else:
            in_keep_rows, in_keep_cols = self.collapse_bg_lines(in_grid,bg_val)
            out_keep_rows, out_keep_cols = self.collapse_bg_lines(out_grid,bg_val)
            keep_rows = sorted(set(in_keep_rows)|set(out_keep_rows))
            keep_cols = sorted(set(in_keep_cols)|set(out_keep_cols))
            in_cut = in_grid[keep_rows,:]
            in_cut = in_cut[:,keep_cols]
            out_cut = out_grid[keep_rows,:]
            out_cut = out_cut[:,keep_cols]

        return in_cut, out_cut

    # ----------------------------
    # MAKE OBJECT DATASET
    # ----------------------------
    def make_object_dataset(self, make=False, pad='bg'):
        """
        Build object dataset: only objects stacked, one row of padding.
        Original grid excluded. Uses per-example background for padding.
        """
        obj_ds = defaultdict(lambda: {"train": [], "test": []})
        keys = []

        for key, example in self.queries.items():
            keys.append(key)
            for split in ["train", "test"]:
                for idx, item in enumerate(example.get(split, [])):
                    bg_val = self.detect_background(item)
                    pad_val = self.background_values[(key, split, idx)] = bg_val if bg_val is not None else -1
                    if pad==-1:pad_val=-1

                    if pp and make:
                        new_item = {}
                        for gkey in ["input", "output"]:
                            if gkey in item:
                                objs = self.find_objects(item[gkey],bg=bg_val)
                                full_grid_shape = np.array(item[gkey]).shape
                                # exclude object covering full original grid
                                # objs = [o for o in objs if o.shape != full_grid_shape]
    
                                if objs:
                                    obj_grid = self.pad_between_objects(objs, pad_val)
                                else:
                                    obj_grid = np.array([[pad_val]])
    
                                new_item[gkey] = obj_grid.tolist()
                        obj_ds[key][split].append(new_item)

        # return obj_ds
        if pp and make:
            return self.__class__(queries=obj_ds, keys=keys, background_values=self.background_values)
        else:
            return self.__class__(queries=self.queries, keys=keys, background_values=self.background_values)

    # ----------------------------
    # MAKE BACKGROUND-CUT DATASET
    # ----------------------------
    def make_background_cut_dataset(self):
        """
        Build background-cut dataset. Removes background-only rows/cols
        using per-example background values.
        """
        bg_cut_ds = defaultdict(lambda: {"train": [], "test": []})
        keys = []
        # print(self.background_values)
        
        for key, example in self.queries.items():
            keys.append(key)
            for split in ["train", "test"]:
                for idx, item in enumerate(example.get(split, [])):
                    bg_val = self.background_values.get((key, split, idx), None)
                    if bg_val is None:
                        # break
                        # no background detected, copy original
                        new_item = deepcopy(item)
                    else:
                        if split == "train":
                            new_in, new_out = self.cut_background(item["input"], item["output"], bg_val)
                            new_item = {"input": new_in.tolist(), "output": new_out.tolist()}
                        else:
                            new_item = deepcopy(item)
                    bg_cut_ds[key][split].append(new_item)

        return self.__class__(queries=dict(bg_cut_ds), keys=keys, background_values=self.background_values)

    @staticmethod
    def forward_mod(a, key, use_perm=True, is_output=True):
        if a is None: return a
        for op in key.split('.')[1:]:
            if op.startswith('I'):
                if is_output: continue
                op = op[1:]
            if   op=='rot90':              a = np.rot90(a)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('dif'):     a[a==-1] = int(op[-1])
            elif op.startswith('permute'): a = permute_mod(a, op, invert=False) if use_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('c'):       a = a
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('fix'):     a = a
            elif op.startswith('ice'):     a = a  # for adding icecuber solutions
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    @staticmethod
    def invert_mod(a, key, inv_perm=True, is_output=True):
        if a is None: return a
        for op in key.split('.')[1:][::-1]:
            if op.startswith('I'):
                if is_output: continue
                op = op[1:]
            if   op=='rot90':              a = np.rot90(np.rot90(np.rot90(a)))
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('dif'):     a[a==int(op[-1])] = -1
            elif op.startswith('permute'): a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('c'):       a = a
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('fix'):     a = a
            elif op.startswith('ice'):     a = a  # for adding icecuber solutions
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    def __init__(self, queries, replies={}, keys=None, is_orig=False, is_fake=False, background_values={}):
        if keys is not None: keys = [k for k in keys if k is not None]
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.is_orig = is_orig
        self.is_fake = is_fake
        self.keys = sorted(queries.keys()) if keys is None else keys
        self.faulty = {}
        self.transposed_dataset = None
        self.background_values = background_values

    @classmethod
    def empty(cls):
        return cls(queries={}, replies={}, keys=[])

    def change_keys(self, keys, keep_flags=False):
        flags = dict(is_fake=self.is_fake, is_orig=self.is_orig) if keep_flags else {}
        return self.__class__(queries=self.queries, replies=self.replies, keys=keys, **flags)

    @classmethod
    def from_file(cls, queries_file, start=0, limit_n=20, get_diff=False):
        import numpy as np
        print(f"*** Load challanges from '{queries_file}'...")
        with open(queries_file) as f: queries = f.read()
        import os
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'): #Real submit
            is_fake = False
        else: #Fake run
            is_fake = True
        #is_fake = hashlib.md5(queries.encode('utf-8')).hexdigest().lower()=='a6b7dac3cab03abf2eb333e16610d6dc'
        if is_fake: print("*** -> Fake test set detected, setting flag 'is_fake' to True.")
        queries = json.loads(queries)
        if is_fake:
            start = start
            limit_n = limit_n
            print(f'From {len(queries)} to load only {limit_n}')
            # queries = {k:v for n,(k,v) in enumerate(queries.items()) if start <= n < start + limit_n}
            with open("/kaggle/input/arc2024ev-sol/arc-agi_evaluation_solutions.json",'r') as f:
                ev2024 = json.load(f)
            only2025 = {}
            for i,(k,v) in enumerate(queries.items()):
                # if i < start:
                #     continue
                if k not in ev2024:
                    only2025[k] = v
            print('only2025 len',len(only2025))
            new_queries = {}
            if 'training' not in queries_file: 
                queries = only2025
            only2025count = 0
            non_det_size_count = 0
            has_same_value_count = 0

            for i,(k,v) in enumerate(queries.items()):
                if i < start:
                    continue
                if len(new_queries)==limit_n:
                    break
                if get_same_value_count:
                    all_ok = True
                    shared_counts_per_train = []
                
                    for train in v['train']:
                        in_arr = np.array(train['input'])
                        out_arr = np.array(train['output'])
                
                        shared_counts = {}
                        for val in range(10):
                            in_count = np.sum(in_arr == val)
                            out_count = np.sum(out_arr == val)
                            if in_count > 0 and in_count == out_count:
                                shared_counts[val] = in_count
                
                        if not shared_counts:
                            all_ok = False
                            break
                        shared_counts_per_train.append(shared_counts)
                
                    if all_ok and k not in ev2024:
                        has_same_value_count += 1
                        new_queries[k] = v
                        # print(f"{k}: {shared_counts_per_train}")
                        # visualize_task(k, title=f'{k}:{shared_counts_per_train}')
                    continue
                # get_det_size = True
                # if get_det_size:
                    # if not all(np.array(w['input']).shape==np.array(w['output']).shape for w in v['train']):
                    #     non_det_size_count += 1
                    #     continue
                    # if not all(np.array(w['output']).shape==np.array(v['train'][0]['output']).shape for w in v['train']):
                    #     non_det_size_count += 1
                    #     continue
                if k not in ev2024 or 'training' in queries_file:
                    if get_diff and all(np.array(w['input']).shape==np.array(w['output']).shape for w in v['train']):
                        for t in v['train']:
                            in_arr = np.array(t['input'])
                            out_arr = np.array(t['output'])
                            out_arr[out_arr==in_arr] = -1
                            t['output'] = out_arr.tolist()
                        
                    new_queries[k] = v
                    if k not in ev2024:
                        only2025count += 1
            queries = new_queries
            if pp:print('loaded',len(queries),'of which only 2025 is',only2025count,'skipped non_det_size',non_det_size_count,'same_value_count',has_same_value_count)
        return cls(
            queries=queries,
            is_fake=is_fake,
            is_orig=True,
        )

    def load_replies(self, replies_file):
        print(f"*** Load solutions from '{replies_file}'...")
        with open(replies_file) as f: replies = f.read()
        replies_parsed = json.loads(replies)
        self.replies = {k: replies_parsed[k] for k in self.keys}
        return self

    def split_multi_replies(self, train=False):
        key_indices = [(k, i) for k in self.keys for i in range(len(self.queries[k]['test']))]
        if not train:
            return self.__class__(
                keys=[f'{k}_{i}' for k, i in key_indices],
                queries={f'{k}_{i}': {'train': self.queries[k]['train'], 'test': [self.queries[k]['test'][i]]} for k, i in key_indices},
                replies={f'{k}_{i}': [self.replies[k][i]] for k, i in key_indices if k in self.replies},
            )
        else:
            new_queries = {}
            new_replies = {}
            new_keys = []
        
            for k,i in key_indices:
                bk,aug = k.split('.',1)
                bk = bk.split('_')[0]
                t_ind = int(aug[-2])
                new_k = f'{bk}_{9-t_ind}.{aug}'
                new_queries[new_k] = {
                    'train': self.queries[k]['train'],
                    'test': [self.queries[k]['test'][i]],
                }
                new_replies[new_k] = [self.replies[k][i]]
                new_keys.append(new_k)
        
            return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def move_test_to_train(self, only_when_2=False):
        new_queries = {k: {'train': self.queries[k]['train'] if only_when_2 and len(self.queries[k]['train'])>2 else self.queries[k]['train'] + [{**t, 'output': self.replies[k][i]} for i, t in enumerate(self.queries[k]['test'])], 'test': []} for k in self.keys}
        return self.__class__(queries=new_queries, keys=[k for k in self.keys])

    def last_train_ex_for_test(self):
        assert not self.replies
        new_keys = [k+'t' if 'ex' in k else k for k in self.keys]
        new_queries = {k+'t' if 'ex' in k else k: {'train': self.queries[k]['train'][:-1], 'test': [{'input': self.queries[k]['train'][-1]['input']}]} for k in self.keys}
        new_replies = {k+'t' if 'ex' in k else k: [self.queries[k]['train'][-1]['output']] for k in self.keys}
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def length(self):
        return len(self.keys)

    def shuffled(self, seed=None):
        if seed is not None: np.random.seed(seed)
        return self.__class__(queries=self.queries, replies=self.replies, keys=shuffled(self.keys))

    def sorted(self, **kwargs):
        return self.__class__(queries=self.queries, replies=self.replies, keys=sorted(self.keys, **kwargs))

    def append(*datasets):
        return datasets[0].__class__(
            queries={k: v for d in datasets for k, v in d.queries.items()},
            replies={k: v for d in datasets for k, v in d.replies.items()},
            keys   =[k    for d in datasets for k    in d.keys           ],
        )

    def sort_ex_by_input_size(self, seed=42, reverse=False):
        np.random.seed(seed)
        sort_key = lambda ex: np.prod(np.shape(ex['input']))
        new_queries = {k2: {k: (sorted(np.random.permutation(np.array(v, dtype=object)), key=sort_key, reverse=reverse) if k=='train' else v) for k, v in v2.items()} for k2, v2 in self.queries.items()}
        return self.__class__(queries=new_queries, replies=self.replies, keys=[k for k in self.keys])

    def interleave(self, block_size, num_gpus=None):
        keys = np.reshape(self.keys, (-1, block_size)).T
        if num_gpus is None: return self.change_keys(keys.ravel().tolist())
        ret, num_gpus = (None, num_gpus) if isinstance(num_gpus, int) else num_gpus
        keys = np.concatenate([keys, np.full((-keys.shape[0]%num_gpus, keys.shape[1]), None)])
        keys = np.reshape(keys, (keys.shape[0]//num_gpus, num_gpus, -1)).swapaxes(0, 1).reshape(num_gpus, -1)
        new_datasets = [self.change_keys(gpu_keys.tolist()) for gpu_keys in keys]
        return new_datasets if ret is None else new_datasets[ret]

    def remove(self, *datasets):
        remove_keys = {k for d in datasets for k in d.keys}
        new_keys = [k for k in self.keys if k not in remove_keys]
        return self.change_keys(new_keys)

    def keep_key_startswith(self, key_start):
        new_keys = [k for k in self.keys if k.startswith(key_start)]
        return self.change_keys(new_keys)

    def mod_single(self, mod_func, descriptor, i, keep_key, inputs_only, esc=False):
        queries = {}
        replies = {}
        keys    = []
        for k0 in self.keys:
            desc = (('copy{i}' if mod_func is np.copy else mod_func.__name__) if descriptor is None else descriptor if isinstance(descriptor, str) else descriptor(self.queries[k0])).format(i=i)
            func = lambda a, d: np.asarray(mod_func(a) if descriptor is None else mod_func(a, d)).tolist()
            k1 = k0 if keep_key else f"{k0}.{'I' if inputs_only else ''}{desc}"
            esc_id = np.random.randint(0,10) if esc and 'perm' not in desc else -1
            # if pp:print('k1',k1,'desc',desc, 'esc_id',esc_id)
            keys.append(k1)
            queries[k1] = {m: [{t: (func(a, desc) if (t=='input' or not inputs_only) and not(m=='train' and n==esc_id) else a) for t, a in x.items()} for n,x in enumerate(e)] for m, e in self.queries[k0].items()}
            if k0 in self.replies:
                replies[k1] = [func(a, desc) for a in self.replies[k0]]
            # if 0<=esc_id<len(queries[k1]['train']) and pp:
            #     import rich
        # if pp:print(queries[k1])
            #     visualize_task(queries[k1])
        ret = self.__class__(queries=queries, replies=replies, keys=keys)
        return ret

    def mod(self, mod_func, descriptor=None, n=1, stack=None, keep=False, keep_key=False, shuffle=False, join=True, inputs_only=False, esc=False):
        assert not (keep and keep_key)
        cur = self
        ret = [cur.shuffled() if shuffle else cur] if keep else []
        if stack is None: stack = mod_func.__name__.startswith('rot')
        for i in range(n):
            cur = (cur if stack else self).mod_single(mod_func, descriptor, i=i, keep_key=keep_key, inputs_only=inputs_only, esc=esc)
            ret.append(cur.shuffled() if shuffle else cur)
        return self.__class__.append(*ret) if join else ret

    def get(self, key, formatter):
        # if pp:print('get',self.queries[key]['train'])
        assert formatter.out2_token is None or key in self.replies
        train = formatter.fmt_train(self.queries[key]['train'], key=key.split('.c')[-1].replace('i','') if '.c' in key else None)
        query = formatter.fmt_query(self.queries[key]['test'], i=len(self.queries[key]['train']), key=key.split('.c')[-1].replace('i','') if '.c' in key else None)
        reply = formatter.fmt_reply(self.replies[key], self.faulty.get(key)) if key in self.replies else ''
        text = train+query+reply if reply else formatter.fmt_train(self.queries[key]['train'], last_is_challenge=True, key=key.split('.c')[-1].replace('i','') if '.c' in key else None)
        if random.randint(0,3000)==0 and pp:print('get text',key,text)
        # elif pp:print('get train valid',key)
        return dict(key=key, train=train, query=query, reply=reply, input=train+query, text=text)

    def as_list(self, formatter):
        return [self.get(key, formatter) for key in self.keys]

    def as_dataset(self):
        from datasets import Dataset
        return Dataset.from_list([{'key': k, 'query': self.queries[k], 'reply': self.replies[k]} for k in self.keys])

    def get_length(self, key, formatter, name, max_of_transposed=False):
        if formatter is None:
            if   name=='input': return sum(np.prod(np.shape(v)) for v3 in self.queries[key].values() for v2 in v3 for v in v2.values())
            elif name=='reply': return sum(np.prod(np.shape(v)) for v in self.replies[key])
            else: assert False
        else:
            datasets = [self]
            if max_of_transposed:
                if self.transposed_dataset is None: self.transposed_dataset = self.mod(np.transpose, keep=False, keep_key=True)
                datasets.append(self.transposed_dataset)
            return max(len(formatter.tokenizer(ds.get(key, formatter=formatter)[name])['input_ids']) for ds in datasets)

    def get_lengths(self, formatter, name, max_of_transposed=False):
        return {key: self.get_length(key, formatter=formatter, name=name, max_of_transposed=max_of_transposed) for key in self.keys}

    def sorted_by_len(self, reverse=False, **kwargs):
        new_keys = [key for _, key in sorted([(v, k) for k, v in self.get_lengths(**kwargs).items()], reverse=reverse)]
        return self.change_keys(new_keys)

    def filter_by_len(self, min_len=0, max_len=float('inf'), **kwargs):
        new_keys = [k for k, v in self.get_lengths(**kwargs).items() if min_len<=v<=max_len]
        return self.change_keys(new_keys)

    import numpy as np
    import random
    
    def one_color(self, seed=0):
        rng = random.Random(seed)
        new_queries = {}
        new_replies = {}
        new_keys = []
        # new_queries.update(self.queries)
        # new_replies.update(self.replies)
        # new_keys.extend(self.keys)
    
        for n,key in enumerate(self.keys):
            if rng.choice([0,1]):
                new_queries[key] = self.queries[key]
                if key in self.replies:
                    new_replies[key] = self.replies.get(key,{})
                new_keys.append(key)
                continue
                
            train_colors = []
            filtered_trains = []
            new_key = key
                
            for example in self.queries[key]['train']:
                in_arr = np.array(example['input'])
                most_frequent = np.bincount(in_arr.flatten()).argmax()

                out_arr = np.array(example['output'])
                diff_arr = out_arr.copy()
                if in_arr.shape==out_arr.shape:
                    diff_arr[diff_arr==in_arr] = 0
                attention_colors = [v for v in list(np.unique(diff_arr)) if v not in [0,most_frequent,key.split('permute')[-1].find('0')]]
                # if pp:print('attention_colors',attention_colors)

                in_filter = False
                if attention_colors:
                    c = rng.choice(attention_colors)
                    train_colors.append(str(c))
                    in_new = in_arr.copy()
                    out_new = out_arr.copy()
                    if rng.choice([0]):
                        in_filter = True
                        train_colors.append('i')
                        in_new[in_new != c] = 0
                    out_new[out_new != c] = 0
                    filtered_trains.append({
                        'input': in_new.tolist(),
                        'output': out_new.tolist()
                    })
                else:
                    train_colors.append('_')
                    filtered_trains.append(None)  # Mark as skipped
    
            if all(t is None for t in filtered_trains):
                new_queries[key] = self.queries[key]
                if key in self.replies:
                    new_replies[key] = self.replies.get(key,{})
                new_keys.append(key)
                # if pp:print('No attention colors',key)
                # if pp:visualize_task(key)
                continue  # skip this key if no usable train
    
            # Clean color string: e.g., c1_4
            color_str = ''.join(train_colors)
            color_str = 'c' + ''.join(train_colors)
            new_key = f'{key}.{color_str}'
    
            # Process test
            test_list = self.queries[key]['test']
            reply_list = self.replies.get(key, [])

            filtered_replies = []

            for idx, test_example in enumerate(test_list):
                in_arr = np.array(test_example['input'])
                most_frequent = np.bincount(in_arr.flatten()).argmax()
    
                if idx < len(reply_list):
                    out_arr = np.array(reply_list[idx])
                    diff_arr = out_arr.copy()
                    if in_arr.shape==out_arr.shape:
                        diff_arr[diff_arr==in_arr] = 0
                    attention_colors = [v for v in list(np.unique(diff_arr)) if v not in [0,most_frequent,key.split('permute')[-1].find('0')]]
                    # if pp:print('test attention_colors',attention_colors)
                else:
                    attention_colors = []
    
                if not attention_colors:
                    continue  # skip test if no colors
                if rng.choice([0]):
                    continue
    
                c_test = rng.choice(attention_colors)
                test_color_str = f't{c_test}'
                if idx==0:
                    new_key += f'{test_color_str}'
                else:
                    new_key += f'{c_test}'
    
                # Add reply if exists
                if idx < len(reply_list):
                    out_arr = np.array(reply_list[idx])
                    out_new = out_arr.copy()
                    out_new[out_new != c_test] = 0
                    filtered_replies.append(out_new.tolist())

            # if new_key==key:
            #     continue
                
            if new_key not in new_queries:
                new_queries[new_key] = {'train': [], 'test': test_list}
                new_keys.append(new_key)
                if reply_list and filtered_replies:new_replies[new_key] = []
    
            # Add filtered train examples
            for filtered in filtered_trains:
                if filtered is not None:
                    new_queries[new_key]['train'].append(filtered)

            for filtered in filtered_replies:
                if filtered is not None:
                    new_replies[new_key].append(filtered)
            # if pp:visualize_task(new_queries[new_key],new_replies[new_key],title=f'#{n},{new_key}')

        # if pp:print('new_keys',new_keys)
        return self.__class__(
                keys=new_keys,
                queries=new_queries,
                replies=new_replies
            )


    def get_diff(self):
        import copy
        new_queries = {}
        new_replies = {}
        new_keys = []
        for k in self.keys:
            modified_key = k
            if all(np.array(v['input']).shape == np.array(v['output']).shape for v in self.queries[k]['train']):
                unique_values = set()
                for v in self.queries[k]['train']:
                    unique_values.update(np.unique(v['input']))
                    unique_values.update(np.unique(v['output']))
                for v in self.queries[k]['test']:
                    unique_values.update(np.unique(v['input']))
                remain_values = sorted(set(range(10)) - unique_values)
                if remain_values:
                    diff_marker = remain_values[-1]
                    modified_key = f'{k}.dif{diff_marker}'
                    
                    queries_copy = copy.deepcopy(self.queries[k])
                    replies_copy = copy.deepcopy(self.replies.get(k,{}))
    
                    for v in queries_copy['train']:
                        in_arr = np.array(v['input'])
                        out_arr = np.array(v['output'])
                        out_arr[out_arr == in_arr] = diff_marker
                        v['output'] = out_arr.tolist()
    
                    for i, v in enumerate(queries_copy['test']):
                        if len(replies_copy) > i:
                            in_arr = np.array(v['input'])
                            out_arr = np.array(replies_copy[i])
                            out_arr[out_arr == in_arr] = diff_marker
                            replies_copy[i] = out_arr.tolist()
                else:
                    queries_copy = self.queries[k]
                    replies_copy = self.replies.get(k,{})
            else:
                queries_copy = self.queries[k]
                replies_copy = self.replies.get(k,{})
    
            new_queries[modified_key] = queries_copy
            if replies_copy:
                new_replies[modified_key] = replies_copy
            new_keys.append(modified_key)
            
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)
                    
    def cut_to_query_count(self, max_count, from_end=False, p=1):
        import copy
        import random
    
        new_queries = {}
        new_keys = []
        new_replies = copy.deepcopy(self.replies)
    
        for k in self.keys:
            if random.random() < p:
                # Rename key if needed
                if not from_end:
                    new_key = '.'.join([
                        v[:max_count + 2] if v.startswith('ex') else v
                        for v in k.split('.')
                    ])
                else:
                    new_key = '.'.join([
                        'ex' + v[-max_count:] if v.startswith('ex') else v
                        for v in k.split('.')
                    ])
                new_keys.append(new_key)
    
                # Copy replies if present
                if k in self.replies:
                    new_replies[new_key] = copy.deepcopy(self.replies[k])
    
                # Deep copy of queries[k]
                q = copy.deepcopy(self.queries[k])
                if 'train' in q and isinstance(q['train'], list):
                    if from_end:
                        q['train'] = q['train'][-max_count:]
                    else:
                        q['train'] = q['train'][:max_count]
                new_queries[new_key] = q
            else:
                new_keys.append(k)
                new_queries[k] = self.queries[k]
    
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def cut_to_len(self, formatter, name, max_len, max_new_tokens='auto', from_end=False, quiet=False, **kwargs):
        if max_new_tokens:
            if max_new_tokens=='auto': max_new_tokens = formatter.max_new_tokens()
            max_len_old, max_len = max_len, max_len - max_new_tokens
            if not quiet: print(f'*** Reducing task size to max. {max_len_old} tokens ({max_len} input + {max_new_tokens} generated)...')
        elif not quiet: print(f'*** Reducing task size to max. {max_len} tokens...')
        temp_ds = self.change_keys(self.keys)
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in (self.keys if quiet else tqdm(self.keys, file=sys.stdout)):
            reply = temp_ds.replies.get(key)
            while max_len<temp_ds.get_length(key, formatter=formatter, name=name, **kwargs):
                query = temp_ds.queries[key]
                if not key.split('.')[-1].startswith('ex'): key = f"{key}.ex{''.join(map(str, range(len(query['train']))))}"
                key_split = key.split('.')
                assert key_split[-1].startswith('ex')
                key = '.'.join(key_split[:-1] + [f'ex{key_split[-1][2:-1] if from_end else key_split[-1][3:]}'])
                temp_ds.queries[key] = {k: ((v[:-1] if from_end else v[1:]) if k=='train' else v) for k, v in query.items()}
                if reply is not None: temp_ds.replies[key] = reply
            new_keys.append(key)
            new_queries[key] = temp_ds.queries[key]
            if reply is not None: new_replies[key] = reply

        # if pp:print('new_keys in cut_to_len',new_keys)
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)

    def shuffle_ex(self, perm=None, keep_max=None):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['train'])
            p = np.random.permutation(n) if perm is None else perm
            local_keep_max = keep_max
            if local_keep_max == 'rand':
                local_keep_max = np.random.randint(0, n + 1)  # 0 to n inclusive
            if local_keep_max is not None:
                p = p[:local_keep_max]
            new_key = f'{key}.ex' + ('-' if len(p) and (p.max() > 9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {
                k: (np.array(v, dtype=object)[p].tolist() if k == 'train' else v)
                for k, v in self.queries[key].items()
            }
            if key in self.replies:
                new_replies[new_key] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def shuffle_rp(self, keep_max=None):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['test'])
            p = np.random.permutation(n)
            if keep_max is not None: p = p[:keep_max]
            new_key = f'{key}.rp' + ('-' if (p.max()>9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {k: (np.array(v, dtype=object)[p].tolist() if k=='test' else v) for k, v in self.queries[key].items()}
            if key in self.replies: new_replies[new_key] = np.array(self.replies[key], dtype=object)[p].tolist()
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def append_to_keys(self, text):
        return self.change_keys([f'{k}{text}' for k in self.keys])

    def random_select(self, n):
        keys = np.array(self.keys).reshape(n, -1).T
        choice = np.random.randint(0, n, size=[len(keys)])
        return self.change_keys(keys[np.arange(len(keys)), choice])

    def augment(self, tp=False, rot=False, n=1, perm=None, perm_append=False, shfl_keys=False, shfl_ex=False, seed=None, quiet=False, inputs_only=False, esc=False, keep_max=None):
        if not quiet: print(f"*** Augment dataset{' (inputs only)' if inputs_only else ''}...")
        np.random.seed(seed)
        d = self
        if tp: d = d.mod(np.transpose, keep=True, inputs_only=inputs_only, esc=esc)
        if tp=='rand': d = d.random_select(n=2)
        if rot: d = d.mod(np.rot90, n=3, keep=True, inputs_only=inputs_only, esc=esc)
        if rot=='rand': d = d.random_select(n=4)
        if perm is None and n<=1: d = d.shuffled() if shfl_keys else d
        else: d = d.mod(*([np.copy] if perm is None else globals()[f"permute_{perm}"]), n=n, shuffle=shfl_keys, keep=perm_append, inputs_only=inputs_only)
        np.random.seed(seed)
        if shfl_ex: d = d.shuffle_ex(keep_max=keep_max)
        return d


    def one_rotp(self, p=1, n=4):
        import copy  # for deep copying replies safely

        new_queries = {}
        new_keys = []
        new_replies = copy.deepcopy(self.replies)  # avoid in-place mutation
    
        for key in self.keys:
            if random.random() < p:
                new_key = key + f'_rotpn{n}'
                new_queries[new_key] = {}
                if key in self.replies:new_replies[new_key] = copy.deepcopy(self.replies[key])
                for k, v in self.queries[key].items():
                    new_examples = []
                    for example in v:
                        input_arr = np.array(example['input'])
                        all_rows = []
                        for i in range(n):
                            rotated = np.rot90(input_arr, k=i % 4)  # ensure 0–3
                            all_rows.extend(rotated.tolist()+[[]])  # row-wise flatten
                        new_example = {'input': all_rows}
                        if 'output' in example:
                            new_example['output'] = example['output']
                        new_examples.append(new_example)
                    new_queries[new_key][k] = new_examples
                new_keys.append(new_key)
            else:
                new_queries[key] = self.queries[key]
                new_keys.append(key)
    
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def remove_replies(self):
        return self.__class__(queries=self.queries, replies={}, keys=[k for k in self.keys])

    def split_at_pos(self, pos=None, random_seed=None):
        keys = self.keys
        keys_split = [keys[i::4] for i in range(4)]

        return tuple(self.change_keys(new_keys, keep_flags=True) for new_keys in keys_split)

    def get_submission(self, results=None):
        assert self.is_orig==True, 'Must be run on original dataset.'
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(self.queries[k]['test']))] for k in self.keys}
        if results is not None: self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        print(f'*** Generating submission for {len(results)} outputs...')
        for k, v in results.items():
            base_id, base_nr = k.split('_')
            try:
                target_dict = submission[base_id][int(base_nr)]
                for i, g in enumerate(v[:len(target_dict)]):
                    
                    target_dict[f'attempt_{i+1}'] = g.tolist()
            except:
                pass

    def validate_submission(self, submission, queries_file):
        assert self.is_orig==True, 'Must be run on original dataset.'
        score = 0
        for k, v in self.replies.items():
            print(k)
            for i, r in enumerate(v):
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        print(score)
                        # print(r)
                        visualize_task(k,[None]*i + [r], file=queries_file)
                        # break
                    else:
                        if submission[k][i][attempt]==[[0]]:# and (attempt=='attempt_2' or i>0):
                            continue
                        print(attempt)
                        visualize_task(k,[None]*i + [submission[k][i][attempt]], file=queries_file)
                            # print(r)
                            # print(submission[k][i][attempt])
        return score
def get_class_MyDataCollator(cache=[]):
    if not cache:
        from trl import DataCollatorForCompletionOnlyLM
        class MyDataCollator(DataCollatorForCompletionOnlyLM):
            def setup(self, out2_token_id=None, fault_token_id=None, fault_freq=0, sample_tries=8, mask_first_output=False):
                self.out2_token_id = out2_token_id
                self.fault_token_id = fault_token_id
                self.fault_freq = fault_freq
                self.sample_tries = sample_tries
                self.mask_first_output = mask_first_output
                self.need_print = True
                return self

            def torch_call(self, examples):
                batch = super().torch_call(examples)
                if self.out2_token_id is not None:
                    # if pp:print('out2_token_id: batch:',batch)
                    assert not self.fault_freq
                    for i in range(len(batch['input_ids'])):
                        end_pos = ((batch['labels'][i] != -100              ).nonzero().max()).item() + 1
                        mid_pos = ((batch['labels'][i] == self.out2_token_id).nonzero().max()).item() + 1
                        beg_pos = mid_pos - (end_pos - mid_pos)
                        batch['labels'][i][beg_pos:mid_pos] = batch['labels'][i][mid_pos:end_pos]
                elif self.fault_freq:
                    # if pp:print('fault_freq',batch)
                    for i in range(len(batch['input_ids'])):
                        end_pos = ((batch['labels'][i] != -100).nonzero().max()).item() + 1
                        if not isinstance(self.fault_freq, float):
                            eos_token_id = batch['labels'][i][end_pos - 1]
                            num_examples = (batch['labels'][i] == eos_token_id).sum().item() - 1
                            fault_freq = self.fault_freq[num_examples]
                        else: fault_freq = self.fault_freq
                        if random.random() < fault_freq:
                            beg_pos = ((batch['labels'][i][:end_pos]==-100).nonzero().max()).item() + 1
                            fault_pos = random.randint(beg_pos, end_pos-2)
                            fault_tok = batch['labels'][i][fault_pos].item()
                            for t in range(self.sample_tries):
                                new_tok = batch['labels'][i][random.randint(beg_pos, end_pos-2)].item()
                                if fault_tok!=new_tok:
                                    batch['input_ids'][i][fault_pos] = new_tok
                                    batch['labels'][i][fault_pos+1:end_pos] = self.fault_token_id
                                    break
                # else:
                #     if pp:print('batch as is :',batch)
                for i in range(len(batch['labels'])):
                    for _ in range(self.mask_first_output):
                        beg_pos = ((batch['labels'][i] != -100).nonzero().min()).item()
                        mid_pos = ((batch['labels'][i][beg_pos:] == -100).nonzero().min()).item() + beg_pos
                        end_pos = ((batch['labels'][i] != -100).nonzero().max()).item() + 1
                        if mid_pos<end_pos: batch['labels'][i][beg_pos:mid_pos] = -100

                return batch
        cache.append(MyDataCollator)
    return cache[0]

class ArcFormatter(object):
    def __init__(self, inp_prefix, out_prefix, arr_sep, out2_use=False, out2_token=None, arr_beg='', arr_end='', pretext='', pre_out=None, exa_sep='', exa_end='', qry_prefix=None, rpl_prefix=None, rpl_sep=None, dec_sep=None, min_wid=0, min_pad='', pretext_corpus_split='', masking=0, tokenizer=None, collator_kwargs={}, repeat_input_aug=None, repeat_input_pre=None):
        self.tokenizer = tokenizer
        self.inp_prefix = inp_prefix
        self.out_prefix = out_prefix
        self.out2_token = out2_token
        self.out2_use = out2_use
        assert not out2_use or out2_token is not None
        assert not out2_use or masking in [1, 2]
        assert masking!=2 or out2_use or rpl_prefix is not None
        self.qry_prefix = qry_prefix if qry_prefix is not None else inp_prefix
        self.rpl_prefix = rpl_prefix if rpl_prefix is not None else out_prefix
        self.rpl_sep = rpl_sep if rpl_sep is not None else self.rpl_prefix
        self.arr_sep = arr_sep
        self.arr_beg = arr_beg
        self.arr_end = arr_end
        self.pretext = pretext
        self.pre_out = pre_out
        self.pre_out_empty = ['']*99
        self.pretext_corpus_split = pretext_corpus_split
        self.exa_sep = exa_sep
        self.exa_end = exa_end
        self.dec_sep = arr_sep if dec_sep is None else dec_sep
        self.min_wid = min_wid
        self.min_pad = min_pad
        self.masking = masking
        self.collator_kwargs = collator_kwargs
        self.repeat_input_aug = repeat_input_aug
        self.repeat_input_pre = repeat_input_pre

    def fmt_array(self, array):
        return self.arr_beg + self.arr_sep.join(str(row).replace(' ', '').replace(',', '').replace('[', '').replace(']', '')+self.min_pad*max(0, self.min_wid-len(row)) for row in array) + self.arr_end

    def get_pre_out(self, pretext_split):
        if self.pre_out is None: return self.pre_out_empty
        if pretext_split: return [self.pretext_corpus_split.join(list(p) + ['']) for p in self.pre_out]
        return self.pre_out

    def fmt_train(self, train, last_is_challenge=False, pretext_split=False, key=None):
        po = self.get_pre_out(pretext_split=pretext_split)
        ex = [(f"{self.fmt_query([x], i, pretext_split=pretext_split, key=key)}{self.fmt_reply([x['output']])}" if last_is_challenge and i+1==len(train) else
               f"{self.inp_prefix}{self.fmt_array(x['input'])}{self.repeat_input(x, no_aug=pretext_split)}{po[i]}{key[i] if key and key[i]!='_' else ''}{self.out_prefix}{self.fmt_array(x['output'])}") for i, x in enumerate(train)]
        pre = self.pretext_corpus_split.join(list(self.pretext)+['']) if pretext_split else self.pretext
        end = '' if last_is_challenge else (self.exa_end + self.tokenizer.eos_token)
        return pre + (self.exa_end + self.tokenizer.eos_token + self.exa_sep).join(ex) + end

    def fmt_query(self, query, i, pretext_split=False, key=None):
        po = self.get_pre_out(pretext_split=pretext_split)
        return ''.join(f"{self.qry_prefix}{self.fmt_array(x['input'])}{self.repeat_input(x, no_aug=pretext_split)}{po[i]}{key.split('.')[0][-1] if key and 't_.' not in key else ''}{self.rpl_prefix}" for x in query[:1])

    # def fmt_query(self, query, i, pretext_split=False, key=None):
    #     po = self.get_pre_out(pretext_split=pretext_split)
    #     return ''.join(f"{self.qry_prefix}{self.inp_prefix.join([self.fmt_array(x['input'][b*len(x['input'])//4:b+len(x['input'])//4]) for b in range(4)])}{self.repeat_input(x, no_aug=pretext_split)}{po[i]}{key.split('.')[0][-1] if key and 't_.' not in key else ''}{self.rpl_prefix}" for x in query[:1])

    def repeat_input(self, x, no_aug=False):
        if self.repeat_input_aug is None: return ''
        return f"{self.repeat_input_pre}{self.fmt_array(((lambda x: x) if no_aug else self.repeat_input_aug)(x['input']))}"

    def fmt_reply(self, reply, fault=None):
        ids = self.fmt_array(reply[0]) + self.exa_end + self.tokenizer.eos_token
        if self.out2_use:
            if fault is None: fault = reply
            ids = self.fmt_array(fault[0]) + self.exa_end + self.out2_token + ids

        return ids

    def quick_test(self, decoded, done):
        sp = decoded.split(self.tokenizer.eos_token)[0].split(self.dec_sep)
        sl = len(sp[0])
        is_prefix = sl>0 and len(sp[-1])<=sl and (len(sp)==1 or len(sp[-2])==sl) and all(x.isdigit() for x in sp[-1])
        return is_prefix and (not done or len(sp[-1])==0 or len(sp[-1])==sl)

    @staticmethod
    def is_valid_solution(guess):
        return isinstance(guess, np.ndarray) and guess.ndim == 2 and all(0 < x <= 30 for x in guess.shape)

    def max_new_tokens(self, safety_margin=1):
        max_sized_reply = np.zeros([30, 30], dtype=int)
        max_sized_reply[:,::2] = 1
        tokenized = self.tokenizer(self.fmt_reply([max_sized_reply]))['input_ids']
        max_new_tokens = len(tokenized)
        if tokenized[0]==self.tokenizer.bos_token_id: max_new_tokens -= 1
        return max_new_tokens + safety_margin

    def de_tokenize(self, tokens, scores=None, classifier=None):
        import torch, copy
        if classifier is not None:
            de_tokenized = copy.deepcopy(tokens)
            tokens_cut = tokens = np.array(tokens).ravel()
            # if pp:print('tokens_cut',tokens_cut,'de_tokenized',de_tokenized, 'scores',scores)
        else:
            tokens_cut = cut_at_token(tokens, self.tokenizer.eos_token_id)
            de_tokenized = self.tokenizer.batch_decode([tokens_cut])[0]
        # if pp:print('detokenized',de_tokenized)
        score_val = None
        if scores is not None and len(tokens_cut)>0:
            tokens_with_eos = tokens[:len(tokens_cut)+1]
            score_val = torch.nn.functional.log_softmax(torch.tensor(scores), dim=-1).numpy().copy()[np.arange(len(tokens_with_eos)), tokens_with_eos].sum()
            # if pp:print(f'score_val{score_val:.1f} for len',len(de_tokenized),'detokenized ',de_tokenized,'\n=sum of len',len(tokens_with_eos),torch.nn.functional.log_softmax(torch.tensor(scores), dim=-1).numpy().copy().round(1)[np.arange(len(tokens_with_eos)), tokens_with_eos])
            # if pp:print(f'In detokenized, score_val{score_val:.1f} for len',len(de_tokenized),'detokenized ',de_tokenized)
            if classifier is not None:
                number_token_ids = list(range(11))
            else:
                number_token_ids = [self.tokenizer.vocab[k] for k in map(str, range(10))]
            fault_token_id = self.collator_kwargs.get('fault_token_id')
            if fault_token_id is not None: number_token_ids.append(fault_token_id)
            number_token_ids = np.array(number_token_ids)
            number_positions = (tokens_cut[..., np.newaxis] == number_token_ids).any(-1)
            scores = scores[:len(tokens_cut), number_token_ids][number_positions]
            scores = torch.nn.functional.log_softmax(torch.tensor(scores), dim=-1)[:, :10 if classifier is None else 11].numpy().copy()
        return max(len(tokens)+1, len(tokens_cut)), score_val, de_tokenized, scores

    def decode_to_array_single(self, text, score=None, limit_rows=30):
        # if pp:print('text',text)
        try:
            import traceback
            if isinstance(text,list):
                by_rows = text
                limited = False
            else:
                by_rows = [row for row in [[int(x) for x in line if x.isdigit()] for line in text.split(self.dec_sep)] if len(row)]
                if limit_rows and len(by_rows) > limit_rows:
                    by_rows = by_rows[:limit_rows]
                    limited = True
                else: limited = False
            decoded = np.array(by_rows, dtype=int)
            if self.is_valid_solution(decoded):
                try:
                    assert score is not None
                    decoded_flat = decoded.ravel()
                    if limited: score = score[:len(decoded_flat)]
                    score_all = score.reshape(decoded.shape + score.shape[1:])
                    # if pp:print('decoded',decoded.shape,'decoded_flat',decoded_flat.shape,'score',score.shape,'score_all',score_all.shape)
                    score_result = score[range(len(decoded_flat)), decoded_flat]
                    score_reshaped = score_result.reshape(decoded.shape)
                    score_cum_reshaped = score_result.cumsum().reshape(score_reshaped.shape)
                    score_all_cum = score_cum_reshaped[..., np.newaxis] - score_reshaped[..., np.newaxis] + score_all
                except: 
                    if pp:print(traceback.format_exc())
                    score_reshaped = score_cum_reshaped = np.full(decoded.shape, -float('inf'))
                # if pp:print('output',decoded,'score',score_reshaped.shape,'score_cum',score_cum_reshaped.shape,'score_all',score_all.shape,'score_all_cum',score_all_cum.shape)
                return {'output': decoded, 'score': score_reshaped, 'score_cum': score_cum_reshaped, 'score_all': score_all, 'score_all_cum': score_all_cum}
        except: 
            pass
            # if pp:print('D not is_valid_solution',text,traceback.format_exc())
        if pp:print('D not is_valid_solution',text,traceback.format_exc())
        return {}

    def decode_to_array(self, text, score=None, limit_rows=30):
        if not self.out2_use: text, score = [text], [score]
        else:
            text = text.split(self.out2_token)
            if score is None: score = [None]*len(text)
            else:
                lengths = np.cumsum([len(list(filter(str.isdigit, t))) for t in text])
                score = [score[s:e] for s, e in zip([0]+lengths[:-1].tolist(), lengths)]
        return [self.decode_to_array_single(t, s) for t, s in zip(text, score)]

    def get_corpus(self):
        try:
            old_min_wid, self.min_wid = self.min_wid, min(self.min_wid, 2)
            return self.fmt_train([{'input': [[i] for i in range(10)], 'output': [[i] for i in range(10)]}]*3, last_is_challenge=True, pretext_split=True)
        finally: self.min_wid = old_min_wid

    def get_data_collator(self):
        if not self.masking: return None
        from transformers import DataCollatorForLanguageModeling
        collator_params = dict(tokenizer=self.tokenizer, mlm=False)
        pass_out2_token = self.tokenizer.vocab[self.out2_token] if self.out2_use and self.masking==1 else None
        # if pp:print('pass_out2_token',pass_out2_token)
        # if pp:print('vacab',self.tokenizer.vocab)
        if self.masking:
            assert not self.collator_kwargs.get('mask_first_output') or self.masking==1
            data_collator = get_class_MyDataCollator()(
                **collator_params,
                instruction_template=[self.inp_prefix, self.tokenizer.bos_token][self.masking - 1],
                response_template=[self.out_prefix, (self.out2_token if self.out2_use else self.rpl_sep)][self.masking - 1],
            ).setup(out2_token_id=pass_out2_token, **self.collator_kwargs)
        else:
            assert not self.collator_kwargs, 'only supported with masking on'
            data_collator = DataCollatorForLanguageModeling(**collator_params)
        return data_collator

    def get_output_token_ids(self):
        assert not self.out2_use
        num_tokens = [self.tokenizer.vocab[str(i)] for i in range(10)]
        sep_tokens = [tok for txt in [self.arr_beg, self.arr_sep, self.arr_end, self.exa_sep] if txt for tok in self.tokenizer(txt)['input_ids'][1:]]
        sep_tokens.append(self.tokenizer.eos_token_id)
        return num_tokens + sorted(set(sep_tokens))

ArcFormatter_pretext2 = lambda **kwargs: ArcFormatter(masking=1, inp_prefix='I', out_prefix='O', arr_sep='\n', arr_end='\n', pretext='ABCDEFGHJKLMNPQRSTUVWXYZ', pretext_corpus_split='\n', **kwargs)
ArcFormatter_pretext3 = lambda **kwargs: ArcFormatter(masking=1, inp_prefix='I', out_prefix='O', arr_sep='\n', arr_end='\n', pretext='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz', pretext_corpus_split='\n', **kwargs)
ArcFormatter_premix_2 = lambda **kwargs: ArcFormatter(masking=1, inp_prefix='I', out_prefix='O', arr_sep='\n', arr_end='\n', pretext='ABCDEFGHJKLMNPQRSTUVWXYZ', pre_out=['+/-=']*99, pretext_corpus_split='\n', **kwargs)
ArcFormatter_premix_3 = lambda **kwargs: ArcFormatter(masking=1, inp_prefix='I', out_prefix='O', arr_sep='\n', arr_end='\n', pretext='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz', pre_out=['+/-=']*99, pretext_corpus_split='\n', **kwargs)

available_formatters = dict(
    ArcFormatter_pretext2=ArcFormatter_pretext2,
    ArcFormatter_pretext3=ArcFormatter_pretext3,
    ArcFormatter_premix_2=ArcFormatter_premix_2,
    ArcFormatter_premix_3=ArcFormatter_premix_3,
)
