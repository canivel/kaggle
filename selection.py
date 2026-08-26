import numpy as np
from pp import *
from visualize import *

def hashable(guess):
    return tuple(map(tuple, guess))

def make_unique(guess_list, indices=None):
    used = set()
    out = []
    out_ind = []
    for i, g in enumerate(guess_list):
        h = hashable(g)
        if h not in used:
            used.add(h)
            out.append(np.array(g))
            if indices is not None: out_ind.append(indices[i])
    return out if indices is None else (out, out_ind)

def first_only(guesses):
    return [g['output'] for g in guesses.values()][:1]

def keep_order(guesses):
    return [g['output'] for g in guesses.values()]

def keep_order_unique(guesses):
    return make_unique(keep_order(guesses))

def get_best_shape_by_score(guess_list, getter, once_per_result=True):
    seen_outputs = set()
    shape_scores = {}
    for i, g in enumerate(guess_list):
        shape = tuple(g['output'].shape)
        scores = shape_scores[shape] = shape_scores.get(shape, [[], []])
        scores[1].append(i)
        h = hashable(g['output'])
        if h in seen_outputs: continue
        if once_per_result: seen_outputs.add(h)
        scores[0].append(g)
    shape_scores = [(getter(scores), shape, indices) for shape, (scores, indices) in shape_scores.items()]
    shape_scores = sorted(shape_scores, key=(lambda x: x[0]), reverse=True)
    if pp:print('shape_scores',shape_scores)
    return shape_scores[0]

def score_sum(guesses, getter, shape_getter=None, prefer_common_shape=True, ppp=False):
    if pp and ppp:print('\nguesses in score_sum, len:',len(guesses), guesses.keys())
    if shape_getter is None: shape_getter = getter
    guess_list = list(guesses.values())
    common_shape_indices = set(get_best_shape_by_score(guess_list, shape_getter)[2]) if prefer_common_shape else []
    scores = {}
    for i, g in enumerate(guess_list):
        h = hashable(g['output'])
        x = scores[h] = scores.get(h, [i in common_shape_indices, [], g['output']])
        x[1].append(g)
        if pp and ppp:print(sum(np.exp(v['score_val']) for v in x[1]),[v['score_val'] for v in x[1]],[np.exp(v['score_val']) for v in x[1]])
    scores = [(cs, getter(sc), o) for cs, sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[:2]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    if pp and ppp and guesses:
        ordered_scores = [(x[0],round(x[1],3)) for x in scores]
        print('ordered_scores in score_sum',ordered_scores)
        visualize_task(ordered_outputs,score=ordered_scores,key=list(guesses.keys())[0].split('.')[0])
    return ordered_outputs

getter_all_probsum = lambda guesses: sum(np.exp(g['score_val']) for g in guesses)
def score_all_probsum(guesses): return score_sum(guesses, getter_all_probsum, ppp=True)

def getter_full_probmul(p):
    def _getter(guesses, baseline=p):
        inf_score = sum([g['score_val']+baseline for g in guesses])
        # if pp:print('inf_score is sum of [g[score_val]+baseline for g in guesses]',[g['score_val']+baseline for g in guesses])
        try:
            aug_score = np.mean([sum(s+baseline for s in g['score_multi_nl']) for g in guesses])
            # if pp:print('aug_score is mean of [sum(s+baseline for s in g[score_multi_nl]) for g in guesses]',[sum(s+baseline for s in g['score_multi_nl']) for g in guesses])
            # if pp:print('inf_score',inf_score,'aug_score',aug_score)
        except:
            if pp:print('aug_score fail')
            aug_score = -1000
        return inf_score + aug_score
    return _getter

def score_full_probmul_3(guesses): return score_sum(guesses, getter_full_probmul(3), prefer_common_shape=False, ppp=use_aug)

selection_algorithms = [
    # first_only,
    # keep_order,
    # keep_order_unique,
    # score_all_probsum,
    score_full_probmul_3,
]
