# VOI (Value of Information) Early Stopping for AIMO3

**Date**: 2026-04-01
**Source**: `jonathanchan/aimo3-gpt-oss-120b-with-bayesian` (pulled via `kaggle kernels pull`)
**Purpose**: Extract and document the exact VOI formula so we can implement it

---

## Executive Summary

Jonathan Chan's notebook replaces the `early_stop=4` majority-vote trigger with a
Bayesian posterior + Value of Information stopping rule. The idea: after each attempt
completes, compute a normalized probability distribution over candidate answers, then
stop if the expected gain from running more attempts is less than the compute cost.

The mechanism has three key components that work together:
1. **Bayesian posterior** -- weights each answer by entropy, reliability, and tool use
2. **VOI decision rule** -- compares submit utility vs continue utility
3. **Hard minimum** -- never stop before 3 attempts regardless of posterior

---

## Part 1: The Bayesian Posterior

### What it computes

A normalized probability distribution over candidate answers, where each attempt is
weighted by three factors multiplied together:

```python
def _compute_bayesian_posterior(self, detailed_results):
    posterior = defaultdict(float)

    for r in detailed_results:
        answer = r['Answer']
        entropy = r['Entropy']           # mean token entropy across the chain
        python_errors = r['Python Errors']
        python_calls = r['Python Calls']

        if answer is None:
            continue

        # Factor 1: inverse-entropy weight (low entropy = confident = higher weight)
        entropy_weight = 1.0 / (1.0 + entropy)

        # Factor 2: reliability penalty for Python errors
        # error_penalty = 1.0, so 1 error halves the weight
        reliability = 1.0 / (1 + self.cfg.error_penalty * python_errors)

        # Factor 3: tool-use bonus/penalty
        if python_calls > 0 and python_errors == 0:
            tool_bonus = 1.2   # rewarded for clean code execution
        elif python_errors > 0:
            tool_bonus = 0.8   # penalized for broken code
        else:
            tool_bonus = 1.0   # no code: neutral

        weight = entropy_weight * reliability * tool_bonus
        posterior[answer] += weight

    # Normalize to sum to 1 (this makes it a proper probability distribution)
    total = sum(posterior.values())
    if total == 0:
        return {}
    for k in posterior:
        posterior[k] /= total

    return posterior
```

### Key formula breakdown

```
weight_i = (1 / (1 + H_i)) * (1 / (1 + errors_i)) * tool_bonus_i

where:
  H_i       = mean token entropy of attempt i (bits, computed from top-5 logprobs)
  errors_i  = number of Python errors in attempt i
  tool_bonus = 1.2 if clean tool use, 0.8 if errored tool use, 1.0 if no tool

posterior[answer_i] += weight_i
posterior = normalized so sum = 1
```

### Difference from our `_select_answer`

Our current v14 notebook uses `w = 1 / max(entropy, 1e-9)` (pure inverse-entropy).
Jonathan Chan uses `w = 1 / (1 + entropy)` which:
- Is always finite (never explodes to infinity like `1/0`)
- Is bounded between 0 and 1 (softened weight, less aggressive)
- Adds multiplicative error and tool-use factors on top

---

## Part 2: The VOI Stopping Rule

### The full decision tree (from `solve_problem`)

```python
for future in as_completed(futures):
    result = future.result()
    detailed_results.append(result)

    if result['Answer'] is not None:
        valid_answers.append(result['Answer'])

    posterior = self._compute_bayesian_posterior(detailed_results)

    # Gate: must have at least min_attempts_before_stop = 3 results
    if posterior and len(detailed_results) >= self.cfg.min_attempts_before_stop:

        max_prob = max(posterior.values())
        entropy = self._posterior_entropy(posterior)

        # VOI decision
        submit_utility     = max_prob
        expected_improvement = entropy * self.cfg.voi_entropy_weight  # 0.6
        continue_utility   = max_prob + expected_improvement - self.cfg.voi_compute_cost  # 0.04

        if submit_utility >= continue_utility:
            stop_event.set()
            for f in futures:
                f.cancel()
            break
```

### The posterior entropy function

```python
def _posterior_entropy(self, posterior):
    return -sum(p * math.log(p + 1e-9) for p in posterior.values())
```

This computes the Shannon entropy of the posterior distribution in **nats** (natural log,
not log base 2). A uniform distribution over N candidates has entropy = ln(N). A distribution
concentrated on one answer has entropy = 0.

### The VOI formula spelled out

```
submit_utility   = max_prob
                 = max over all answers of: P(answer_a | attempts_so_far)

continue_utility = max_prob + expected_improvement - cost
                 = max_prob
                   + (posterior_entropy * 0.6)   # voi_entropy_weight
                   - 0.04                         # voi_compute_cost

STOP if: submit_utility >= continue_utility
       = max_prob >= max_prob + entropy*0.6 - 0.04
       = 0 >= entropy * 0.6 - 0.04
       = entropy * 0.6 <= 0.04
       = entropy <= 0.04 / 0.6
       = entropy <= 0.0667 nats
```

### The simplified stopping condition

**After algebraic simplification:** stop when `posterior_entropy <= 0.0667 nats`.

Note that `max_prob` cancels from both sides of the inequality. This means the VOI rule
depends ONLY on the spread of the posterior distribution -- not on the absolute confidence
in the top answer. A posterior of {A: 0.51, B: 0.49} and {A: 0.99, B: 0.01} are treated
identically ("do not stop") as long as entropy exceeds 0.0667. Whether this was intentional
or an oversight is unknown, but it means a highly confident 51%-49% split will NOT trigger
early stopping.

The two parameters `voi_entropy_weight=0.6` and `voi_compute_cost=0.04` collapse to a
single effective threshold `entropy_threshold = 0.04 / 0.6 = 0.0667 nats`. For tuning,
only one knob is needed; changing both in proportion changes nothing.

This is equivalent to the posterior being VERY concentrated. For comparison:
- 2 equally likely answers: entropy = ln(2) = 0.693 nats (do NOT stop)
- 1 answer at 90%, 1 at 10%: entropy = -0.9*ln(0.9) - 0.1*ln(0.1) = 0.325 nats (do NOT stop)
- 1 answer at 99%, 1 at 1%: entropy = -0.99*ln(0.99) - 0.01*ln(0.01) = 0.056 nats (STOP)
- 1 answer at 100% (all agree): entropy = 0 (STOP)

**In plain English:** stop only when one answer has overwhelmingly dominated the posterior,
meaning all (or nearly all) attempts agree on the same answer AND those attempts have low
token entropy (high confidence).

### How this compares to `early_stop=4`

| Condition | early_stop=4 | VOI |
|-----------|-------------|-----|
| Trigger | 4 identical answers in valid list | posterior entropy <= 0.0667 |
| Weights | Equal weight per answer | entropy + reliability + tool_bonus |
| Minimum attempts | None (can stop at 4th) | 3 attempts before checking |
| Can stop early on attempt 3? | Never | Yes, if strong consensus |
| Can stop early on attempt 5? | Yes | Yes, same condition |
| Can stop late (>4 attempts)? | Never (always stops at 4) | Yes, if answers keep disagreeing |

The VOI approach can stop EARLIER (with 3 attempts if all 3 agree with high confidence)
or LATER (if attempts keep producing different answers, use all 8) than early_stop=4.

---

## Part 3: The CFG Parameters

```python
posterior_stop_threshold = 0.82    # NOT USED in code -- dead parameter
voi_entropy_weight = 0.6           # multiplier on entropy in VOI formula
voi_compute_cost = 0.04            # fixed cost per additional attempt
error_penalty = 1.0                # multiplier on errors in reliability formula
tool_bonus_weight = 0.2            # NOT USED directly -- baked into tool_bonus values
min_attempts_before_stop = 3       # minimum attempts before VOI is evaluated
```

Note: `posterior_stop_threshold = 0.82` is defined but NOT referenced anywhere in the
actual VOI stopping code. The stopping is purely entropy-based, not threshold-based.
The threshold may have been an earlier design that was replaced by the VOI formula.

---

## Part 4: Integration with Phase Splitting

Our current architecture (v13/v14) does NOT have explicit phases -- it runs 8 identical
attempts concurrently. But if we add phases (phase 1: 4 quick, phase 2: 4 deep), VOI
interacts cleanly:

### Phase 1 (4 quick attempts, temperature=0.5, ctx=32K)

```python
# After each quick attempt completes:
posterior = _compute_bayesian_posterior(phase1_results)
if len(phase1_results) >= 3 and posterior:
    entropy = _posterior_entropy(posterior)
    if entropy * 0.6 <= 0.04:  # posterior_entropy <= 0.0667
        # All 3 quick attempts agree with high confidence -> stop NOW
        # Saved time: skipped 1 quick + 4 deep attempts
        return select_answer(phase1_results)

# If phase 1 ends without VOI stop -> proceed to phase 2
```

### Phase 2 (4 deep attempts, temperature=1.0, ctx=65K)

```python
# Combine phase1 + phase2 results in posterior
all_results = phase1_results + phase2_results
posterior = _compute_bayesian_posterior(all_results)
# VOI stopping checks: min_attempts = 3, already satisfied
```

### Expected time savings vs early_stop=4

On easy problems where 3-4 attempts rapidly agree:
- `early_stop=4`: stops after 4th identical answer (may wait for all 4 to start)
- VOI: stops after 3rd if entropy is very low (saves 1 quick attempt ~15-30s)

On hard/ambiguous problems where attempts diverge:
- `early_stop=4`: NEVER stops early (waits for 4 identical -- may exhaust budget)
- VOI: stops after 8 attempts if entropy persistently high (same time cost)
- VOI: explicitly handles the case where early_stop=4 CANNOT stop

Key insight: `early_stop=4` can HANG on problems where attempts always disagree (common
on hard problems). VOI will always stop after all 8 attempts, explicitly acknowledging
the uncertainty.

---

## Part 5: Implementation Code

Here is a clean implementation that can be dropped into our current framework:

```python
# CFG additions (add to class CFG):
voi_entropy_weight = 0.6      # weight on posterior entropy in VOI formula
voi_compute_cost = 0.04       # fixed cost per additional attempt
error_penalty = 1.0           # error weight in reliability formula
min_attempts_before_stop = 3  # min completed attempts before VOI check

def _compute_bayesian_posterior(self, detailed_results):
    """Compute normalized posterior over candidate answers.
    
    Weight = (1 / (1 + entropy)) * (1 / (1 + errors)) * tool_bonus
    where tool_bonus = 1.2 (clean tool use), 0.8 (errored), 1.0 (no tool)
    """
    posterior = defaultdict(float)
    for r in detailed_results:
        answer = r['Answer']
        if answer is None:
            continue
        entropy = r['Entropy']
        entropy_weight = 1.0 / (1.0 + entropy)
        reliability = 1.0 / (1 + self.cfg.error_penalty * r['Python Errors'])
        pc = r['Python Calls']
        pe = r['Python Errors']
        tool_bonus = 1.2 if (pc > 0 and pe == 0) else (0.8 if pe > 0 else 1.0)
        posterior[answer] += entropy_weight * reliability * tool_bonus
    total = sum(posterior.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in posterior.items()}

def _posterior_entropy(self, posterior):
    """Shannon entropy of posterior in NATS (uses math.log = natural log).
    
    Note: this is a DIFFERENT entropy from _compute_mean_entropy, which measures
    per-token uncertainty in BITS (uses math.log2). They measure different things:
    - _compute_mean_entropy: how uncertain is the model at each token? (per-attempt signal)
    - _posterior_entropy: how spread are the candidate answers? (VOI stopping signal)
    Do not compare the two values or use one threshold for both.
    """
    return -sum(p * math.log(p + 1e-9) for p in posterior.values())

def _should_stop_voi(self, detailed_results):
    """Return True if VOI says to stop gathering more attempts.
    
    The stopping condition simplifies to:
        posterior_entropy <= voi_compute_cost / voi_entropy_weight
        posterior_entropy <= 0.04 / 0.6 = 0.0667 nats
    
    But we keep the full formula for clarity and future parameter tuning.
    """
    if len(detailed_results) < self.cfg.min_attempts_before_stop:
        return False
    posterior = self._compute_bayesian_posterior(detailed_results)
    if not posterior:
        return False
    max_prob = max(posterior.values())
    entropy = self._posterior_entropy(posterior)
    submit_utility = max_prob
    continue_utility = max_prob + entropy * self.cfg.voi_entropy_weight - self.cfg.voi_compute_cost
    return submit_utility >= continue_utility

# In solve_problem, replace the early_stop block:
# OLD:
#   c = Counter(valid).most_common(1)
#   if c and c[0][1] >= self.cfg.early_stop:
#       stop.set(); break
#
# NEW:
#   if self._should_stop_voi(detailed):
#       stop.set(); break
```

---

## Part 6: Expected Impact vs early_stop=4

### On easy problems (3-4 attempts agree quickly)

Both approaches stop at similar times. VOI may stop 1 attempt earlier if entropy is
already very low after 3 attempts. Marginal time savings.

### On medium problems (4-6 attempts agree after divergence)

`early_stop=4` may stop after 4th identical answer even if total is 4/8 valid attempts.
VOI requires the entropy to be very low (nearly all weighted mass on one answer), so it
may require 5-6 attempts. This is SAFER -- you don't stop on a 4/4 streak that happens
early while the 5th would have contradicted.

### On hard problems (answers keep diverging, never 4 in agreement)

`early_stop=4`: never triggers, all 8 attempts run, highest-count answer selected.
VOI: never triggers (entropy stays high), all 8 attempts run, weighted posterior selected.
Same behavior in terms of stopping, but VOI uses a better voting rule for the final answer.

### On compute budget

Both approaches run up to 8 attempts. Neither wastes significantly more time than the
other. The primary win for VOI is the better ANSWER SELECTION (weighted posterior vs
plain majority vote), not the early stopping itself.

---

## Part 7: The ACTUAL Difference vs Our Code

Comparing Jonathan Chan's notebook to our `submission_v6_huikang.ipynb` (which is the
same base as v13/v14), the differences are:

1. **Model**: Jonathan Chan uses `huikang/gpt-oss-120b-aimo3/transformers/160a/14`
   (fine-tuned). We use base `danielhanchen/gpt-oss-120b`.

2. **Temperature**: Jonathan Chan uses `temperature=1.0`. Our v13/v14 uses `temperature=0.5`.
   (v6_huikang also uses `temperature=0.5`.)

3. **Stopping**: Jonathan Chan uses VOI. We use `early_stop=4` majority vote.

4. **Answer selection**: Jonathan Chan uses `_compute_bayesian_posterior` (entropy *
   reliability * tool_bonus). Our v14 uses `1 / max(entropy, 1e-9)` (entropy only, no
   reliability or tool bonus).

5. **ANSWER_ONLY_PROMPT**: Jonathan Chan has this but it is defined in CFG and not used
   in the visible inference code. Likely legacy.

The three biggest opportunities from this notebook for us are:
- Replace `early_stop=4` with the VOI rule (low implementation risk)
- Add reliability * tool_bonus factors to the posterior weight (low risk)
- Use the fine-tuned model (medium risk -- need to confirm it beats base on our setup)

---

## Sources

- Jonathan Chan notebook: `jonathanchan/aimo3-gpt-oss-120b-with-bayesian`
  Pulled via: `kaggle kernels pull jonathanchan/aimo3-gpt-oss-120b-with-bayesian`
- Our existing research: `/f/kaggle/aimo-progress-prize-3/research/huikang_model_research.md`
- Our existing research: `/f/kaggle/aimo-progress-prize-3/research/advanced_voting_mechanisms.md`
