# Multi-Turn Follow-Up Prompting: Exact Implementation

**Date**: 2026-04-01  
**Source**: `submission_v10_followup.ipynb` (our own built notebook)  
**Status**: Implementation confirmed, ready to integrate

---

## The Problem

Some attempts in `_process_attempt` exhaust the main turn loop (or exit via `channel == 'final'` or token
exhaustion) without ever producing a `\boxed{}` answer. These attempts return `Answer: None` and are
excluded from voting entirely. Injecting a follow-up USER message recovers a committed answer in many
of these cases.

---

## How the Harmony Protocol Handles Injection

The Harmony protocol uses `Conversation.from_messages(messages)` to hold a list of `Message` objects.
The conversation is mutable — you append to `conversation.messages` directly. This is already how tool
responses are added in the main loop:

```python
conversation.messages.extend(tool_resp)  # tool response injection
```

To inject a follow-up USER turn, you use the same `Message.from_role_and_content` constructor that
`AIMO3Template.apply_chat_template` already uses for the initial user message:

```python
followup_msg = Message.from_role_and_content(Role.USER, FOLLOWUP_PROMPT)
conversation.messages.append(followup_msg)
```

That is the complete injection mechanism. No special API. The conversation renders correctly via:
```python
prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
```

---

## The Follow-Up Prompt Text

The v10 notebook uses:

```python
FOLLOWUP_PROMPT = (
    'You have been working on this problem. Based on your analysis so far, '
    'what is the final integer answer? The answer must be between 0 and 99999. '
    'Please state your answer inside \\boxed{}.'
)
```

This is the correct text. It:
1. Acknowledges the model's prior work (avoids confusion about "new" question)
2. States the constraint (0-99999 integer)
3. Requires `\boxed{}` explicitly
4. Is concise — the model's context already has its reasoning

---

## Temperature for the Follow-Up

The follow-up uses `temperature=0.0` (greedy). This is correct because:
- The follow-up is committing to an answer from existing reasoning, not generating new reasoning
- Greedy maximizes the probability of the most confident answer given the context
- The original temperature was for diversity across attempts; the follow-up is a single deterministic
  extraction, not exploration

The main stream omits `logprobs` (not needed for the follow-up).

---

## Token Budget for the Follow-Up

The follow-up limits `max_tokens` to `min(max_tokens, 512)`. The model should respond with
just a short acknowledgment + `\boxed{N}`. 512 tokens is generous. Using `min()` ensures we
don't request more than the context window allows.

---

## How Many Follow-Up Turns

**One.** The v10 implementation does a single follow-up turn. If the model can't commit to
`\boxed{}` after one nudge with its full reasoning in context, a second nudge adds latency
for negligible gain. The turn loop already ran up to `cfg.turns = 128` — the model is done.

---

## Exact Code to Insert into `_process_attempt`

Add immediately after the main `for turn_idx in range(self.cfg.turns):` loop closes and
before the `except Exception` clause. The insertion point is where `final_answer` is still
`None` and the main loop has exited cleanly.

### Step 1: Define FOLLOWUP_PROMPT (module-level constant, before AIMO3Solver class)

```python
FOLLOWUP_PROMPT = (
    'You have been working on this problem. Based on your analysis so far, '
    'what is the final integer answer? The answer must be between 0 and 99999. '
    'Please state your answer inside \\boxed{}.'
)
```

### Step 2: Replace `_` with `turn_idx` in the for-loop header (for clarity only, not required)

```python
for turn_idx in range(self.cfg.turns):
```

### Step 3: Insert the follow-up block after the for-loop, before `except Exception`

```python
            # === MULTI-TURN FOLLOW-UP ===
            # If no answer found after main loop, inject a follow-up asking for the answer
            if final_answer is None and not stop_event.is_set() and time.time() < deadline:
                followup_msg = Message.from_role_and_content(Role.USER, FOLLOWUP_PROMPT)
                conversation.messages.append(followup_msg)
                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
                if max_tokens >= self.cfg.buffer_tokens:
                    stream = self.client.completions.create(
                        model=self.cfg.served_model_name, temperature=0.0,  # Greedy for follow-up
                        max_tokens=min(max_tokens, 512),  # Short response expected
                        prompt=prompt_ids, seed=attempt_seed, stream=True,
                        extra_body={'stop_token_ids': self.stop_token_ids, 'return_token_ids': True}
                    )
                    try:
                        text_chunks = []
                        for chunk in stream:
                            if stop_event.is_set() or time.time() > deadline: break
                            nx = chunk.choices[0].text
                            if nx: text_chunks.append(nx)
                            if '}' in (nx or ''):
                                st = ''.join(text_chunks[-16:])
                                a = self._scan_for_answer(st)
                                if a is not None: final_answer = a; break
                    finally: stream.close()
```

---

## Where in the Method Structure

The full structure of `_process_attempt` after the change:

```
try:
    sandbox = ...
    local_tool = ...
    conversation = ...

    for turn_idx in range(self.cfg.turns):   # main turn loop
        ...streaming...
        if final_answer is not None: break
        if last.channel == 'final': break
        if last.recipient == 'python': ...tool call...

    # === FOLLOW-UP (new) ===
    if final_answer is None and not stop_event.is_set() and time.time() < deadline:
        ...inject USER message + one greedy generation...

except Exception:
    python_errors += 1
finally:
    sandbox.reset(); sandbox_pool.put(sandbox)

return {..., 'Answer': final_answer}
```

The follow-up is inside the `try` block but outside the `for` loop. This ensures it shares
the same `conversation` object and is cleaned up properly by the `finally` block.

---

## Does the Model Produce Answers Without `\boxed{}`?

Yes. The `_scan_for_answer` method also matches:
- `final answer is <N>` (natural language)
- `answer is **<N>**` (bold markdown)

These patterns catch cases where the model concludes in natural language instead of LaTeX.
However, `\boxed{}` is the primary format and the follow-up prompt explicitly requests it.

The `last.channel == 'final'` check catches the Harmony protocol's own "final answer" signal
(when the model internally routes its response to the `final` channel). In that case,
`_scan_for_answer` is called on the channel content. If this returns `None` (e.g., the model
said "therefore the answer is N" without `\boxed{}`), `final_answer` remains `None` and the
follow-up would fire — correctly.

---

## Expected Impact

- Attempts that run out of turns or context but have done good reasoning will now commit an answer
- This converts `Answer: None` results into potentially correct answers
- Estimated gain: +1 to +2 problems correct on the 50-problem test set
- Risk: near zero — if the follow-up produces a bad answer, the voting ensemble dilutes it;
  the alternative (no answer) contributes nothing

---

## Files Containing This Implementation

- `/f/kaggle/aimo-progress-prize-3/notebooks/submission_v10_followup.ipynb` — complete working implementation
