# Learning: Harmony Protocol is Non-Negotiable

## Discovery
GPT-OSS-120B was trained on the Harmony token-level protocol. Using the standard OpenAI
chat/completions API puts it into an off-distribution mode where it cannot function.

## Evidence
- v13 (chat/completions, text TIR): 1/50
- v18 (Harmony completions, native tools): 3/3 test problems correct

## Technical Details
- Must use `openai_harmony` library for encoding/decoding
- Must use `client.completions.create(prompt=token_ids)` NOT `client.chat.completions.create(messages=...)`
- Stop tokens: `encoding.stop_tokens_for_assistant_actions()` (integer IDs)
- Tool calling: `last_message.recipient == 'python'` (native token routing)
- Reasoning: `ReasoningEffort.HIGH` activates extended thinking
- Requires `openai_harmony` package + tiktoken encodings (offline)

## Impact
This is a ~40 point improvement. Nothing else matters if this isn't correct.
