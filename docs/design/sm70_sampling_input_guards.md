# Sampling input guards

`SamplingParams.verify` validates sampling indices against the model's logits
vocabulary before requests reach a worker. Stop IDs include the user-provided
IDs and previously merged stop metadata. This prevents minimum-token masking
from indexing outside a logits row.

Allowed IDs are bounded by the model vocabulary even for token-ID requests
that skip tokenizer initialization. When a tokenizer is present, its existing
vocabulary limit also applies. Invalid indices produce `VLLMValidationError`
with the parameter name and offending values.

Bad-word preprocessing retains literal whitespace-only strings. Spaces, tabs
and newlines can therefore be tokenized and blocked as requested. A tokenizer
that produces an empty sequence yields a structured validation error before
prefix handling indexes that sequence. Normal-word prefix handling and valid
sampling masks retain their existing behavior.

## Validation

CPU tests cover negative and upper-bound IDs, tokenizer absence and vocabulary
mismatch, literal whitespace, empty encodings, valid boundary preservation,
and chat-request conversion through `InputProcessor`:

```bash
bash tools/merge_gate.sh \
  tests/v1/worker/test_sampling_input_guards.py \
  tests/v1/worker/test_gpu_sampler_runtime_states.py \
  tests/v1/worker/test_gpu_model_runner_v2_greedy.py
```

`tests/v1/worker/test_sampling_input_guards_gpu.py` additionally checks valid
minimum-token and allowed-token masks on CUDA. Invalid indices are verified
on the CPU before worker execution.
