# Sampling input guards for the Qwen3.8 runtime

This follow-up to PR533 fixes three request-boundary defects on public main
`4f19ef7a20db60bb0685e599bd3f4dd156202eed`. Only `SamplingParams` production
code changes; model weights, precision, CUDA kernels and default sampling are
unchanged.

## Confirmed defects

1. **Stop IDs were not bounded by the model vocabulary.** A negative or
   out-of-vocab `stop_token_ids` entry passed model-aware verification.
   MRv2's minimum-token mask uses these IDs as raw logit indices, so enabling
   `min_tokens` could turn a malformed request into an out-of-row GPU store.
   Validate user stop IDs and already-merged stop metadata before scheduling.
   No malformed IDs were deliberately executed on CUDA during the audit.
2. **Allowed IDs were validated only against the tokenizer.** With
   `skip_tokenizer_init`/token-ID-only requests, there was no range validation;
   a tokenizer larger than the model could also admit indices beyond the
   model's logit width. Validation now uses the model width even without a
   tokenizer, while retaining the existing tokenizer limit when present.
3. **Whitespace-only bad words became empty token sequences.** Stripping
   leading whitespace turned spaces, tabs and newlines into empty strings,
   and prefix handling then indexed an empty list. Keep literal whitespace
   sequences. If a tokenizer still produces no tokens, return a structured
   `VLLMValidationError` instead of an `IndexError` or an empty GPU bad-word
   sequence. Normal-word prefix handling is unchanged.

The actual local RadixArk Qwen3.8-Flash-Next-NVFP4 tokenizer reproduced
`IndexError` for space, double space, Tab and newline before the fix.
After the fix their literal IDs are respectively 220, 256, 197 and 198, with
nonempty prefix variants. English and Chinese controls retain their old IDs.
This tokenizer has 248,077 entries; the model logit width is 248,320.

## Validation

The initial CPU suite on the integration base had **24 failures and 11 passing
controls in 4.25 s**. This includes model vocabulary sizes 128 and 248,320,
negative and upper-bound IDs, tokenizer absence/mismatch, whitespace, and
empty encodings. The final CPU suite plus existing state/greedy regressions
has **67 passes in 6.84 s**, including chat request conversion through actual
`InputProcessor._validate_params`.

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q --tb=short \
  --confcutdir=tests/v1/worker \
  tests/v1/worker/test_sampling_input_guards.py \
  tests/v1/worker/test_gpu_sampler_runtime_states.py \
  tests/v1/worker/test_gpu_model_runner_v2_greedy.py
.venv/bin/python -m pytest -q --tb=short --confcutdir=tests/v1/worker \
  tests/v1/worker/test_sampling_input_guards_gpu.py
```

Use the installed source runtime or the repository's source-overlay bootstrap.
Tests load no model weights. CPU masks compare exact values, valid token lists
remain unchanged, and invalid IDs are checked on the CPU instead of inducing
a GPU fault. The GPU suite checks valid minimum-token/allow masks at model
vocabulary width and bad-word preprocessing against the existing CPU oracle.

At publication, local scoped pre-commit passed. The 11 GPU controls were added
but not run: the cooperative GPU groups were occupied, so the test launcher
exited without taking over another task's devices. No full model or service
was started for this CPU preprocessing change.

This audit is not a new 98 tok/s benchmark or a universal output-quality
certificate. Validated behavior is the changed input boundary; broader GDN/W13
numerical auditing, late runner capacity limits and generated EOS configuration
validation are separate scopes. The no-option fast decode path is untouched.
