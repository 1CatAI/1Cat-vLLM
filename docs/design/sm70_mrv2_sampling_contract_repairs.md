# MRv2 sampling contract repairs

This batch repairs three independently reproduced sampling defects on top of
`099d9841f542f1b71121b4aff49e3aa29053a489`. They affect the shared MRv2 runner,
including Qwen3.8-Flash-Next-NVFP4; none changes model weights, attention,
quantization, or the precision of model kernels.

## Defects and scope

| Request | Defect | Repair |
| --- | --- | --- |
| `min_tokens > 0` with stop IDs | EOS remains masked for one extra step. | Compare the input-token position against `prompt_len + min_tokens - 1`. |
| `logprobs=-1`, when the server allows full-vocabulary logprobs | Public `-1` aliases the internal "no logprobs" marker: a single request loses its logprobs, and a mixed batch can truncate them. | Normalize the public value to vocabulary size when adding the request. |
| Bad words with speculative rows | The draft-prefix lookup reads the last committed token as draft zero, producing both false bans and missed bans. | Skip expanded input row zero when reading the draft prefix. |

For `L` prompt tokens and `t` already generated tokens, the logits input's
zero-based position is `L + t - 1`. The stop condition must therefore mask
exactly while `t < min_tokens`. The regression compares both the mask and the
actual greedy sampled token against the existing min-token reference.

Expanded speculative inputs are `[last_committed, draft_0, draft_1, ...]`.
Row `r` must match bad words against committed output followed by the first
`r` draft tokens. Non-speculative row zero never reads a draft. Tests compare
the complete output masks exactly with the existing CPU bad-word reference.

Full-vocabulary logprobs still require a server configuration that permits
them, such as `--max-logprobs -1`. This repair does not bypass server limits.
The existing internal output layout (sampled token followed by requested top-k)
is preserved. Full-vocabulary requests thus have `vocab_size + 1` columns, with
the sampled token appearing again in the vocabulary columns.

The full-output logprob calculation uses FP32 `log_softmax` followed by a gather.
The small-output Triton calculation is unchanged. This avoids compiling a
single enormous Triton output program for 248,320 vocabulary entries; the old
program passed correctness tests but incurred a long first compilation. The
dense branch uses an additional FP32 vocabulary-sized intermediate, so full
logprobs remain a memory-intensive, opt-in diagnostic feature. It does not feed
back into token selection.

## Focused validation

Environment: NVIDIA V100-SXM2-32GB (SM70), driver 580.173.02, Torch
2.10.0+cu128, Triton 3.6.0, and installed CUDA toolkit 12.0.140. Tests use one
idle device, the source overlay with the accepted native extension bundle,
and task-owned compiler caches. No model weights are initialized and no
environment dependencies or native binaries are rebuilt.

From the checked-out source tree, with its runtime installed or the documented
source-overlay bootstrap configured:

```bash
.venv/bin/python -m pytest -q --tb=short --confcutdir=tests/v1/worker \
  tests/v1/worker/test_gpu_sampler_runtime_contracts.py
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q --tb=short \
  --confcutdir=tests/v1/worker \
  tests/v1/worker/test_gpu_sampler_runtime_states.py \
  tests/v1/worker/test_gpu_model_runner_v2_greedy.py
```

- Original 20 GPU reproductions on the base source: **16 failed, 4 passed**.
  Nine min-token boundaries, four full-logprob requests, and three speculative
  bad-word cases fail independently. Default/non-speculative controls pass.
- Fixed, expanded GPU suite: **28 passed in 6.11 s**. This includes eager and
  CUDA Graph checks, changing draft prefixes, request-slot reuse, FP16/FP32
  inputs, vocabularies 257 and 248,320, single/mixed requests, and prompt
  logprobs across the 1,024-token chunk boundary.
- CPU metadata and existing greedy-dispatch tests: **17 passed in 7.31 s**.
- Logprob values match FP32 PyTorch references at `atol=2e-5, rtol=1e-5`;
  mask/token checks are exact. This is not a claim of bitwise equality of
  different log-softmax algorithms or a full-model numerical certification.

The GPU test durations include cache-dependent initialization and are not
model throughput measurements. The initial repaired 20-test suite, before
the dense logprob branch, also passed (135.66 s including compilation).
Do not interpret that difference as a steady-state sampling speedup.

## Baseline and remaining scope

The accepted approximately 98 tok/s no-MTP baseline uses `min_tokens=0`, no
bad words, and no requested logprobs. Its greedy dispatch is unchanged and
covered by the state tests. It was **not rebenchmarked** for this batch; no
new prefill/decode performance claim is made. These tests validate the changed
runtime boundaries without another full-model startup.

The apparent tied-logprob rank issue was rejected: the existing reference
uses the same `>=` convention. Earlier ngram EOS/padding probes also passed.
Neither is counted as a repaired bug. Actual-input GDN/W13 numerical auditing
and a broader end-to-end model quality evaluation remain separate work.
