# SM70 prefill workspace tuning

The FP32-accumulated Q8000/Q8192 prefill route now retains a 1.50 GiB score
workspace per device instead of 2.25 GiB, saving 768 MiB. This applies wherever
the existing route is selected; no tensor-parallel or model-quantization gate
is added. FP32 accumulation, causal masking, the tail implementation, and
CUDA Graph address lifetime are preserved.

## Configuration

Set `VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS` before worker startup to a
multiple of 8192 in [8192, 131072]. The default is 16384; 24576 restores the
previous capacity. The score buffer uses
`block_tokens * 8192 * 6 * sizeof(half)` bytes per device and is shared by the
Q8000 and Q8192 specializations. Do not change this setting after workspace
initialization. Rebuild the native FA2 extension and restart workers to apply
this change; a Python-only update does not resize an older native extension.
No wheel is required for a source build.

Smaller prefix blocks change the online softmax merge order, so outputs are
not guaranteed to be bitwise identical to the previous capacity. The change
keeps FP16 operands and FP32 accumulation; it does not select FP16 accumulation.

## Matched validation

Integration base: `8d5d82334d0f0b32153fa2c66f9ce587f438ed7c`.
Tested runtime source: `b424eae3e04181758f95ab2c3db41b6affe74662`.
Environment: V100-SXM2-32GB, 185 W per GPU, CUDA 12.8, Torch 2.10.0+cu128.
Model: `Qwen3.8-27B-QUASAR-NVFP4-d8e6fbfa`, FP16 activations;
DFlash2 revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, 7 draft tokens;
target E4M3 KV and draft FP16 KV. Serving uses Flash-V100,
`FULL_AND_PIECEWISE` CUDA Graph, chunk 8192, max sequences 32, GPU memory
utilization 0.85. TP1/TP2 use max length 65536; TP4 uses 262144.
All comparisons use the same source-built artifact: the control explicitly sets
24576, while the candidate has no capacity override. No enforce-eager serving,
preload, private sidecar kernel, or borrowed task library is used.

### Memory

Values below are per device, before KV allocation. TP2/TP4 are matched runs on
the same GPUs; TP1 compares with the preceding audit at the same configuration.

| TP | Profile live allocation, old → new (GiB) | Available KV budget, old → new (GiB) |
| --- | --- | --- |
| 1 | 25.996 → 25.246 | -6.557 → -5.807 |
| 2 | 15.030 → 14.280 | 6.766 → 7.516 |
| 4 | 9.335 → 8.585 | 12.680 → 13.430 |

TP1 still fails the memory admission check for this configuration; no TP1 serving
throughput is claimed. No 16 GiB cards were tested. At fixed memory utilization,
vLLM spends the freed workspace on more KV capacity, so total device usage need
not decrease. TP2 request-resident device usage is unchanged; TP4 changes from
28.201 to 28.244 GiB after the 256K benchmark, including allocator/Graph reserve.
The measurable gain is 768 MiB less non-KV allocation and 768 MiB more KV budget
on every rank, totaling 1.50 GiB at TP2 and 3.00 GiB at TP4.

### Performance

`vllm bench serve` uses random fixed-length prompts, 256 output tokens,
temperature 0.7, top-p 0.8, top-k 20, request seed 20260922, and a cold prefix
cache after kernel warmup. `ignore_eos` is used only for this synthetic speed
contract. Quality requests use natural EOS. Each row is one matched benchmark,
not a statistical confidence interval or a natural-text decode rate.

| TP / input tokens / client concurrency | TTFT old → new (s) | TPOT old → new (ms) | Full request output TPS old → new |
| --- | --- | --- | --- |
| 2 / 32,768 / 1 | 18.047 → 18.028 | 6.210 → 6.281 | 13.040 → 13.041 |
| 4 / 32,768 / 1 | 9.231 → 9.230 | 4.009 → 4.056 | 24.964 → 24.937 |
| 4 / 256,000 / 1 | 124.279 → 124.645 | 10.376 → 10.427 | 2.017 → 2.011 |
| 4 / 2,048 / 1 | 0.617 → 0.612 | 3.513 → 3.547 | 169.137 → 168.753 |
| 4 / 2,048 / 8 | 3.410 → 3.429 | 21.603 → 22.428 | 170.375 → 175.449 |
| 4 / 2,048 / 32 | 13.655 → 11.361 | 41.193 → 42.621 | 223.670 → 220.612 |

The 256,000-token cold TTFT increases 0.29% (124.279 → 124.645 s), corresponding
to approximately 2060 → 2054 input tokens/s including TTFT overhead. TP4 short
C1/C8/C32 full-request throughput changes -0.2%/+3.0%/-1.4%. Per-request median
TPOT varies more; retain the separate columns rather than calling every metric
unchanged. DFlash streaming intervals can contain multiple tokens and are not
per-token decode latency. No prefix hits or KV preemptions occurred in these
benchmark intervals. C32 is offered client concurrency: 0.5-second server
sampling observed peak active requests of 15 (control) and 16 (candidate). This
is not evidence of 32 requests simultaneously resident on the GPU.

### Numerical and output checks

Matched CUDA Graph operator results at 128K/256K across Q8000/Q8192 show
0.49–0.81% latency increase with 16384. The 8192 option saves 1.50 GiB but
increases latency 2.00–2.26%, so it was rejected as the default. All sampled
outputs are finite; FP32 oracle relative L2 is 0.00054–0.00184. The normal-case
output delta between 16384 and 24576 is at most 1.52587890625e-5. The 24576
override is bitwise identical to the previous binary on all five tested shapes.
256K biased-value/high-score and periodic-score stress cases pass the FP32
oracle and exact repeated Graph replay.

TP2 32K and TP4 32K/128K/262128-token retrieval responses are correct, identical
between capacities, and finish with natural EOS. The boundary request uses
262128 input plus 16 output tokens. The paired MBPP32 and four supplementary
32K-prefix code checks used for promotion are recorded in
[PR #673](https://github.com/1CatAI/1Cat-vLLM/pull/673); their answer limit is
65536 tokens and scoring runs all held-out tests.

22 CPU allocator/distributed tests and 21 attention/memory GPU tests pass,
including a fresh process with no capacity override. Focused suites:

```bash
pytest -q tests/v1/worker/test_gpu_worker_allocator.py \
  tests/distributed/test_custom_all_reduce_allocator.py
pytest -q tests/v1/worker/test_gpu_worker_memory_profile.py \
  tests/kernels/attention/test_sm70_79t_stability.py \
  tests/kernels/attention/test_sm70_prefill_score_capacity.py
```

Task-owned source/build commands, native hashes, mapped-library provenance,
benchmark commands, per-request results, and failed harness launches are retained
in the local handoff outside Git. Early harness failures (overlong Unix socket
path, occupied GPU, unwritable temporary directory) occurred before inference
and are excluded from model results. GPU ownership uses shared flock files;
unrelated services are left running.

## Allocator and budget accounting

The worker preserves the complete allocator configuration while temporarily
changing `max_split_size_mb` for model loading. The previous partial settings
update reset user rounding and garbage-collection options and ignored the unified
`PYTORCH_ALLOC_CONF` alias. The post-capture KV capacity suggestion now includes
persistent warmup allocation/reserve, as the actual KV budget already does.
This corrects the suggestion without increasing the budget or removing the
Graph reserve.

Expandable allocator segments were rejected as a universal default in the prior
audit: they improved TP1's estimated KV budget but reduced TP2's budget and
increased TP2 request-resident memory. This change is separate from workspace
growth and decode Graph pointer lifetime fixes in PRs 660/661.
