# Flash-Next prefill with sparse prefix-cache checkpoints

The default `prefix_cache_retention_interval=0` retains safe replay and
detected shared-prefix boundaries. Previously the align-mode scheduler still
stopped at every recurrent-state block, including states that admission would
discard. On Flash-Next with native MTP4 the state block is 816 tokens: this
turned an 8192-token prefill budget into many complete model forwards, including
the dense projections, experts and TP collectives.

Sparse scheduling now stops at both replay boundaries, detected shared-prefix
boundaries, and any explicitly configured periodic retention boundary. It can
batch across discarded states. The state grid, KV page size, cache lookup and
MTP back-off are unchanged. Explicit dense retention (`None` through Python)
and mixed cache alignments that require dense admission retain the old split.
The allocator continues to materialize one running state at each chunk end;
the scheduler must end a chunk at every boundary that admission will retain.

This is a local adaptation of the retention-aware scheduling idea discussed
in upstream [PR #53479](https://github.com/vllm-project/vllm/pull/53479).
Its other proposed changes, including removing the speculative block back-off,
are outside this fix. Internal state exporters for Mamba2 in upstream
[PR #57329](https://github.com/vllm-project/vllm/pull/57329) target a different
backend and are not needed to skip states the current policy discards.

## Baseline and validation

Baseline: main `d30469863287471a7082842500ae73299a697e0d`, native 1.5.1
wheel, Torch 2.10.0+cu128, CUDA 12.8, 4x V100-SXM2-32GB, TP4/PP1/DCP1,
`RadixArk/Qwen3.8-Flash-Next-NVFP4`, FP16 activations/KV, native MTP4,
V2 runner, FULL_AND_PIECEWISE graphs, synchronous scheduling, max length
131072, max sequences 1, token budget 8192, memory utilization 0.90,
PLE host budget 12 GiB per rank. The PLE budget and toolkit configuration
remain explicit diagnostics; this is not the zero-configuration release gate.

Two cold requests per input length use independent cache salts and compute
all input tokens. Timing uses the native completed-request prefill histogram,
separately from client TTFT. Exact-length design-document fixtures use
temperature 0, seed 0, max output 16 and normal EOS; natural health requests
use checkpoint temperature 1, top-k 20, top-p 0.95 and normal EOS.

| Input tokens | Cache off tokens/s | Cache on tokens/s | On throughput change |
| ---: | ---: | ---: | ---: |
| 8192 | 5821 | 2487 | -57.3% |
| 32768 | 5409 | 3046 | -43.7% |
| 131040 | 4734 | 2818 | -40.5% |

The first 8K request includes JIT work; the second alone still loses 50.0%.
Default warm repeats reuse 7344, 31824 and 129744 tokens respectively.
Increasing the grid to 8160 partially recovers cold throughput but removes
the 8K repeat hit, so that configuration is not the fix.

CPU regressions exercise both replay boundaries at exact/partial block ends,
shared junctions, periodic retention, sub-block progress, dense fallback,
mixed alignments and allocation/admission together. Every admitted state must
belong to the actual chunk end, with identical-repeat lookup preserved.

Candidate wheel performance, repeated-request quality, long-context quality
and decode regression validation are pending. No measured candidate speedup
is claimed until these gates complete. AWQ, E4M3 and concurrent prefill are
outside the existing baseline evidence.
