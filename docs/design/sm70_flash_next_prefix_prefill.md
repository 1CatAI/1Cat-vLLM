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

When a recurrent-state block is at least the scheduler's token budget,
contending requests retain dense scheduling boundaries to preserve concurrent
decode throughput. A single request keeps sparse replay boundaries; applying
the contention fallback unconditionally changed prefix reuse and reduced
32K speculative decode throughput in the release check. Flash-Next's
816-token state block with an 8192-token budget continues to use sparse
scheduling for both single and concurrent requests.

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

## Candidate wheel result

Candidate source `8daf57282e107752378be6551a183720422def14`, wheel SHA256
`dbd63e4d6007ffd88e5f8ba37a6770fa3d9e61cae81fb07f744f4e593b3c54bf`.
The wheel installs with its declared dependencies in a fresh uv environment.
All 15 native binary hashes match the main wheel; the installed scheduler
matches the owned source. No source overlay, copied private extension or
new performance flag is used. Key runtime dependency versions match baseline.

| Input tokens | Main prefix-on tokens/s | Fixed prefix-on tokens/s | Main / fixed warm prefill seconds | Reused tokens, both |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 3060 | 5356 | 0.438 / 0.237 | 7344 |
| 32768 | 3046 | 5068 | 0.479 / 0.282 | 31824 |
| 131040 | 2818 | 4666 | 0.723 / 0.416 | 129744 |

The 8K throughput row uses the second, warmed cold request in both runs:
the candidate's first request still includes JIT and takes 4.049 seconds,
versus 1.529 seconds on its second. The 32K/128K rows use total computed
tokens divided by total prefill time across two independent cold requests.
Each configuration has only one successful startup. Relative to prefix off,
the fixed second 8K remains 12.6% slower; 32K/128K means remain 6.3%/1.4%
slower. Retained boundary forwards still have a cost; this does not promise
identical cold performance to disabling caching.

All nine performance responses finish with normal EOS and the same three
token IDs as main. Six fact/trace interpretation requests over real 32K/128K
design-document contexts pass the answer oracle with checkpoint sampling;
repeat outputs are token-identical and reuse 31008/128928 tokens. The short
natural response is identical to main's 315-token response; native decode
time is 3.967 versus 3.941 seconds (+0.65%, single request).
These are focused quality checks, not a broad model accuracy benchmark.

All 152 CPU regressions and scoped pre-commit pass. The service exits zero
and releases all four GPUs. Candidate startup takes 450.35 seconds with empty
task caches; compilation-cache defaults, first-request JIT coverage, toolkit
dependence and the separate 0.95-memory failure remain release work. AWQ,
E4M3, concurrent prefill and the 256K boundary are untested in this fix.
