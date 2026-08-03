# SM70 DeepSeek V4 TP8 Graph All-Reduce

## Scope

This route targets the exact batch-one DeepSeek-V4-Flash decode collective:

- 8 Tesla V100-SXM2 GPUs, TP8, CUDA Graph, and no eager execution;
- FP16 input/output with 4096 elements, or 8 KiB per rank;
- topology `0-3` and `4-7` as four-GPU NVLink cliques, with NV2 links
  `0-4`, `1-5`, `2-6`, and `3-7`;
- 87 all-reduces per decoded token.

It is guarded by `VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR=1`. Other tensor
sizes, dtypes, GPU capabilities, world sizes, and topologies retain their
existing communicator path.

## Baseline Diagnosis

The accepted 1024-input/256-output TP8 baseline is 25.630149 ms/token, or
39.0166 token/s. Nsight Systems reports 4.192 ms/token of NCCL kernel service,
but the transport kernel is not the whole cost.

Across 10,788 steady collectives:

| Phase | Mean |
|---|---:|
| Rank-local input-ready skew | 28.817 us |
| NCCL kernel-start skew | 48.083 us |
| Last NCCL start after every rank input is ready | 38.564 us |
| Collective end after every rank input is ready | 63.740 us |
| Transport tail after the last NCCL start | 25.176 us |

The graph-captured NCCL launch is collective. Its rank coordination delays the
last kernel well after every input is ready. Changing Ring to Tree reduced an
isolated transport kernel but moved full-model TPOT only 0.014 ms, so that
path was rejected.

## Hierarchical Algorithm

Each rank opens only the IPC mappings required by its four-rank clique and
its paired rank. One 512-thread CTA then:

1. publishes input readiness within the local four-rank clique;
2. reads ranks in fixed global-rank order and forms an FP32 clique partial;
3. stores the FP32 partial in peer-visible metadata;
4. exchanges readiness and the partial over the paired NV2 link;
5. adds the paired partial in FP32 and performs one final FP16 downcast.

The fixed rank order and FP32 intermediate are deliberate. The optimization
does not use reduced precision, skip ranks, or change the tensor shape.
Required P2P edges and IPC opens are validated before enabling the route.

The graph-safe protocol uses two alternating slots for clique-ready signals,
pair-ready signals, and FP32 partials. After consuming the four local inputs,
threads 0-3 publish clique completion while thread 4 exchanges the paired
partial-ready signal. These two handshakes run concurrently. This prevents a
faster four-GPU clique from overwriting a signal, input, or partial still in
use by the slower clique without adding a serialized eight-rank barrier.

## Correctness

The exact 8-rank CUDA Graph microbenchmark reports:

| Check | NCCL | Hierarchical |
|---|---:|---:|
| Max absolute error vs FP32 reference | 0.136719 | 0.061523 |
| Mean absolute error vs FP32 reference | 0.014117 | 0.006093 |
| Bitwise equal across all ranks | yes | yes |

The hierarchical result is numerically closer to the FP32 reference because
it retains FP32 clique partials until the final downcast. A model-level
official-sampling quality gate is still required before enabling the route by
default; stochastic token identity is not used as a component correctness
test.

## Performance

The multi-stream graph-join microbenchmark reproduces the projection/event
join surrounding each model collective:

| Route | Median per call | Projected 87-call time |
|---|---:|---:|
| NCCL Ring LL | 39.384 us | 3.426 ms |
| Hierarchical | 32.154 us | 2.798 ms |
| Saving | 7.230 us | 0.629 ms/token |

The full model exposes a larger benefit because the custom kernel launches
independently on each rank and waits on device instead of adding a collective
graph-launch delay:

| Metric | NCCL baseline | Hierarchical | Change |
|---|---:|---:|---:|
| TPOT, three-run mean | 25.630149 ms | 23.460869 ms | -2.169281 ms (-8.46%) |
| Decode throughput | 39.0166 token/s | 42.6242 token/s | +9.25% |
| Three-run TPOT stdev | - | 0.020381 ms | - |

The candidate runs are 23.465680, 23.438512, and 23.478413 ms/token under the
same prompt, seeds, sampling parameters, graph mode, and route stack.

The candidate graph-node trace closes the endpoint result:

| Phase | NCCL | Hierarchical | Change |
|---|---:|---:|---:|
| Last kernel start after all inputs ready | 38.564 us | 0.924 us | -37.640 us |
| Collective end after all inputs ready | 63.740 us | 40.835 us | -22.905 us |
| Projected 87-call critical saving | - | - | 1.993 ms/token |

Custom-kernel GPU service is 4.490 ms/token versus 4.192 ms for NCCL. The win
is therefore graph critical-path scheduling, not a misleading reduction in
summed kernel service.

## Graph-Skew Stability Fix

Combining the first hierarchical kernel with the sparse-MLA QK D-split made a
latent graph race reproducible. Two 32-token warmups completed near 20.5
ms/token, but the following request stopped after 22 emitted tokens. Disabling
hierarchical all-reduce completed 128/128 tokens at 23.198 ms/token, isolating
the fault to rank-skew tolerance rather than sparse-MLA arithmetic.

The original protocol reused one exact-valued signal slot and one FP32 partial
buffer. A fast clique could advance by one collective before its peer consumed
the previous slot. Merely double-buffering the partial changed timing and made
the same signal overtake deadlock reproducible in the 87-call microbenchmark.
The accepted protocol therefore double-buffers both signals and data and
acknowledges local-input consumption in parallel with the pair exchange.

Validation after the fix:

| Gate | Result |
|---|---:|
| Pure CUDA Graph stress | 8,700 collectives, 17.099 us/call, complete |
| Eight-rank numerical result | Bitwise equal across ranks |
| Graph-join microbenchmark | 32.154 us/call versus 39.384 us NCCL |
| Stacked 1024/256 endpoint | 20.768 / 20.758 / 20.770 ms/token |
| Stacked endpoint mean | 20.765 ms/token, 0.006 ms stdev |

The stacked endpoint also includes compact/direct MXFP4 and the sparse-MLA QK
D-split candidate, so 20.765 ms is not an all-reduce-only A/B. It proves that
the corrected communication protocol transfers through the current combined
graph and remains stable for three full 256-token requests.

## Rejected Paths

| Path | Evidence | Decision |
|---|---|---|
| NCCL Tree/LL | 21.584 us isolated versus 33.122 us Ring, but only 0.014 ms full-model movement | Reject; does not remove collective graph launch coordination |
| 87 consecutive collectives | 16.055 us custom versus 16.021 us NCCL | Diagnostic only; it pipelines away the model's rank-join bubble |
| 256 threads, two FP32 partials per thread | Eight GPUs remain in low-power 100% synchronization spin | Reject and remove |
| 128 threads, four FP32 partials per thread | Not run after the 256-thread protocol failure | Reject this family |
| One signal/partial slot | Full model stopped after 22 tokens under larger graph skew | Reject; unsafe slot reuse |
| Partial double-buffer only | 87-call graph microbenchmark entered synchronization spin | Reject; signal overtake remains |
| Serialized clique completion | Stable at 21.055 ms/token in the stacked endpoint | Correct diagnostic; replaced by concurrent acknowledgement at 20.765 ms/token |

## Artifacts

```text
/home/fudanwl/v100-worktrees/runs/
  dsv4-tp8-hierarchical-ar-micro-20260803/
  dsv4-tp8-hierarchical-ar-fullmodel-20260803/
  dsv4-tp8-hierarchical-ar-nsys-i1024-o128-20260803/
  dsv4-tp8-hierarchical-ar-threads-micro-20260803/
  dsv4-sparse-mla-qk-dsplit-fullmodel-20260803/
```

## Remaining Gates

1. Run the model-specific official-sampling quality suite.
2. Keep the route default-off until that suite passes.
3. Verify a second host with the same required clique-and-pair topology.
4. Do not generalize this kernel to other shapes without separate numerical,
   graph, and endpoint evidence.
