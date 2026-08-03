# SM70 DeepSeek V4 MoE TP8 Join

## Scope

This route targets batch-one DeepSeek-V4-Flash decode on eight V100 GPUs. It
keeps TurboMind MXFP4, FP8 shared experts, CUDA Graph execution, top-k=6, and
the existing hierarchical TP8 all-reduce. It is stacked on the active-expert,
FP8 dense, hierarchical all-reduce, and sparse-MLA QK-D worktrees.

## Trace Diagnosis

The 64-token graph-node trace contains 87 all-reduces per token. The 44
elementwise-preceded calls have 44.333 us mean input-ready skew, versus 18.527
us for the 43 FP8-GEMM-preceded calls. The former are the shared+routed MoE
boundary. Their 43 steady MoE segments average 167.386 us/layer, including
43.011 us of routed MXFP4 GEMM and 28.022 us of shared FP8 GEMM.

The generic shared-expert auxiliary stream is active. An exact-shape graph
screen measures 110.653 us/layer when serialized and 98.318 us/layer with the
existing overlap, so disabling or replacing that overlap is not useful.

## Candidate

The direct top-6 MXFP4 path now reuses the exact single-token weighted-reduce
kernel. When `VLLM_SM70_MXFP4_MOE_FUSED_SHARED_REDUCE=1`, the fused variant
also applies the production FP16 shared-expert scale and performs the following
FP16 shared+routed add before writing the all-reduce input. W13, SwiGLU, W2,
route order, FP32 FMA order, both FP16 downcast points, and the hierarchical
cross-rank reduction order remain unchanged.

The generic `FusedMoEMethodBase` finalization hook is a no-op for every other
quantization method. The MXFP4 hook is restricted to batch one, direct top-6,
fully replicated experts, a present shared expert, and the new operator.

## Rejected V1 Screening

The first fused kernel omitted DeepSeek-V4-Flash's
`routed_scaling_factor=1.5` compensation. Production computes
`fp16(shared * (1 / 1.5)) + routed`, while V1 computed `shared + routed` and
would then have allowed the runner to scale the combined tensor again. It was
rejected before endpoint timing. The following results only quantify why the
fusion idea was retained for a numerically corrected V2; they are not accepted
performance evidence.

Single-GPU exact-shape CUDA Graph:

| Route | Median per layer | Change from current overlap |
|---|---:|---:|
| Current unpermute + add | 97.861 us | - |
| Exact weighted-reduce + add | 94.537 us | -3.323 us |
| Fused weighted-reduce-add | 92.620 us | -5.240 us |

The fused route projects to 0.225 ms/token over 43 layers. Its output is
bitwise equal to generic `moe_unpermute` followed by FP16 add.

The eight-rank joined graph includes the MoE tail and hierarchical all-reduce:

| Route | Rank-max median per layer | 43-layer projection |
|---|---:|---:|
| Unpermute + add + hierarchical AR | 118.437 us | - |
| Fused reduce-add + hierarchical AR | 113.899 us | -0.195 ms/token |

Initial and changed-input graph replays were bitwise equal to the incomplete
V1 control. V2 must instead match the production scale-and-add sequence with
zero tolerance.

## V2 Microbenchmarks

The corrected oracle performs the same FP16 shared scaling before the same
FP16 add. On one GPU:

| Route | Median per layer | Change from current overlap |
|---|---:|---:|
| Current unpermute + scale + add | 99.165 us | - |
| Exact weighted-reduce + scale + add | 95.962 us | -3.203 us |
| Fused weighted-reduce-scale-add | 92.585 us | -6.580 us |

The fused route projects to 0.283 ms/token over 43 layers. Its output is
bitwise equal to the production sequence.

The eight-rank joined graph includes the corrected MoE tail and hierarchical
all-reduce:

| Route | Rank-max median per layer | 43-layer projection |
|---|---:|---:|
| Unpermute + scale + add + hierarchical AR | 120.290 us | - |
| Fused reduce-scale-add + hierarchical AR | 113.989 us | -0.271 ms/token |

Initial and changed-input graph replays are bitwise equal to the corrected
control on all eight ranks. The focused CUDA test covers both scale 1.0 and
the model's `1 / 1.5` compensation with zero tolerance.

## Full-Model Result

The low-overhead endpoint A/B used the same source, binary, prompt, TP8
topology, FP8 KV cache, CUDA Graph mode, and official `temperature=1`,
`top_p=1` sampling. Each side ran two 32-token warmups followed by three
256-token requests.

| Route | Mean TPOT | Decode throughput | Change |
|---|---:|---:|---:|
| Generic unpermute + scale + add | 21.3091 ms | 46.928 token/s | - |
| Fused reduce-scale-add | 21.2797 ms | 46.993 token/s | -0.0294 ms (-0.138%) |

All six 256-token requests completed with no NUL, replacement character, or
single-character repetition. The candidate's 0.0294 ms endpoint saving is
only 10.9% of the 0.271 ms joined-microbenchmark projection. It is too small
to justify defaulting the extra runner state and custom-op ABI path, so the
route remains default-off and work moves to the larger TP collective and
rank-arrival costs.

The historical 20.7655 ms stacked result used a different evolving dependency
set and is not the control for this A/B. The fused route is also not promoted
as a replacement baseline while it remains slower than that recorded stack.

## Rejected Paths

| Path | Result | Decision |
|---|---:|---|
| Main stream priority -1/-2 | At most 0.004 ms/token projected | Reject |
| One-CTA hierarchical sum2 | 18.096 to 20.150 us/call | Remove; local add and fence dominate |
| Fused reduce-add, 128 threads | Lower than 256-thread benefit | Reject |
| Fused reduce-add, 64 threads | 0.191 ms/token in TP8 join | Reject; keep 256 |
| Scale-free fused reduce-add V1 | Omits routed scale compensation | Reject; output semantics differ |

## Artifacts

```text
/home/fudanwl/v100-worktrees/runs/dsv4-tp8-latest-graphtrace-20260803/
/home/fudanwl/v100-worktrees/runs/dsv4-shared-moe-priority-micro-20260803/
  screen_v3_fused_reduce_add.json
  moe_tp8_join_v1.json
  moe_tp8_join_t64.json
  tp8_hierarchical_sum2_v1.json
  screen_v4_scaled_fused.json
  moe_tp8_join_scaled_v3.json
/home/fudanwl/v100-worktrees/runs/dsv4-moe-fused-reduce-control-20260803/
/home/fudanwl/v100-worktrees/runs/dsv4-moe-fused-reduce-fullmodel-20260803/
```

## Decision

Keep the route default-off. Do not spend another full-model cycle tuning this
tail unless a new design materially increases the joined critical-path saving.
