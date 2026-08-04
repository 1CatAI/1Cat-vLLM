# SM70 DeepSeek V4 TP8 Two-CTA All-Reduce

## Scope

This rejected experiment targeted the exact DeepSeek-V4-Flash TP8 decode collective:
4096 FP16 elements, eight V100 GPUs, CUDA Graph execution, and the existing
two-clique hierarchical reduction order. It is gated by
`VLLM_SM70_TP8_HIERARCHICAL_TWO_CTA=1`. The implementation and environment
gate were removed after the microbenchmark regression.

## Trace Rationale

The latest trace executes 87 hierarchical collectives per token. Their summed
rank service is about 4.50 ms/token, while MoE-preceded calls also expose about
1.95 ms/token of rank-arrival skew. The accepted one-CTA kernel assigns all 512
packed values to one SM. This experiment tests whether two independent CTAs
can hide peer-memory latency without changing arithmetic.

## Protocol

Each 256-thread CTA owns one half of the packed tensor and one independent
double-buffered signal counter. Both CTAs preserve global rank order when
forming the FP32 four-rank clique partial, exchange the FP32 partial with the
same paired rank, then perform the same final FP16 downcast. Signal and data
slots are not reused until the paired consumer acknowledges the matching CTA.

## Gates

1. Eight-rank initial and changed-input outputs must be bitwise equal to the
   one-CTA route.
2. The 87-call CUDA Graph stress must complete repeatedly without signal
   overtake or a GPU synchronization spin.
3. The joined projection/collective microbenchmark must reduce rank-max wall
   time before any full-model launch.
4. A candidate that only moves isolated service but not joined wall time is
   rejected.

## Result

Both routes produced the same SHA-256 output on all eight ranks. The two-CTA
graph completed the short 87-call stress without signal overtake.

| Test | One CTA | Two CTA | Change |
|---|---:|---:|---:|
| Pure 87-call graph | 18.710 us/call | 19.879 us/call | +6.25% |
| Projection join + collective | 32.498 us/call | 35.035 us/call | +7.80% |

The joined regression projects to `+0.221 ms/token` over 87 calls. At this
8 KiB shape, duplicating clique and pair handshakes costs more than using a
second SM saves in peer-memory latency. The candidate was removed without a
full-model launch.

Artifacts:

```text
/home/fudanwl/v100-worktrees/runs/dsv4-tp8-two-cta-ar-micro-20260803/
  one_cta_short_pure.json
  two_cta_short_pure.json
  join_two_cta_0.json
  join_two_cta_1.json
```

## Rank-Arrival Follow-Up

CUDA graph nodes cross the next host replay range boundary in this trace. A
timestamp-window assignment therefore reports 85-89 collectives for some
rank/replay pairs even though every replay contains exactly 87. The retained
analyzer aligns the full per-rank collective sequence by ordinal first, then
splits it into fixed 87-call replays.

Across 61 steady replays and 5,307 aligned collectives:

| phase | mean | p90 | sum/token |
|---|---:|---:|---:|
| Rank arrival skew | 31.578 us | 51.589 us | 2.747 ms |
| Tail after last rank arrives | 40.532 us | 51.823 us | 3.526 ms |
| First-start to last-end envelope | 72.110 us | 90.810 us | 6.274 ms |

The 43 attention segments average 18.527 us of arrival skew. The 43 MoE
segments average 41.363 us and 22.159 us of rank-local idle time. No rank is a
fixed straggler: rank 2 is last most often at 18.64%, but the last rank rotates
across all eight GPUs. GPU remapping or a different clique pairing therefore
cannot remove this skew.

The trace also shows different TurboMind FP8 tactics on different ranks, but
total FP8 service spans only 5.502-5.593 ms/token. Selecting the best observed
tactic everywhere has less than 0.1 ms/token rank-max headroom and is below the
production-change threshold.

## Router-First Shared Expert Screen

The MoE timeline runs routed MXFP4 work on the main stream and shared FP8 work
on an auxiliary stream. A default-off prototype delayed the auxiliary-stream
release until the top-k router completed, without changing any arithmetic.

| Test | Root release | Router first | Projected saving |
|---|---:|---:|---:|
| Single-GPU exact-shape graph | 81.741 us/layer | 75.615 us/layer | 0.263 ms/token |
| TP8 graph joined to hierarchical AR | 102.664 us/layer | 101.142 us/layer | 0.065 ms/token |

Initial and changed-input TP8 graph replays were bitwise equal to the control
on every rank. The TP8 joined saving is nevertheless below the 0.2 ms/token
gate, and the earlier shared-tail experiment showed that joined projections
can realize only about 11% end to end. The production scheduling change was
removed without a full-model launch.

Follow-up artifacts:

```text
/home/fudanwl/v100-worktrees/runs/dsv4-tp8-latest-graphtrace-20260803/
  tp8_collective_skew.json
  tp8_collective_skew.md
/home/fudanwl/v100-worktrees/runs/dsv4-tp8-shared-moe-stagger-20260803/
  single_gpu_v1.json
  tp8_join_v1.json
```

The next independent trace hotspot is the existing FP16 GEMV/compressor scope
in Draft PR #168. This branch does not duplicate that production route.
