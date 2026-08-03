# SM70 DeepSeek V4 TP8 Small All-Reduce

## Scope

DeepSeek-V4-Flash TP8 batch-one decode performs 87 FP16 all-reduces per token.
The traced payload is 4096 elements, or 8 KiB, and NCCL selects `RING_LL`.
The latest graph-node trace attributes 4.176 ms/token of GPU service to these
calls, with a mean of 47.996 us per launch.

The target host has a DGX-1-style hybrid topology. It is not fully connected,
but every rank has NVLink peers and CUDA P2P validation decides whether all
rank pairs are accessible. vLLM's generic custom all-reduce rejects this
topology before running the P2P test.

## Candidate

`VLLM_SM70_TP8_NONFULL_CUSTOM_AR=1` is an explicit, default-off experiment. It
only relaxes the topology gate for SM70, TP8, non-fully-connected groups. The
normal all-pairs P2P test remains mandatory. The existing one-stage custom
kernel and CUDA Graph buffer registration are reused unchanged.

The acceptance gate is the exact 4096-element FP16 graph benchmark:

1. output must equal the sum of ranks;
2. rank-max latency must beat NCCL across repeated runs;
3. projected saving across 87 calls must exceed 0.2 ms/token;
4. a later full-model quality gate must pass before enabling the route.

The experiment remains default-off until the microbenchmark and accumulated
endpoint gate are complete.

## 2026-08-03 Result

The all-pairs custom all-reduce route is rejected on the target host. The
DGX-1-style topology has direct NVLink edges but is not fully connected, and
the mandatory CUDA P2P test fails for cross-`SYS` rank pairs. Relaxing only the
topology check therefore cannot make the existing one-shot kernel valid.

An exact 4,096-element FP16 CUDA Graph screen compared NCCL algorithms:

| algorithm | rank-max median |
|---|---:|
| automatic | 50.998 us |
| Ring + LL | 33.122 us |
| Tree + LL | 21.584 us |
| Ring + LL128 | 53.053 us |
| Tree + LL128 | 40.064 us |
| Ring + Simple | 69.951 us |
| Tree + Simple | 79.440 us |

NCCL 2.27 supports function-specific selection, so the full-model candidate
used `NCCL_ALGO="Ring;allreduce:Tree"` and `NCCL_PROTO=LL`. This preserves Ring
for the model's INT8 AllGather while selecting Tree only for AllReduce.

The endpoint gate did not transfer the isolated result:

| mode | TPOT runs | mean |
|---|---|---:|
| accepted automatic baseline | 25.664 / 25.606 / 25.621 ms | 25.630 ms |
| function-scoped Tree AllReduce | 25.728 / 25.743 / 25.379 ms | 25.616 ms |

The 0.014 ms difference is noise and is below the 0.2 ms acceptance gate.
Random FP16 inputs also produce different Ring and Tree output hashes because
the reduction order changes. Do not enable this route or infer endpoint gain
from the isolated NCCL kernel duration.

Artifacts:

```text
/home/fudanwl/v100-worktrees/runs/dsv4-tp8-small-allreduce-micro-20260803/
/home/fudanwl/v100-worktrees/runs/dsv4-tp8-nccl-matrix-20260803/
/home/fudanwl/v100-worktrees/runs/dsv4-tp8-nccl-random-parity-20260803/
/home/fudanwl/v100-worktrees/runs/
  dsv4-tp8-accumulated-tree-ar-ring-other-i1024-o256-20260803/
```
