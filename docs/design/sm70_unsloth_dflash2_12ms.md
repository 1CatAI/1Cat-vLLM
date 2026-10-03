# SM70 mixed NVFP4 DFlash2 complete-round latency

## Acceptance contract

The target is a complete single-request speculative round below 12 ms on
four V100-SXM2-32GB GPUs: target verification, rejection sampling, and DFlash2
draft forward with seven speculative tokens. Freeze a new baseline on the
measurement host before accepting performance deltas. The 21.1 ms observation
from another host is context only.

Use the unsloth Qwen3.8-27B mixed checkpoint: layers 0–55 have NVFP4 MLPs;
GDN QKV/Z/output projections, attention projections, layers 56–63 MLPs, and
the LM head have channel FP8 weights; GDN A/B projections are BF16.
Checkpoint identity must be verified before substituting any existing model.

Launch with maximum model length 262144. Measure 1024- and 8192-token inputs
with thinking disabled and temperature 0.7. Recover the remaining sampling,
output-length, KV, graph, and prompt settings from the original baseline
scripts before freezing the contract. Report complete-round wall time and
emitted tokens per round separately from TTFT, prefill, and GPU kernel service
sums. Use unprofiled runs for acceptance and short Torch profiler captures for
attribution. Preserve acceptance and concurrent-request performance.

Numerical changes require mean KL <= 0.001, p99 KL <= 0.01, maximum KL <= 0.05,
top-1 agreement >= 99%, and maximum logit difference <= 0.5. Sampling changes
must preserve the distribution, including ties, top-p boundaries, and
presence/frequency penalties. Draft or LM-head precision reductions require
separate approval. Accepted routes must select automatically by capability;
new per-optimization environment switches are outside this scope.

## Initial evidence, 2026-10-03

Integration base: `8002bc10709c25db8dc576e4c96ab3e614d6131e`.

All six physical GPU pairs report NV2. A standalone SM-store benchmark tests
all twelve directed pairs with an 81920-byte payload and a system fence,
using 8, 20, and 40 blocks of 256 threads. Each sample times a CUDA Graph with
100 store kernels; three warmup replays precede twenty timed replays. All
payload checks pass. Mean per-node times range from 7.139 to 8.591 us across
pairs and grids. CUDA 12.8 compiled the benchmark for SM70. This is a peer-store
diagnostic, not collective latency, a receiver handshake benchmark, or a
model speedup. Percentiles describe graph-average samples rather than
individual stores.

Reproduce under exclusive ownership of the selected GPUs:

```bash
nvcc -O3 -arch=sm_70 benchmarks/csrc/sm70_peer_store_latency.cu -o peer_store
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 ./peer_store
```

The measurements support trying the existing fully connected push collective
first. No slow edge requiring a butterfly was observed on this host. At this
base, `CustomAllreduce.sm70_tp4_all_reduce_gemma_rms_norm` is explicitly
benchmark-only; its existence does not prove model dispatch.

The specified target and draft are not present at the requested destinations.
Source-host authentication is pending, so the matching model baseline and
Torch profile have not run. No numerical or runtime routing change has been
made, and no model performance result is claimed.

## Next evidence required

### Hardware and communication follow-up

The measurement host has one Xeon E5-2680 v4 socket (14 cores/28 threads),
one NUMA node, 64 GB nominal system RAM, and four V100-SXM2-32GB GPUs. Software
is Ubuntu 24.04.2, NVIDIA driver 580.173.02, CUDA toolkit 12.8.93, Python
3.12.14, Torch 2.10.0+cu128, NCCL 2.27.5, Triton 3.6.0, and Transformers
5.17.0. Each GPU has six active NVLink links, reported at 25.781 GB/s each;
each peer pair has two links. ECC is enabled.

The original 185 W GPU power caps were explicitly raised to the 300 W hardware
default before the following tests. Preserve 300 W for the forthcoming model
baseline. GPU ownership locks covered both tests, and their processes exited.

| Model-free test, 81920 bytes per call | Mean for 140 calls | Amortized call |
| --- | ---: | ---: |
| NCCL, four processes, captured graph | 4.340 ms | 31.003 us |
| Existing push kernel, 80 blocks, captured graph | 0.913 ms | 6.524 us |
| Existing push kernel, 40 blocks, captured graph | 0.909 ms | 6.495 us |

The push probe directly instantiates the unchanged kernel from this source
base using one host process and four CUDA devices. It validates every output
element on every rank while changing the input markers between graph replays.
It does not validate multi-process IPC setup or model dispatch. The NCCL probe
uses separate processes, checks eager and captured sums, and reports the
critical rank's event interval. These different harnesses are diagnostic
evidence; their difference is not an accepted end-to-end speedup. A single
collective per graph is dominated by rank launch skew and is not used to infer
communication limits.

The approximately 1.6 us needed to serialize 81920 bytes at the reported
two-link rate is only an ideal wire-time floor. Synchronization, protocol,
reduction, and kernel execution are additional costs. The measured push
chain establishes a practical sub-10-us candidate on this topology; it does
not establish the absolute communication lower bound.

1. Recover the original model identities and baseline scripts, copy missing
   artifacts, and pin the complete runtime contract.
2. Establish the unprofiled baseline and short Torch profile on this host.
3. Measure the production push collective and fused norm at the actual
   verifier shapes; prioritize according to the new profile.
4. Validate each retained change against the numerical and acceptance gates,
   then validate the complete installed artifact before promotion.
