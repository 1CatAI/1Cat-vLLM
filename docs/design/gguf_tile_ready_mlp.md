# GGUF tile-ready MLP experiment on SM70

The current GGUF verifier executes gate/up and down as separate GPU graph
nodes. The proposed research kernel publishes each paired gate/up N64 tile
as soon as its FP16 SiLU product is complete. Down loads immutable weight
planes before waiting for the two producer tiles needed by its next K128
activation chunk. All 80 CTAs are admitted cooperatively, so a consumer cannot
prevent an unscheduled producer from running. There is no grid-wide phase
barrier. Generation counters distinguish successive CUDA graph replays.

This is a research extension, not a serving dependency. Its translation unit
mechanically reuses the shipped quantization readers and preserves the
original control kernel. Activation reads within the new producer/consumer
kernel use coherent L2 loads; they cannot use the read-only `__ldg` path on
data that another CTA writes during the same launch. Every producer writer
fences before its CTA publishes completion. FP16 scale reconstruction,
FP32 MMA accumulation, four-way K reduction order, and the paired FP16
rounding points stay unchanged.

## Workload and before ledger

The benchmark uses Qwen3.8-27B GSQ-RCO IQ3_S, Q8_0 DFlash2, TP4, four
V100-SXM2-32GB GPUs with NV2 between every pair, CUDA 12.8, Torch 2.10,
M=8, FP16 activations, FP32 recurrent state, E4M3 target KV, FP16 draft KV,
full decode graphs, maximum length 32768, and up to four sequences.
The fresh trace uses a 1024-token input and 64 output tokens.
Its one-second clock sampler misses the 260 ms decode interval; immediately
bracketing SM samples are 1230–1402 MHz before decode and 1530 MHz after.
The trace therefore has no verified constant SM clock. Memory samples are
877 MHz throughout. The steady C1 clock claim below belongs to the longer
unprofiled A/B runs, not this short trace. Both unprofiled
A/B arms used the same wheel and 16 prompts; steady C1 clocks were 1530 MHz
SM and 877 MHz memory. No separate baseline run was added.

The existing unprofiled control averages are 14.431 ms/round at 1K and
14.804 ms/round at 8K. The rejected projection/collective fusion averaged
14.592 and 14.986 ms respectively. It is not selected by the new worktree.
Small-layer speedups did not establish a model benefit.

The node-profiled rank-0 round averages 16.146 ms. This is a diagnostic
composition and must not be subtracted from the unprofiled round.

| Work | Trace service ms/round | Selected weight bytes/card | Payload floor ms |
| --- | ---: | ---: | ---: |
| Gate/up, plane path | 2.861 | 1,316,951,072 | 1.467 |
| Gate/up, native fallback | 0.136 | 28,897,280 | 0.032 |
| Down | 1.682 | 723,568,334 | 0.806 |
| Qkvz and a/b | 1.524 | 476,364,800 | 0.530 |
| GDN out | 0.896 | 188,743,680 | 0.210 |
| Full-attention q/k/v | 0.512 | 141,117,440 | 0.157 |
| Full-attention output | 0.278 | 61,194,240 | 0.068 |
| **Projections** | **7.888** | **2,936,836,846** | **3.270** |

The target graph envelope is 12.826 ms, including 1.114 ms of inter-kernel
gaps. Its auxiliary kernel service includes 1.304 ms communication/reduction,
1.127 ms GDN state/gating, and 0.609 ms attention. The tail envelope is
3.320 ms, including 0.526 ms of gaps. Tail kernel service includes 0.962 ms
draft GEMM, 0.267 ms draft attention, 0.268 ms target head, 0.274 ms shared
draft head, and 0.156 ms sampling/sorting. Kernel service sums can overlap;
they are not a closed wall-time partition.

The MLP weight payload floor is 2.305 ms versus 4.679 ms trace service.
Thus 2.374 ms is an optimistic *profiled-service* ceiling for this scope,
not a predicted model saving. The prototype needs to beat the original
whole-chain control before any production registration is justified.

The [machine-readable ledger](data/gguf_tile_ready_54633_before_20261008.json)
records raw input hashes, loaded stream provenance, target/draft/head payloads,
state snapshot bytes, collective counts, and unmeasured fields. Payload floors
use 898.048 GB/s theoretical HBM bandwidth at 877 MHz. They do not describe
measured DRAM transactions, include repeated activation requests, or establish
instruction/protocol floors. Those entries remain explicitly incomplete.

## Validation gates

1. Real rank-local weights in layers 6, 24 and 51, official GGUF
   dequantization oracle, unchanged-control bitwise comparison, and changed
   activation graph replays. Illegal accesses fail this gate.
2. Cold-weight whole-MLP graph ABBA on all four ranks, with clocks and complete
   samples. No model run follows a slower chain.
3. A positive prototype must enter the normal source-complete wheel before
   same-wheel model A/B, verifier logits, acceptance and genuinely concurrent
   C4 validation. A workload with four submitted requests but no rounds with
   four active decode sequences is not a saturated C4 speed result.

The 512-thread version passed Compute Sanitizer with zero errors after
removing an unused optional down-output gate input. All four rank-local
shards passed bitwise comparisons for three real layers and 20 changed-input
graph replays. Its whole-chain timing nevertheless regressed:

| Layer / down format | Original graph us | Tile-ready graph us |
| --- | ---: | ---: |
| 6 / IQ3_S | 64.322 | 74.764 |
| 24 / IQ4_XS | 64.282 | 75.141 |
| 51 / Q4_K | 64.305 | 72.430 |

These are four-rank means of each rank's same-run ABBA samples, not model
rounds. Post-measurement clocks were 1522–1530 MHz SM and 877 MHz memory.
The compiler reserved 128 registers/thread for the persistent block, with
one four-byte spill in its Q4_K specialization. Down only uses eight of the
sixteen warps; those inactive warps still reserve registers. This is an
observed resource cost, not yet a measured attribution of every lost cycle.
The [negative record](data/gguf_tile_ready_512_rejected_20261008.json)
preserves all samples. The slower version is not admitted to a model run.

Performance counters on 54633 are currently denied with `ERR_NVGPUCTRPERM`;
passwordless sudo is unavailable. No password or driver setting was changed.
Missing instruction and actual-traffic fields therefore remain incomplete.

No new model benefit or accepted after ledger is available yet. The prior full-MLP
prototype using a global cooperative barrier was slower and is not repeated.
This experiment tests per-tile dependencies and pre-wait weight loads instead.

## Additional structural screens

The compact task ABI uses 160 cooperatively resident 256-thread CTAs: 136
N32 gate/up producer tasks and 80 N64 down tasks. The 24 producer-free CTAs
can start reading down weights immediately. Resource admission requires two
blocks per SM. Registers fall to 114–122 with zero spills, but the chain
still regresses to 82.5/82.0/80.1 us against 63.2/63.4/63.6 us controls for
layers 6/24/51. All four ranks retain bitwise results and changed-input
replay stability. This does not isolate register pressure from readiness
traffic and changed activation sharing. It is rejected.

The original 512-thread task ABI was then tested with `ld.acquire.gpu` flag
reads and `st.release.gpu` publication instead of atomic zero-add polls.
These instructions are supported on Volta. It remains slower: 75.1/75.3/72.5
us against 64.3/64.3/64.2 us controls. Bitwise numerical and graph replay
checks still pass. Eliminating the atomic RMW did not fix the regression,
so atomic polling is not established as its dominant cause. This is also
rejected; none of the three variants is taken into a model run.

The [compact](data/gguf_tile_ready_compact_rejected_20261008.json) and
[acquire](data/gguf_tile_ready_acquire_rejected_20261008.json) records retain
all paired measurements. No gains are assigned to the model and no accepted
after ledger exists. The next structural scope needs a new dependency and
resource analysis; further variations of this two-stage MLP screen are not
planned.

## Related execution designs

[Cohere's task graph](https://github.com/cohere-ai/cohere-megakernel) uses
per-tile readiness rather than a full-layer phase barrier. Its H100-specific
TMA, WGMMA and register repartitioning cannot be assumed on SM70. This
experiment independently tests the dependency idea using existing readers;
it does not import those hardware mechanisms.

[Flux's design](https://github.com/bytedance/flux/blob/main/docs/design.md)
moves communication into tile completion and explains that remote I/O can
insert pipeline bubbles. Its reported improvements are not a prediction for
this M8 workload. The already-rejected projection/collective model A/B is
retained separately, and the current experiment claims no overlapping
communication benefit.
