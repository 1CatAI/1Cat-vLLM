# SM70 TP4 No-MTP Long-Context Decode Ledger

## Scope

This ledger covers Qwen3.6-27B-AWQ no-MTP, single-request TP4 decode on four
V100 SM70 GPUs. The intended route is TurboMind AWQ, Flash-V100, CUDA graphs,
and no eager execution. MTP acceptance metrics do not apply here: every output
token performs one target forward.

## 2026-07-13 Current-Binary 256K Cold Prefill and Decode Sweep

This sweep measures the repaired G6/P7/P2-D16 candidate under one consistent
256K service configuration: Qwen3.6-27B-AWQ, TP4 on V100 `0,1,2,3`,
TurboMind AWQ, Flash-V100, non-eager CUDA graphs, p256, no MTP,
`max_model_len=262144`, `max_num_batched_tokens=1024`, `max_num_seqs=1`, and
prefix caching explicitly disabled. Each request uses official sampling
(`temperature=1.0`, `top_p=0.95`, `top_k=20`) with `ignore_eos` only to hold a
64-token diagnostic window. The 256K row has 262,080 input tokens to reserve
the 64 output tokens within the model limit.

`prefill_s` is scheduler-to-first-token time, so it includes the first-token
forward/sample boundary but excludes model startup and graph capture.
`prefill_tok/s = input_tokens / prefill_s`; `TPOT` and decode throughput cover
the remaining 63 decode intervals. Prefix-cache hit rate is zero throughout.
This is one post-startup sample per length because a single 256K cold prefill
takes about nine minutes on the current 1024-token chunked path; it is not a
p50/p90 study.

| input tokens | prefill_s | prefill tok/s | TTFT_s | TPOT | steady decode |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.478 | 2,143.8 | 0.527 | 13.917 ms | 71.86 tok/s |
| 4,096 | 1.244 | 3,293.2 | 1.246 | 13.350 ms | 74.90 tok/s |
| 16,384 | 6.220 | 2,634.2 | 6.224 | 13.589 ms | 73.59 tok/s |
| 65,536 | 45.093 | 1,453.4 | 45.105 | 15.398 ms | 64.94 tok/s |
| 131,072 | 154.237 | 849.8 | 154.261 | 19.274 ms | 51.88 tok/s |
| 262,080 | 548.697 | 477.6 | 548.742 | 27.804 ms | 35.97 tok/s |

The corresponding XQA active p256-partition counts are `5`, `17`, `65`,
`257`, `513`, and `1024`. From 128K to 256K, cold prefill time grows `3.56x`
for a `2x` input increase, while decode TPOT grows `19.274 -> 27.804 ms`
(`+44.3%`). From 4K to 256K, decode throughput falls `74.90 -> 35.97 tok/s`
(`-52.0%`). This explicitly identifies two independent long-context problems:
the chunked cold-prefill path is severely superlinear, and q=1 decode still
degrades with active partition count.

The later `14.290 ms` entry in this ledger is invalid as a 128K reference: its
artifact name says `i131008`, but its actual prompt has ten token IDs. A fresh
same-binary reproduction with 131,008 real input tokens and
`max_model_len=131072` measures `19.306 ms` / `51.798 tok/s`, statistically
consistent with this sweep's 131,072-token, 256K-capacity result of
`19.274 ms` / `51.883 tok/s`. The max-model-length/workspace difference is
therefore not the source of the earlier apparent `69.98 tok/s` result.

Artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/length_sweep_20260713/g6_p7_p2d16_cold_i1k_256k_o64_single.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/length_sweep_20260713/cold_single_run.log`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/length_sweep_20260713/cases.json`

## 2026-07-13 Current-Binary 64K Prefill Decay Root Cause

This is separate from the q=1 decode NCU result below. It profiles the actual
long-prefill shape for Qwen3.6-27B-AWQ: TP4 on V100 `0,1,2,3`, TurboMind AWQ,
Flash-V100, non-eager CUDA graphs, no MTP, `max_num_batched_tokens=1024`,
fp16 KV, and 784-token hybrid-cache pages. The model has 64 layers but only
16 full-attention layers; each TP rank therefore uses `Hq=6`, `Hkv=1`,
`D=256` for full attention.

### What Grows With Context

At 64K, chunked prefill uses one initial dense chunk and 63 subsequent
1024-token prefix chunks. Every prefix chunk invokes all 16 full-attention
layers, so each TP rank makes exactly `16 * 63 = 1008`
`prefill_prefix_paged` calls. This is confirmed by the four-rank route
summary from the instrumented real-model run.

The event logger records every layer and forces synchronization, so it is used
for attribution only. The accepted cold absolute time remains `45.093 s` from
the sweep above; the event-enabled one-token run is `43.024 s`, close enough
to use for a share calculation but not a replacement baseline.

For each chunk and layer, the critical path below is the maximum CUDA-event
duration across the four TP ranks. The chunk total is the serial sum over the
16 full-attention layers.

| prefix seq_len | 16-layer Flash critical path | mean per full-attention layer |
|---:|---:|---:|
| 2,048 | 19.581 ms | 1.224 ms |
| 4,096 | 41.471 ms | 2.592 ms |
| 8,192 | 85.494 ms | 5.343 ms |
| 16,384 | 174.010 ms | 10.876 ms |
| 32,768 | 356.895 ms | 22.306 ms |
| 49,152 | 541.889 ms | 33.868 ms |
| 65,536 | 751.055 ms | 46.941 ms |

Summing all 63 prefix chunks gives `23.466 s` of TP critical-path
full-attention work, or `54.5%` of the `43.024 s` instrumented prefill. The
last 32K of the request alone costs `17.664 s`, `75.3%` of that
prefix-attention sum; the last 16K costs `10.380 s`, `44.2%`. This is the
source of the superlinear cold-prefill curve: fixed-size chunking repeatedly
attends each new 1024-token query block over an increasingly long KV prefix.
This is exact causal attention's `O(L^2)` work, scheduled as
`O(L / chunk_size)` paged calls. The fixed small query tile prevents the
large dense kernel from amortizing paged-layout and launch/synchronization
costs; it is not a delayed HBM saturation point.

### Exact-Shape Microbenchmark and NCU Attribution

The isolated op uses the real per-rank shape
`M=1024,Hq=6,Hkv=1,D=256,page=784`, with the 16 full-attention layers only.
Its per-layer CUDA-event medians are nearly linear in prefix length:

| M | N | median per layer | effective attention throughput |
|---:|---:|---:|---:|
| 1,024 | 16,384 | 10.979 ms | 9.10 TF/s |
| 1,024 | 32,768 | 22.054 ms | 9.20 TF/s |
| 1,024 | 65,536 | 44.118 ms | 9.27 TF/s |
| 1,024 | 131,072 | 88.195 ms | 9.31 TF/s |

The prefill microbenchmark formerly reported `ctas=192` when the low-SMEM
environment variable was unset, even though the launcher enables that D256
route by default. Its reporting logic now mirrors the launcher and records the
actual `BLOCK_M=16`, grid `(64,1,6)`, and 384 CTAs. This corrects metadata
only; the timing evidence was produced by the same dispatched kernel.

NCU on the `M=1024,N=65536` measured kernel
`flash_attention_forward_kernel_paged<256,...>` shows that the bottleneck is
neither peak HBM bandwidth nor peak Tensor Core compute:

| NCU item | result | interpretation |
|---|---:|---|
| grid / block | `(64,1,6)` / `512` threads | 384 CTAs; not a tiny-grid launch |
| registers / dynamic shared memory | 64 / 36.096 KB per CTA | permits two CTAs per SM |
| theoretical / achieved occupancy | 50.00% / 47.04% | unlike q=1 decode, this prefill kernel is not one CTA per SM |
| DRAM / L1TEX / L2 / SM throughput | 0.48% / 88.60% / 20.73% / 29.33% | HBM is almost idle; pressure is on the on-chip path and issue efficiency |
| L1TEX / L2 hit rate | 33.87% / 99.68% | data is largely served by L2, not HBM |
| scheduler no-eligible cycles | 67.65% | only 0.63 eligible warps per scheduler despite 7.61 active |
| dominant stalls per issued instruction | long scoreboard 7.03; barrier 5.43; short scoreboard 3.13; MIO throttle 2.61 | dependency and synchronization latency is not hidden |
| shared-memory conflict | 4.9-way load, 4.6-way store | 1.037B excessive shared wavefronts, 51% of shared wavefronts |
| global-load sector use | 21.3 / 32 bytes | avoidable L1TEX transaction waste, secondary to the shared layout |

The Nsight `Memory Throughput=81.27%` headline here is driven by L1/TEX; it
must not be read as 81% HBM use. The actual DRAM counter is only `0.48%`.
Likewise, the `2.67` CTA waves/SM leave a partial third wave, but that tail is
secondary to the shared-memory conflict and the long-scoreboard/barrier stalls.

### Layout Experiment Ledger

All layout candidates must use the exact local shape above, preserve two
CTAs/SM, and pass same-process `torch.equal` before timing is considered.

| candidate | correctness | M=1024, N=65536 median | decision |
|---|---|---:|---|
| score-buffer pitch `136 -> 144` floats | invalid pilot | n/a | The candidate template flag was not forwarded into the device `KernelConfig`; both sides used the baseline layout. Do not use its prior timing as evidence. |
| output-accumulator pitch `264 -> 268` floats | `torch.equal`, max diff `0`; 64K model output equal | micro `44.039 -> 40.222 ms` (`-8.67%`) | Accept for D256 low-SMEM page-784 route; +256 B/CTA retains 64 registers and two CTA/SM. |
| `p_strict` two `16x24` half slabs | bitwise equal, max diff `0` | `40.227 -> 40.686 ms` (`+1.14%`) | Reject and remove: candidate wins 1/200 alternating pairs; extra address/layout cost outweighs its shared benefit. |
| contiguous `p_strict` pitch `40 -> 48` half | bitwise equal, max diff `0` | `40.127 -> 40.125 ms` (`-0.006%`) | Reject and remove: candidate wins only 103/200 alternating pairs, so this is noise. A pitch-only change does not move the residual `LDS.128` critical path enough to justify NCU or model work. |
| full current-V-subtile `prefetch.global.L1` | bitwise equal, max diff `0` | `40.173 -> 40.760 ms` (`+1.463%`) | Reject and remove: 128 nonblocking `CCTL.E.PF1` requests per 32-token subtile retain 64 registers but win 0/200 pairs. With V already L2-resident, the extra L1TEX/issue traffic outweighs latency hiding. |
| aligned compact column-major `p_strict` | bitwise equal, max diff `0` | `40.285 -> 41.960 ms` (`+4.157%`) | Reject and remove: candidate wins 0/40 alternating pairs. It preserves 64 registers and fixes the consumer interpretation, but scattered half stores in the row-owned softmax producer cost more than the P-load conflict. |

The accepted `sO` candidate is default-on only for the measured page size
`784` through `VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268`. Set it to
`0` to restore `O_STRIDE=264`; other page sizes stay on the prior layout unless
explicitly enabled.

Its NCU effects are consistent with the CUDA-event speedup: dynamic shared
memory is `36.10 -> 36.35 KB/CTA`, registers remain `64`, and achieved
occupancy is `47.04 -> 47.97%`. Shared load/store conflicts fall
`4.9 -> 4.4-way` and `4.6 -> 3.4-way`; total excessive shared wavefronts fall
`1.037B -> 637M` (`-38.6%`). Scheduler no-eligible cycles fall
`67.65 -> 64.47%`, eligible warps/scheduler rise `0.63 -> 0.71`, and issued
warps/scheduler rise `0.32 -> 0.36`.

Source-correlated NCU identifies the next residual shared target precisely:
four `sP` WMMA-A `LDS.U.128` instructions each contribute `49.988M`
excessive wavefronts, or `31.37%` of the candidate's `637.342M` total. With
`P_STRIDE=40` half, `bank(r,c)=(4 + 20r + floor(c/2)) mod 32`, so rows `r`
and `r+8` alias exactly. A legal WMMA leading dimension must be a multiple of
eight half values, which explains why a `40 -> 48` pitch cannot fix this
topology. The aligned column-major P prototype removed the regular row-major
consumer layout but regressed `4.157%`; future P work must change producer
ownership or use a custom Volta fragment loader, not scatter from the current
one-warp-per-row softmax producer.

The controlled 27B-AWQ TP4 64K model run preserves the prompt hash and output
token `16401`: prefill is `42.811 -> 40.236 s`, a `2.576 s` reduction
(`-6.02%` latency, `+6.40%` throughput). Both sides report exactly 1,008
`prefill_prefix_flash` calls per TP rank. Tensor-level layout A/B is bitwise;
the harness model comparison reports `equal=true`. It remains a model-quality
gate with logprob evidence pending, not a changed attention reduction.

### Volta Software-Pipeline Direction

Update 2026-07-15: the 8-producer/8-consumer shared K/V staging plan below was
implemented and rejected. QK regressed 33.56% and PV regressed 6.31%; reducing
the PV staging pitch still regressed 3.60%. The current accepted micro
candidate is the exact early `sP` store (`-2.596%`), and the next P0 is a
cross-`block_n` warp-specialized QK/PV pipeline that does not add a
global-to-shared round trip. The authoritative experiment table and NCU phase
decomposition are in
`docs/design/sm70_flash_v100_prefill_operator_optimization.md`. Treat the
original numbered plan in this section as historical design context, not an
open implementation instruction.

The accepted-layout NCU report rules out L2 or HBM bandwidth saturation but
does show a load-to-HMMA dependency problem:

| counter | accepted `sO=268` result | implication |
|---|---:|---|
| DRAM / L2 throughput | `0.52% / 22.28%` | neither external memory nor L2 bandwidth is saturated |
| L1TEX throughput / L1 hit rate | `88.12% / 35.16%` | the on-chip load/shared path is busy, but most direct K/V loads still miss L1 |
| tensor-pipe active | `9.95%` | HMMA is severely underfed; this is not peak Tensor Core compute |
| SM throughput | `32.18%` | substantial issue capacity remains idle |
| active / eligible / issued warps per scheduler | `7.71 / 0.71 / 0.36` | nominal occupancy does not hide synchronized dependencies |
| no-eligible cycles | `64.47%` | issue starvation is the dominant utilization failure |
| long scoreboard / barrier / short scoreboard / MIO | `6.997 / 5.117 / 2.643 / 1.433` cycles per issue | direct load dependencies, CTA barriers, and shared serialization all matter |

PC sampling attributes `89.58%` of long-scoreboard samples to HMMA step-0
instructions waiting for operands. Of those HMMA samples, `72.21%` are the PV
`ROW/ROW` path and `27.79%` are QK `ROW/COL`. A cache-only fix is therefore
insufficient: the next candidate must issue future K/V loads from independent
warps while current consumer warps execute HMMA.

SM70 has no `cp.async`, TMA, or asynchronous transaction barrier. The viable
substitute is a manual named-barrier software pipeline, specialized only for
the exact `BM=16, BN=128, D=256, Hq=6, Hkv=1, page=784` route:

1. Divide the 16-warp CTA into eight producer and eight consumer warps.
2. For QK, producers fill two `128 x 16` half K stages (`4 KB` each) while
   consumers compute the current K16 tile. This uses the eight warps that are
   idle in the present eight-tile QK mapping.
3. For PV, reuse the same `8 KB` scratch as two `16 x 128` half V stages.
   Eight consumers process D in two 128-wide halves; producers fill the next
   `(K16,D128)` stage while HMMA consumes the current one.
4. Preserve each output tile's K order, online-softmax order, FP16 probability
   rounding, and FP32 accumulator order. Full-CTA barriers remain at the
   score/softmax and subtile boundaries initially; only the K/V mainloop uses
   named producer/consumer barriers.

The scratch must be a phase-reused union, not simultaneous K and V storage.
`36.35 KB + 8 KB = 44.35 KB/CTA`, so two CTAs still fit under V100's `96 KB`
SM limit. Register count must remain at or below `64`; any spill or increase to
65 registers removes the second CTA and rejects the candidate.

The first micro gate remains `M=1024,N=65536`. Before any model run, the
candidate must be bitwise exact on normal, tail, and reversed block tables,
have no local-memory traffic, preserve two CTAs/SM, and reach at least
`35.0 ms` (`>=12.9%` faster than `40.222 ms`). NCU promotion targets are
tensor-pipe active `>=15%`, long scoreboard `<4.5`, and no-eligible cycles
`<55%`, without increasing excessive shared wavefronts.

At 64K, full attention is `54.5%` of measured prefill. A 20% attention-kernel
speedup maps to only about 10% whole-prefill speedup. Reaching 30% whole-prefill
reduction from this kernel alone would require approximately `2.22x` full-
attention speedup, so the software pipeline is necessary but may not be
sufficient without later work on the remaining GDN/GEMM/prefill components.

### Decision Boundary

The first prefill optimization target is the exact paged Flash-V100
full-attention kernel, not HBM bandwidth tuning and not a generic GEMM change.
The P0 direction is an exact conflict-free shared Q/K/V/probability layout
with fewer barriers and lower dependency depth, while preserving the existing
online-softmax and reduction order. Any double buffer or smaller tile must be
judged against occupancy: increasing shared memory or live registers can
remove the remaining second CTA per SM and regress the kernel. Improving
global-load coalescing is useful, but it is P1 because HBM is not saturated.

Artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_profile_20260713/exact_tp4_fullattn_paged_steps_1k_q6_kv1_d256_page784.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_profile_20260713/model_64k_profiled_i65536_o1.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_profile_20260713/model_64k_profiled_i65536_o1.log`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_profile_20260713/nsys/paged_M1024_N65536_q6_kv1_d256_page784.nsys-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_profile_20260713/ncu/paged_M1024_N65536_q6_kv1_d256_page784_full.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_layout_20260713/output_stride_264_268_forwarded_ab200_m1024_n65536.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_layout_20260713/ncu_output_stride_268_m1024_n65536.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_layout_20260713/model_64k_output_stride_264_268_quality_compare.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_layout_20260713/prob_slab24_ab200_m1024_n65536.json`

## 2026-07-13 Current-Binary 4K-to-64K Per-Token Analysis

This is the current source tree, Qwen3.6-27B-AWQ, TP4 on physical V100
`0,1,2,3`, TurboMind AWQ, Flash-V100, non-eager CUDA graphs, and no
`speculative_config`. Sampling is the official `temperature=1.0`,
`top_p=0.95`, `top_k=20`; `ignore_eos` is used only to hold the diagnostic
output window fixed. The 4K warm request was discarded before collecting the
following low-overhead pure-decode values.

| actual prompt tokens | TPOT | steady decode | delta vs 4K |
|---:|---:|---:|---:|
| 4,104 | 13.342 ms | 74.951 tok/s | reference |
| 8,196 | 13.391 ms | 74.678 tok/s | +0.049 ms |
| 16,391 | 13.474 ms | 74.218 tok/s | +0.132 ms |
| 32,770 | 14.733 ms | 67.875 tok/s | +1.391 ms |
| 65,539 | 17.320 ms | 57.737 tok/s | +3.978 ms |

The curve is flat through 16K and decays after 32K. It is no-MTP target
decode, so neither draft acceptance nor verifier work can explain this loss.

### Nsight Systems Per-Token Critical Path

CUDA-graph node traces at 4K and 64K contain 23 replay groups; the middle 17
tokens are aggregated. Their absolute wall times are tracing-inflated and are
used only for composition. The low-overhead TPOT table above is the speed
authority. The total traced GPU critical path grows `14.509 -> 18.827 ms`, a
`4.318 ms/token` delta, while host slack remains approximately zero.

| category | 4K critical ms/token | 64K critical ms/token | delta | share of 4K-to-64K delta |
|---|---:|---:|---:|---:|
| TurboMind quantized AWQ GEMM | 6.810 | 6.927 | +0.117 | 2.7% |
| Flash-V100 q=1 decode | 1.142 | 5.203 | +4.062 | 94.1% |
| TP communication/all-reduce | 1.886 | 2.012 | +0.126 | 2.9% |
| CUTLASS/cuBLAS FP16 GEMM/GEMV | 1.581 | 1.591 | +0.010 | 0.2% |
| RMSNorm/residual fused Triton | 0.826 | 0.850 | +0.024 | 0.6% |
| Other Torch/Triton elementwise | 0.680 | 0.711 | +0.030 | 0.7% |
| GDN decode / causal convolution | 0.501 | 0.515 | +0.014 | 0.3% |
| SiLU/gating fused Triton | 0.150 | 0.158 | +0.008 | 0.2% |
| KV update, sampling, copies, fill, other | 0.575 | 0.577 | +0.002 | 0.0% |
| **GPU critical-path wall** | **14.509** | **18.827** | **+4.318** | **100.0%** |

The corresponding four-rank GPU-work sum for Flash grows `4.557 -> 20.776
ms/token`; the work is in the Flash partition and reduction kernels, not
hidden behind another rank. Flash launch count remains essentially fixed
(`127.5 -> 128.5` kernels/token): this is not a host launch-count regression.
Each captured partition launch simply has much more live Z-grid CTA work at
64K. The full token rows, p50/p90/p99 values, kernel counts, and top-kernel
lists are retained in:

- `bench_results/nomtp_long_context_20260713/current_tp4/nsys_i4k_o24/per_token.md`
- `bench_results/nomtp_long_context_20260713/current_tp4/nsys_i64k_o24/per_token.md`

### Exact Attention Mechanism

The 64K trace hits
`flash_attention_decode_xqa_tc_partition_kernel_256_wide<256,6>`. On TP4,
each rank has `Hq=6`, `Hkv=1`, `D=256`; the model has 16 full-attention layers
and 48 GDN/linear-attention layers. The q=1 XQA kernel maps the `Hkv=1` head
and each KV partition to a CTA, reads K/V, writes partial softmax/output state,
then runs a separate partition reduction. At the observed p256 specialization,
the live partition count changes from `ceil(4,104/256)=17` to
`ceil(65,539/256)=257` per full-attention layer and rank. This is the required
exact KV work; reducing it through a stale active-partition value would omit
context and is a correctness bug.

The p256 CUDA-graph trace cited in this section was collected with an explicit
p256 route and is operator-candidate evidence, not proof of the current
unpinned long-context default. The runtime selects p1024 when
`max_seq_len_hint >= 32768` unless
`VLLM_FLASH_V100_DECODE_PARTITION_SIZE=256` is set. A CUDA graph preserves its
captured specialization; runtime `active_num_partitions` changes how many
valid partitions are processed, not the template's reduction boundary. Every
future full-model result must record the explicit partition-size environment
instead of inferring it from source or an older graph trace.

### Nsight Compute: Why p256 Does Not Reach V100 Peak Bandwidth

The following NCU result profiles one real current-shape p256 partition
kernel on V100 at `L=65,539`; the accompanying CUDA-event microbenchmark
measures partition plus reduction. It is a kernel attribution, not a whole
model speed claim.

| item | result |
|---|---:|
| partition kernel grid / block | `(1, 1, 257)` / `256` threads |
| NCU partition duration | 281.18 us |
| CUDA-event partition + reduce | 0.4263 ms |
| registers / dynamic shared memory | 186 / 41.22 KB per CTA |
| theoretical / achieved occupancy | 12.50% / 12.47% |
| DRAM / memory / SM throughput | 22.73% / 30.10% / 20.18% of peak |
| effective memory throughput / L1TEX hit / L2 hit | 255.17 GB/s / 6.41% / 8.29% |
| active / eligible warps per scheduler | 2.00 / 0.27 |
| cycles with no eligible warp | 77.19% |
| L1TEX long-scoreboard stall | 2.8 cycles, 32.4% of issue interval |

Thus the long-context cost is not a saturated-HBM ceiling. It is a low
occupancy, memory-latency and synchronization-hiding problem inside a large
shared-memory XQA CTA: 186 registers and 41.22 KB shared memory permit only
one CTA per SM. The direct p1024 diagnostic reduces this synthetic
partition-plus-reduce time from `0.4263` to `0.3500 ms` at 64K, but changes the
softmax partition/reduction order. Earlier long-output evidence has shown such
partition changes can alter sampled tokens, so this is a quality-gated
candidate, not a default performance result.

NCU artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/ncu/xqa_p256_l65539.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/ncu/xqa_p256_l65539_details.csv`
- `bench_results/nomtp_long_context_20260713/current_tp4/ncu/xqa_p256_l65539_warp_details.csv`
- `bench_results/nomtp_long_context_20260713/current_tp4/ncu/xqa64k_cuda_event.jsonl`

### Resulting Optimization Boundary

The immediate P0 is the exact long-q=1 XQA partition plus reduction path. A
valid successor must improve memory-latency hiding or reduce partial-state and
reduction traffic while preserving the p256 numerical/reduction contract. AWQ
GEMM, MTP acceptance, sampler, and host scheduling cannot recover the observed
4K-to-64K loss. A partition-size change is a separate candidate and requires
full no-MTP long-output quality and token/logprob evidence before it can be
enabled.

### 2026-07-13 P0 Candidate: Exact Padded WMMA Shared Layout

The detailed source-counter report changes the priority within the exact p256
path. The partition kernel has real local-memory traffic (`12,336` local loads
and `8,224` local stores) and, more importantly, `2,642,976` excessive shared
wavefronts: `37%` of all `7,229,273` shared wavefronts. The four high-volume
source locations are the Volta WMMA A/B fragment loads immediately before the
`HMMA.884` sequence. They show 16-way and 32-way conflicts respectively:

| shared operand | current WMMA leading dimension | observed conflict | candidate leading dimension |
|---|---:|---:|---:|
| Q, row-major A | 256 fp16 values | 16-way | 264 fp16 values |
| K, column-major B view | 128 fp16 values | 32-way | 136 fp16 values |

`264` and `136` preserve the required 16-byte alignment while moving each
successive shared row/column away from the same Volta bank set. The candidate
keeps p256, all KV loads, the two 128-token online-softmax tiles, fp16
probability storage, and the existing cross-partition reduction unchanged.
It therefore changes shared addresses only, not attention arithmetic or the
reduction order.

The candidate is accepted as the default only for the measured Qwen3.6-27B
TP4 shape: fp16 KV, `D=256`, `q_per_kv=6`, and p256. Set
`VLLM_FLASH_V100_XQA_PADDED_SMEM=0` to restore the dense shared layout. G4/G8
and p512/p1024 retain the prior layout because they have not passed this gate.
The reusable exact microbenchmark is:

```bash
PYTHONPATH=flash-attention-v100 \
VLLM_FLASH_V100_DECODE_PARTITION_SIZE=256 \
python benchmarks/kernels/benchmark_sm70_flash_v100_xqa_layout.py \
  --seq-len 65539 --partition-size 256 --layout compare
```

All gates passed on 2026-07-13:

| exact Hq=6/Hkv=1/D=256 microbenchmark | p256 baseline | padded layout | result |
|---|---:|---:|---:|
| 4,097 tokens, partition + reduce | 0.08850 ms | 0.08547 ms | 1.035x |
| 65,539 tokens, partition + reduce | 0.34078 ms | 0.29438 ms | 1.158x |
| 262,144 tokens, partition + reduce | 1.20156 ms | 1.11657 ms | 1.076x |

All three direct outputs are bitwise equal (`max_abs_diff=0`). The 64K NCU
partition-kernel duration falls `280.03 -> 255.87 us`; source counters show
`2,642,976 -> 279,072` excessive shared wavefronts (`37% -> 6%` of total,
`-89.4%`). The high-volume Q/K WMMA loads fall from 16/32-way conflicts to
the remaining 8-way K-side loads. Registers fall `186 -> 184`; dynamic shared
memory rises `41.22 -> 43.26 KB`, so occupancy remains one CTA per SM
(`12.5%`) and the existing local spill remains a separate problem.

The same full-model no-MTP TP4 CUDA-graph run replays the historical six
official-sampling prompts exactly (`temperature=1.0`, `top_p=0.95`,
`top_k=20`, output 64). All six output token sequences are equal to the
baseline. The steady 64K point improves `17.320 -> 16.993 ms/token`, or
`57.737 -> 58.848 tok/s` (`+1.92%`). The default was enabled only after this
model-level token comparison. The comparison has no logprob/sampler-logit
dumps, so its broad model-quality gate remains `B-pending`; that missing
evidence does not weaken the scoped layout assertion because the direct
operator output is bitwise equal and the full-model tokens are identical.

Artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/ncu/xqa_p256_l65539_padded.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/padded_layout_4k_64k_o64.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/padded_layout_4k_64k_o64_compare.json`

Do not combine this experiment with p512/p1024, a changed active partition
count, a 192-thread G6 kernel, FP8 KV, or a new reduction order. Those are
separate experiments and would make a quality or speed result non-attributable.

### Deferred Follow-Up: G6 Dual-CTA Redesign

If the padded layout removes conflicts but the kernel remains limited by
registers and one resident CTA, the next candidate is a separate G6-specific
kernel. It may use six warps, retain the eight Q rows required by the 8x32
WMMA tile, request the V100 96 KB shared-memory carveout, and target two CTAs
per SM. It is not a free configuration change: with 192 threads, two resident
CTAs require no more than about 170 registers per thread and no more than
49.15 KB dynamic shared memory per CTA. The current 186-register kernel
already spills, so launch-bounds forcing alone is rejected. First reduce live
fragment/accumulator pressure and prove that local traffic falls rather than
merely moving the spill into shared memory.

## Current Short-Context Composition

The current TP4 no-MTP graph-node trace at `i512/o16` is the composition
authority for fixed per-token work. The low-overhead endpoint records
`70.730 tok/s` and `14.138 ms` TPOT; graph-node tracing reports `15.173 ms`
GPU wall and is not used as the absolute throughput result.

| category | critical-path ms/token | short-context wall share |
|---|---:|---:|
| TurboMind quantized GEMM | 8.197 | 54.0% |
| TP communication/all-reduce | 1.860 | 12.3% |
| CUTLASS/cuBLAS FP16 GEMM/GEMV | 1.561 | 10.3% |
| Flash-V100 q=1 decode | 1.058 | 7.0% |
| RMSNorm/residual | 0.799 | 5.3% |
| GDN decode / causal convolution | 0.447 | 2.9% |

The Flash kernel observed in the trace is
`flash_attention_decode_xqa_tc_partition_kernel_256_wide`. Although scalar
paged decode remains enabled as a fallback, the default eligible q=1 request
selects XQA. This is distinct from the MTP M=5 small-query verifier route.

Artifact: `bench_results/tp_scaling_nomtp_20260712/nsys_tp4_i512_o16/`.

## Correct Long-Context Shape

The active-partition-correct TP4 no-MTP sweep is historical route evidence,
not the acceptance baseline for the current July binary. It is still the best
available verified shape of the exact q=1 long-context cost: the runtime
active partition count tracks `ceil(context / 256)`, rather than using a stale
graph-capture value.

| input context | TPOT | steady decode | delta vs 4K TPOT |
|---:|---:|---:|---:|
| 4K | 14.371 ms | 69.587 tok/s | reference |
| 16K | 14.523 ms | 68.856 tok/s | +0.152 ms |
| 32K | 15.965 ms | 62.638 tok/s | +1.594 ms |
| 64K | 18.368 ms | 54.442 tok/s | +3.997 ms |
| 128K | 23.185 ms | 43.131 tok/s | +8.814 ms |
| 256K | 32.544 ms | 30.728 tok/s | +18.173 ms |

There is no draft acceptance term in this table. The `4K -> 64K` loss is
therefore entirely target decode, and almost all of its approximately `4 ms`
TPOT growth is context-dependent KV work in the 16 full-attention layers. The
64-layer model has 16 full-attention and 48 GDN/linear-attention layers; AWQ
GEMM, row-parallel reductions, norms, GDN, and sampling remain largely fixed
per emitted token. At a 256-token partition size the full-attention scan grows
from 16 active KV partitions at 4K to 257 at 64K, 513 at 128K, and 1024 at
256K.

The old TP2 `16K/32K/64K` curve that reported a flat `58.46 tok/s` is
explicitly invalid as long-context speed evidence. It lacked active-partition
route proof and matches the stale-active failure mode that can omit KV
partitions. Do not use it to claim that no-MTP long decode is flat.

Artifacts:

- `bench_results/sm70_migration_20260622/decode_latest_qwen36_27b_awq_flashv100_xqa_adaptive_tp4_i4k_16k_32k_64k_128k_256k_o64_20260622.json`
- `bench_results/sm70_migration_20260621/decode_qwen36_27b_awq_flashv100_xqa_trace_active_i16k_64k_o16_tp2_20260621.json`
- `docs/design/sm70_v100_migration_control.md` dynamic-partition correction

## Decision And Next Measurement

The no-MTP long-context P0 is q=1 Flash-V100 XQA KV scanning and its
cross-partition reduction, not MTP draft sampling, verifier M=5, or AWQ GEMM
tile selection. The MTP API was stopped for the current no-MTP profile so all
four GPUs were available; do not collect this lane concurrently with a service

The current-binary sweep and 4K/64K node traces are now complete. Future
repeats must retain the same model, 256K model length, TurboMind AWQ,
Flash-V100, CUDA graphs, official `temperature=1.0`, `top_p=0.95`, `top_k=20`
sampling, and fixed-output diagnostic policy. Any partition-size or XQA route
candidate must pass the same no-MTP long-output quality and token/logprob gate
before it replaces the default.

## 2026-07-13 P1/P2 Exact Partition Work

P1 is the G6-specific padded p256 partition kernel with 192 threads and a
two-CTA launch bound. At 256K it reduces the partition's register/shared
footprint enough to admit two CTAs per SM. P2 leaves that partition arithmetic
unchanged and replaces only the serial cross-partition output reduction with a
two-stage exact reduction: the original 256-thread max/sum tree materializes
the original fp16 weights, then 16 independent output-dimension tiles consume
partitions in their original ascending order. It is controlled by
`VLLM_FLASH_V100_XQA_SPLIT_REDUCE=1` and remains experimental/default-off
until repeated endpoint evidence is available.

The current-shape CUDA-event result is real but must not be extrapolated
directly to the model: at 262,144 tokens, P1 partition plus reduction is
`0.94694 ms`; P2 is `0.78510 ms` (`-17.09%`), with `max_abs_diff=0`.
NCU attributes the old P1 reducer to a `(1,6,1)` grid, `337.6 us`, and only
`0.032` eligible warps per scheduler. P2's stats stage costs `6.11 us`; its
96-CTA output stage costs `169.34 us`. P2 therefore removes a genuine reducer
hole, but the P1 partition remains the larger 256K component (`702.75 us`,
`64.79%` cycles with no eligible warp, `397.24 GB/s` DRAM).

The first purported whole-model P1/P2 gate was **invalid for P1/P2
attribution**. Its JSON environment does not contain
`VLLM_FLASH_V100_DECODE_PARTITION_SIZE=256`. At `max_seq_len_hint=262144`, the
runtime's documented default is p1024; both P1 and P2 explicitly require p256
and were inactive. The two otherwise identical p1024 runs reported:

| actual route | steady TPOT | steady decode | interpretation |
|---|---:|---:|---|
| p1024 default, P1/P2 inactive | 29.2652 ms | 34.170 tok/s | reference measurement |
| p1024 default, P1/P2 inactive | 28.6604 ms | 34.891 tok/s | normal same-route variation, not P2 gain |

Their 64 output IDs are equal, but that only confirms the unchanged p1024
route. Do not use the apparent `2.07%` difference as P2 endpoint evidence.
That mandatory p256 route gate is now complete. With the exact same model,
TP4, CUDA-graph, TurboMind AWQ, 262,080-token prompt, official sampling, and
64-token diagnostic, explicitly setting
`VLLM_FLASH_V100_DECODE_PARTITION_SIZE=256` plus G6/P2 yields:

| actual route | steady TPOT | steady decode | p1024-relative result | token gate |
|---|---:|---:|---:|---|
| p1024 default | 29.2652 ms | 34.170 tok/s | reference | reference |
| p256 + G6 dual CTA + P2 D-tile 16 | 25.8120 ms | 38.742 tok/s | `-11.80%` TPOT / `+13.38%` tok/s | all 64 IDs equal |

The p256 operator output is not bitwise equal to p1024 (`max_abs_diff` is
`7.62939453125e-06` in the direct comparison), because p256 changes the
cross-partition reduction boundary. The fixed-output model gate nevertheless
passes exactly. This is accepted route and token evidence, but the broad
model-quality gate remains pending until sampler logits/logprobs or a broader
natural-output set is collected.

The `3.4533 ms` endpoint saving is a combined p1024-to-p256, G6, and P2
result; it must not be labeled as P2-only. Against the matching p1024
baseline, a 30% TPOT-reduction objective is `20.486 ms/token`, leaving
`5.326 ms/token` beyond the current p256 result. The next implementation
target is therefore the p256 partition itself, not another reducer tile sweep
or an unpinned partition-size comparison.

Artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/g6_p1_i262080_o64.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/g6_p2_splitreduce16_i262080_o64.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/g6_p2_splitreduce16_i262080_o64_compare.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/p256_g6_p2_splitreduce16_i262080_o64.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/p1024_vs_p256_g6_p2_i262080_o64_compare.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/ncu/xqa_p256_l262144_g6_192.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/ncu/xqa_p256_l262144_split_reduce16_stats.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/ncu/xqa_p256_l262144_split_reduce16_output.ncu-rep`

## Rejected: P3 G6 KV64 Three-CTA Variant

P3 reduced only the K/V shared tile from 128 to 64 rows while retaining the
full 128-token score/probability tile and the original softmax boundary. It
was intentionally gated by `VLLM_FLASH_V100_XQA_G6_KV64=1`, default-off. The
intent was three resident 192-thread CTAs, not a changed p256 reduction
contract.

At 256K NCU confirms that the residency target was reached: `96`
registers/thread, `25.86 KB` dynamic shared memory, and `27.71%` achieved
occupancy (three CTAs/SM). It is nevertheless rejected. Splitting the work
introduced local-store spill traffic and an uncoalesced local-store pattern;
the partition reaches only `216.30 GB/s` DRAM, has `0.28` eligible warps per
scheduler, and spends `81.22%` of cycles with no eligible warp. Its partition
duration is `1.46 ms`, versus P1's `702.75 us`. The 64K whole-operator check
is also slower: `0.53081 ms` with the original reducer versus `0.33067 ms`
for the raw baseline, although it is bitwise exact in that configuration.
Combining P3 with P2 additionally produced a nonzero output difference
(`max_abs_diff=3.814697265625e-05`), so it fails the exact gate as well.

Do not enable, retune, or retest this KV64 variant as a simple occupancy
candidate. A future high-residency design must preserve the full QK warp
parallelism and avoid compiler local spills; merely halving K/V shared storage
creates more synchronization and less useful per-CTA work than it hides.
Artifact:
`bench_results/nomtp_long_context_20260713/current_tp4/ncu/xqa_p256_l262144_g6_kv64.ncu-rep`.

## P4 Decision: Stage the p256 Partition Without Changing Its Arithmetic

The accepted p256+G6+P2 endpoint result makes the next budget measurable. The
p1024-to-p256 direct operator difference is `1.02157 -> 0.78791 ms`
(`0.23365 ms`); across the model's 16 full-attention layers that predicts
`3.738 ms`, close to the measured `3.453 ms` TPOT reduction. The direct
operator is therefore an adequate first gate for the next architecture.

For a matching-config 30% TPOT-reduction objective, p1024's `29.2652 ms`
requires `20.4857 ms`. From the accepted p256 endpoint this leaves
`5.3263 ms`, or approximately `0.3329 ms` per full-attention layer. P4 must
bring p256 partition plus reduction from `0.7879 ms` to **at most
`0.4550 ms`** before a new 256K model run is justified.

The candidate is a staged exact XQA implementation, not a partition-size,
precision, or KV-layout change:

1. QK plus the existing full 128-token softmax retains WMMA, fp32 logits,
   fp16 probability rounding, p256 boundaries, and final partition max/sum.
   It writes the two 128-token fp16 probability tiles to the existing
   `tmp_out[B,H,P,256]` storage before that storage is reused.
2. A small fp32 workspace records the original online-softmax rescale between
   the two 128-token tiles for each `(B,H,P)` row.
3. PV runs separately, reads probabilities in token order, applies that same
   rescale before the second tile, and overwrites `tmp_out` with the original
   partition partial output. P2 then runs unchanged.

This breaks the compiler lifetime coupling between WMMA fragments and the
eight fp32 PV accumulators. QK/softmax can retain its 128-row K tile; PV can
use a 64-row V tile without changing any FMA order, targeting a higher
resident-CTA count without the P3 score/PV spill pattern.

P4 acceptance sequence:

| gate | required evidence |
|---|---|
| direct correctness | p256 raw/P4 `torch.equal`, `max_abs_diff=0` at 4K, 64K, 256K, including tail lengths around 128/256 boundaries |
| resource proof | NCU per stage: no local spill regression; QK/softmax and PV report separate registers, shared memory, occupancy, eligible warps, and HBM throughput |
| micro performance | partition plus P2 reduction `<=0.4550 ms` at 256K; report five independent samples rather than a single timing |
| model quality | same 256K no-MTP TP4 official-sampling output IDs equal to the accepted p256 result and p1024 baseline; broaden with natural-output/logit evidence before defaulting |
| model performance | pure steady decode at or below `20.486 ms/token`; prefill/TTFT reported separately |

If the staged QK/PV split cannot reach the direct gate, reject it before any
additional full-model prefill run. The next lower-level fallback is a further
score-only / softmax / PV separation, again preserving the same probability
rounding and token-order FMA contract.

## P4/P5/P6/P7/P8 Route Ledger

This section supersedes the P4 proposal above with measured outcomes. It is
the authority for avoiding repeated long-context experiments. All iterative
operator measurements below use the resolved Qwen3.6-27B-AWQ TP4 full-attention
shape (`Hq=6`, `Hkv=1`, `D=256`), fp16 KV, p256, G6 dual-CTA, P1 padded shared
layout, and P2 split reduction unless stated otherwise.

| path | result | decision |
|---|---|---|
| P4 staged QK/softmax/PV | Direct 128K exact, but `0.4004 -> 0.4921 ms` (`+23%` slower). The QK stage remains 168 registers / 43.26 KB shared / 18.65% occupancy. | Rejected. Do not retune the staged split without a new resource argument. |
| P5 one-stage KV prefetch | Direct 128K exact, but `0.400256 -> 0.400336 ms`. NCU resource and scheduler counters did not move. | Rejected. Do not retry the same prefetch lifetime arrangement. |
| P6 block16 index specialization | Direct 16-token-page test improves about `0.3985 -> 0.3743 ms`, but the live Qwen hybrid cache is not 16-token paged. | Not an endpoint result; keep mode opt-in only. |
| P7 native block784 index specialization | The Q1 repair restores repeated direct exactness and preserves the P7 partition resource/timing profile. | Direct-operator quality accepted; fresh 128K model quality/performance evidence is still required. |
| C1 block784 `ld.global.cg` K/V load | Sequential 128K is bitwise exact and about 3% faster, but reversed 131,071-token paging changes output (`max_abs_diff=7.629e-06`). | Rejected before NCU or model execution; experimental dispatch removed. |
| P8 transposed QK `m32n8k16` | Sequential 128K is bitwise exact and `0.38634 -> 0.37504 ms` (`-2.93%`), but reverse 131,071-token paging changes output (`max_abs_diff=7.629e-06`). | Rejected before NCU or model execution; experimental dispatch and harness toggle removed. |
| P9 register-resident PV accumulator | Removes the 32-byte local stack and local load/store instructions, but the reverse-page comparison is not quality-safe because the underlying G6 baseline itself is nondeterministic. | Rejected and removed. Do not attribute its small direct timing gain to a valid endpoint. |

### Why P6 Did Not Reach the Model

The previous P6 microbenchmark used a `block_size=16` cache, but the real
Qwen3.6 hybrid allocator aligns full-attention pages to its GDN/Mamba state.
The live TP4 worker log reports:

```text
Setting attention block size to 784 tokens to ensure that attention page size
is >= mamba page size.
FLASH_ATTN_V100 decode active trace: route=decode_xqa_paged ... page_size=784
```

Thus P6's 16-token bit-shift path never dispatched during the prior 128K model
run. This is a route mismatch, not an endpoint-noise explanation. The relevant
allocator is `vllm/platforms/interface.py`; its hybrid page alignment must be
recorded in every future Qwen3.6 long-context experiment.

`VLLM_FLASH_V100_KERNEL_BLOCK_SIZE16=1` is a separate virtual-split path: it
keeps a 784-token manager page but exposes 49 16-token attention blocks. It
does not reduce KV bytes and expands the attention block table roughly 49x.
It has no accepted Qwen3.6 no-MTP decode or quality result, so it is deferred
rather than used as a shortcut to claim P6 endpoint speed.

### P7: Native 784-Token Page Index Specialization

The generic XQA kernel recomputed runtime division/remainder for each KV vector
load. P7 instantiates the existing p256 G6 kernel with `BLOCK_SIZE=784` and
replaces only page-index arithmetic with compile-time constant division and
subtraction. It still uses live K/V strides and the original block table; it
does not assume contiguous physical pages, change K/V layout, precision,
softmax boundaries, probability rounding, or P2 reduction order.

The guard is deliberately narrow: P7 is selected only when the current route
already has G6 dual CTA, `k_cache.size(1)==784`, p256 and the existing padded
G6 route. `VLLM_FLASH_V100_XQA_BLOCK784_INDEX=0` restores the generic kernel.

### Q1 Stability Correction: G6/P1/P7 Is Not Yet Quality-Accepted

The original P7 direct table established output equality across one launch per
layout. That was insufficient to prove deterministic behavior. A later
repeated self-comparison uses the same `131,071`-token reverse physical page
table and tail length, invokes the same kernel twice in one fresh process, and
compares the two final outputs. With the original G6 192-thread p256 path and
P2 disabled, **6 of 10** clean-source P1 self-comparisons differ by
`max_abs_diff=7.62939453125e-06`. The result persists with P2 enabled and
with/without the block784 index specialization. P7 is therefore not the root;
the issue is in the shared G6/P1 partition implementation.

The 256-thread padded route with G6 disabled passed five equivalent reverse
self-comparisons. The leading root-cause candidate is the G6 WMMA use of
shared Q/K pitches `264/136` fp16 elements (`528/272 B`): both are `16 mod 32`
bytes and do not provide the 32-byte fragment alignment required for a
bitwise-stability proof. The thread-count/code-generation change in G6 can
therefore expose a data-dependent Volta WMMA rounding/undefined-layout effect.
This is an inference, not yet an isolated raw-logit proof.

Decision: do not present the prior P7 model result as quality-accepted or use
G6/P1/P7 in a quality-sensitive API until an aligned-layout repair passes the
repeated reverse/self and stable-reference gates. The historical timing result
remains useful only as a performance reference. The next candidate is an
aligned padded pitch (`Q=272`, `K/V=144` fp16 elements) that preserves p256,
the original m8n32 QK, softmax, probability rounding, PV FMA order, and two
resident CTAs.

Direct CUDA-event operator gates at about 128K:

| logical KV mapping | length | generic P1 + P2 | P7 | result |
|---|---:|---:|---:|---|
| sequential | 131,072 | 0.40925 ms | 0.37471 ms | `-8.44%`, bitwise equal |
| reverse | 131,071 | 0.41619 ms | 0.37630 ms | `-9.59%`, bitwise equal |
| permuted, tail | 131,075 | 0.39850 ms | 0.37678 ms | `-5.45%`, bitwise equal |

NCU on the same 131,072-token sequential shape attributes the gain correctly:

| partition-kernel metric | generic block784 | P7 block784 | change |
|---|---:|---:|---:|
| duration | 362.59 us | 336.03 us | `-7.33%` |
| executed instructions | 42.55 M | 38.39 M | `-9.77%` |
| DRAM throughput | 387.59 GB/s | 418.28 GB/s | `+7.92%` |
| registers / dynamic shared | 168 / 43.26 KB | 168 / 43.26 KB | unchanged |
| achieved occupancy | 18.65% | 18.66% | unchanged |
| eligible warps / scheduler | 0.53 | 0.54 | unchanged within noise |

The remaining bottleneck is therefore still memory-latency hiding, not page
address arithmetic: P7 removes instruction overhead but does not create more
resident CTAs. The next architecture must reduce the `43.26 KB` shared-memory
footprint or introduce useful global-load ILP without reintroducing P3's local
spill pattern.

The historical 128K endpoint timing used the current source binary, 4xV100 TP4,
TurboMind-AWQ, Flash-V100, non-eager CUDA graphs, p256/G6/P1/P2,
`input_len=131,008`, 64 fixed official-sampling tokens (`temperature=1.0`,
`top_p=0.95`, `top_k=20`, seed `20260620`, `ignore_eos`). All 64 output IDs are
equal and worker logs show both `decode_xqa_paged page_size=784` and the P7
specialization on all four ranks.

| 128K pure decode | TPOT | steady decode | token gate |
|---|---:|---:|---|
| current generic block784 | 19.36346 ms | 51.64367 tok/s | reference |
| P7 native block784 | 18.92318 ms | 52.84523 tok/s | 64/64 IDs equal |
| result | `-0.44027 ms`, `-2.27%` | `+2.33%` | historical token match only |

The direct sequential microbenchmark predicts at most
`16 * (0.40925 - 0.37471) = 0.5527 ms/token` across the 16 full-attention
layers. The observed `0.4403 ms/token` realizes about 80% of that isolated
operator budget, so P7 is confirmed to propagate through the full graph rather
than being a microbenchmark-only result.

P7 changes only decode. The observed prefill/TTFT variation in this pair is
not attributed to P7 and is excluded from the comparison. The Q1 correction
above supersedes its earlier quality-promotion wording.

### Rejected C1: Volta `ld.global.cg` K/V Cache Policy

SASS confirms the accepted P7 K/V vector reads use the Volta read-only path
(`LDG.E.128.CONSTANT.SYS` from `__ldg`). Since q=1 long-context K/V is a
single-pass stream with low L1/L2 reuse, C1 tested `__ldcg(uint4)` only in the
P7 helper. The sequential 128K direct run improved `0.38889 -> 0.37739 ms`,
but the reverse block-table/tail gate at 131,071 tokens was not bitwise equal
(`max_abs_diff=7.62939453125e-06`).

This is a hard quality rejection: a cache-policy hint must not change the
attention result. Do not retest `cg`, `ca`, `cs`, or an equivalent cache-hint
change as a speed-only candidate unless the visibility/numerical discrepancy
is first explained and eliminated. No NCU or full-model time was spent after
the direct exactness failure.

### Rejected P8: Transposed `m32n8k16` QK Operand Layout

P8 kept p256/G6/P1/P2, the native 784-token index path, fp16 K/V, softmax,
probability fp16 rounding, scalar PV `fmaf`, and the P2 reduction unchanged.
It changed only the QK Tensor Core expression from
`Q[8,256] * K^T[256,32]` (`m8n32k16`) to its transpose
`K[32,256] * Q^T[256,8]` (`m32n8k16`). The padded shared layout allowed a
direct col-major store back to the existing `sS[query_head][token]` layout, so
the candidate had no explicit transpose or extra shared allocation.

This is a numerical hard stop, not a performance rejection. The sequential
128K page table was bitwise equal and improved `0.38634 -> 0.37504 ms`
(`-2.93%`). The same A/B with a reverse physical page table and a 131,071-token
tail improved `0.38862 -> 0.37598 ms` (`-3.25%`) but was not bitwise equal:
`max_abs_diff=7.62939453125e-06`.

The two expressions are algebraically equivalent, but Volta does not promise
that different WMMA shapes have the same internal floating-point association or
rounding. There is a second exactness risk: this padded route uses Q/K leading
dimensions `264/136` halfs (`528/272 B`), which are only `16 mod 32` bytes and
therefore do not provide a clean WMMA fragment-alignment proof. The accepted
`m8n32` route already uses those strides, but the `m32n8` mapping can expose a
different sensitivity to that contract. This experiment did not export the
pre-softmax fp32 `sS` scores, so it does not isolate which of those two causes
dominates; it does prove that an algebraic transpose is not an exact-preserving
implementation substitution on this route.

Do not retry this `m8n32 <-> m32n8` operand transposition, a store-layout-only
variant, or an equivalent WMMA-shape swap under the strict `torch.equal` gate
without first proving bitwise accumulator equivalence on arbitrary K/Q data
with an aligned layout and exported fp32 logits.

No NCU, full-model, or 256K test was run after the reverse-layout failure.
Artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/p8_qk_transpose/p8_qk_transpose_i131072.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/p8_qk_transpose/p8_qk_transpose_reverse_i131071.json`

### Rejected P9: Register-Resident PV Accumulator

P9 scalarized the eight fp32 PV accumulators so the compiler emitted no local
stack (`STACK:32 -> 0`) and no local load/store instructions
(`24,576/12,288 -> 0/0` in the 128K NCU pair), while retaining `168`
registers/thread, `43.26 KB` dynamic shared memory, and two CTAs/SM. One-shot
direct timings were positive (`~1-4%`), but this is not promotable: reverse
page-table comparisons intermittently differ by `7.62939453125e-06`, and the
Q1 audit proved the unchanged G6 baseline can do the same.

P9 does not isolate or repair that root defect. Its dispatch and harness flags
were removed before any model run. Do not revive it as a speed candidate until
the G6 layout has a deterministic, quality-accepted baseline.

Artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/p7_block784_index_i131072.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/p7_block784_index_reverse_i131071.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/p7_block784_index_permuted_i131075.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/ncu/p7_block784_p1_i131072.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/ncu/p7_block784_index_i131072.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/block784_index/p7_base_current_i131008_o64.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/block784_index/p7_block784_index_i131008_o64.json`

### Q1/P2 Correctness Repair: Separate Score/Probability Storage

This correction supersedes the earlier Q1 hypothesis that blamed the padded
WMMA pitch. Compute Sanitizer racecheck found a real same-CTA shared-memory
race in the monolithic G6 partition kernel: `reuse_sp` overlaid fp32 scores
`sS[8][128]` with fp16 probabilities `sP[8][128]`. One softmax warp could
begin writing `sP` while another was still reading aliased `sS`. The
131,071-token reverse-page case is a strong trigger because its final p256
partition has a 127-token second tile, but the defect is independent of the
padded/dense shared pitch and of the block784 index specialization.

The repair makes score and probability storage distinct, adding 2,048 bytes of
dynamic shared memory without changing WMMA, softmax, fp16 probability rounding,
or PV FMA order. The separate P2 defect was also corrected: its output grid was
tiled by `D_TILE` while every CTA launched 32 threads, so adjacent CTAs wrote
the same output dimensions for D-tile 8 or 16. The output CTA now launches
exactly `D_TILE` threads, preserving the dimension-to-thread arithmetic while
removing redundant writes.

All results below use the resolved Qwen TP4 full-attention shape, p256, G6,
padded shared storage, block784 index, `seq_len=131071`, a reverse physical
page table, P2 D-tile 16 where enabled, and ten fresh-process comparisons:

| comparison | result | p50 partition + reduce |
|---|---|---:|
| repaired P7 versus itself, P2 off | 10/10 `torch.equal`, max error 0 | 0.44123 ms |
| 256-thread non-G6 reference versus repaired P7, P2 off | 10/10 `torch.equal`, max error 0 | 0.57286 -> 0.44109 ms |
| repaired P7 safe reducer versus fixed P2 D16 | 10/10 `torch.equal`, max error 0 | 0.44089 -> 0.37682 ms |

Racecheck reports zero hazards for repaired P7 with P2 off and with P2 D16 on.
NCU on repaired P7 records 192 threads, 168 registers/thread, 45.312 KB dynamic
shared memory, and two resident CTAs/SM under both register and shared-memory
limits. Achieved active warps are 18.52%, partition duration is 336.352 us,
DRAM throughput is 36.95% of peak, SM throughput is 30.53% of peak, and
eligible warps per scheduler remain 0.532. The correctness repair therefore
preserves the intended dual-CTA resource profile rather than replacing it with
a slower safety fallback.

Artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/ncu/g6_p7_separate_sp_i131071.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/racecheck_g6_padded_p7_i131071.log`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/racecheck_g6_p7_p2_d16_i131071.log`

Historical P1/P2/P7 full-model token matches were produced before these races
were repaired. They remain useful timing context only and must not promote this
new binary's quality. The next gate is a fresh 128K no-MTP TP4 CUDA-graph
comparison with the repaired P7/P2 route, official sampling, and explicit
route proof. Reserve a 256K endpoint run until that 128K gate passes.

### Invalidated "128K" Full-Model Gate: Short-Prompt Mislabel

The two artifacts below are named `i131008`, but they are not 128K tests.
Their `prompt_generation.input_len` is null and the recorded request contains
only ten prompt token IDs (`Write a concise explanation of why deterministic
validation matters.`). Their route summaries contain only
`prefill_no_prefix_dense_flash=16`; a real 131K request has
`prefill_prefix_flash=2032`. Consequently, the former `14.568 -> 14.290 ms`
and `68.64 -> 69.98 tok/s` table is short-prompt timing only. It must not be
cited as a long-context G6/P2 end-to-end gain, a 128K quality gate, or a
microbenchmark-to-model mapping.

A fresh, same-binary, true-long-context reproduction uses 131,008 actual
prompt token IDs, 64 forced official-sampling output tokens, TP4, TurboMind
AWQ, Flash-V100, non-eager CUDA graphs, p256, G6/P7/P2-D16,
`max_model_len=131072`, `max_num_batched_tokens=1024`, and prefix caching
disabled. The worker log proves `active=512` p256 partitions and the expected
2,032 chunked-prefill Flash calls.

| true 128K current candidate | prefill | TPOT | steady decode |
|---|---:|---:|---:|
| 131,008 real prompt tokens, maxlen 131,072 | 127.883 s | 19.306 ms | 51.798 tok/s |
| 131,072 real prompt tokens, maxlen 262,144 sweep | 154.237 s | 19.274 ms | 51.883 tok/s |

The close decode values reject graph-workspace capacity as the explanation for
the former 69.98 result. No valid post-race full-model T256-versus-G6/P2
long-context A/B exists yet; it must be rerun with actual prompt-token count
checked before any end-to-end speed claim. The isolated 128K-tail operator
microbenchmark `0.57286 -> 0.37682 ms` (`-34.22%`, 1.52x) remains valid, but
its former 1.91% full-model mapping is invalidated with the mislabeled test.

Artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/model_128k/t256_safe_p256_i131008_o64.json` (invalid as 128K)
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/model_128k/g6_p7_p2d16_i131008_o64.json` (invalid as 128K)
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/repro_128k_20260713/max131072_modeltokens_i131008_o64.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/repro_128k_20260713/max131072_modeltokens_i131008_o64.log`
