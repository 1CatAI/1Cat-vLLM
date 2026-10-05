# GGUF DFlash2 projection ledger after native pair integration

This capture freezes main `7f1d25f4d3917d05060444226bff5e5aed559538`.
Its executable code matches `5620850f8119a744d0c7d3d737c40f1255dde9a5`;
the intervening change records the unprofiled measurements in
[the target/draft design](gguf_dflash2_iq3s.md). One capture follows native
a/b and 48-layer gate/up integration. No operator experiment in this
document changes the captured model routes.

## Contract and attribution

Qwen3.8-27B GSQ-RCO IQ3_S target, Qwen3.8-27B DFlash2 Q8_0 draft,
TP4 on four NVLink-connected V100-SXM2-32GB GPUs; CUDA 12.8,
Torch 2.10.0+cu128, Python 3.12.14, normal `dev41+g5620850f81` package.
Input is exactly 1024 tokens, output budget 64, maximum length 32768,
FP16 activation/KV, FP32 SSM and MMA accumulation, seven draft tokens,
temperature 0.7, top-p 0.9, top-k 20, seed 123, thinking disabled.
The four-GPU leases are held throughout capture. Power is 300W;
every timed capture sample records 1290MHz SM and 877MHz memory.

Each round starts at the first GPU node linked to the target graph launch
and ends at the next target graph's first GPU node. Host NVTX duration is
not used as GPU wall time. Dropping graph transitions leaves 13 rounds
per rank. Matrix-call order is matched to loaded projection metadata;
source format signatures and graph call counts are checked. Physical
buffer inventory is restricted to the target, since draft modules repeat
some target module names.

The table uses the sum of kernel service times, divided by call count.
Weight bytes are loaded code/stat or native-record footprints per rank.
Effective GB/s is bytes divided by service time; it excludes activation,
workspace, codebook and repeated cache traffic. It is not an NCU DRAM
bandwidth measurement. Service sums can overlap; only interval unions
are used to close the wall-time table.

## GPU round boundaries

| Rank | Complete round ms | Target graph ms | After target ms |
| --- | ---: | ---: | ---: |
| 0 | 24.093242 | 20.189812 | 3.903430 |
| 1 | 24.124648 | 20.226500 | 3.898148 |
| 2 | 24.111678 | 20.215653 | 3.896025 |
| 3 | 24.108949 | 20.207177 | 3.901772 |

| Rank0 component | Mean ms | p50 ms | p90 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| Complete GPU round | 24.093242 | 24.138832 | 24.168226 | 24.187577 |
| Target graph envelope | 20.189812 | 20.217105 | 20.258756 | 20.261266 |
| Target kernel interval union | 18.540024 | 18.532123 | 18.619366 | 18.639779 |
| Target gaps between kernels | 1.649788 | 1.667377 | 1.715133 | 1.731336 |
| After target envelope | 3.903430 | 3.912255 | 3.926379 | 3.933914 |
| After target kernel interval union | 3.274126 | 3.280485 | 3.294879 | 3.301542 |
| After target gaps | 0.629304 | 0.630523 | 0.638718 | 0.646322 |

These are Nsight Systems diagnostics, including instrumentation and one
short request. They are distinct from the 16-prompt unprofiled results:
22.219657ms at 1K and 23.316529ms at 8K, and cannot be substituted for
a new end-to-end latency result.

## Target projection service

| Projection | Calls/round | us/call | ms/round | Mean bytes/call/rank | Effective GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| qkvz | 83 | 35.966 | 2.985148 | 6259478 | 174.0 |
| a_b | 96 | 4.334 | 0.416030 | 122880 | 28.4 |
| gdn_out | 48 | 25.279 | 1.213393 | 4218880 | 166.9 |
| gate_up.native | 48 | 61.410 | 2.947670 | 18764373 | 305.6 |
| down | 64 | 39.730 | 2.542723 | 11859200 | 298.5 |
| attention.q | 15 | 37.044 | 0.555656 | 7929856 | 214.1 |
| attention.k+v | 6 | 28.473 | 0.170839 | 1556480 | 54.7 |
| attention.o | 16 | 22.167 | 0.354667 | 4177920 | 188.5 |
| gate_up.canonical | 16 | 63.875 | 1.022000 | 22978560 | 359.7 |
| attention.k | 9 | 32.649 | 0.293839 | 728178 | 22.3 |
| attention.v | 10 | 31.270 | 0.312701 | 720896 | 23.1 |
| attention.q+k | 1 | 37.942 | 0.037942 | 9584640 | 252.6 |

Q includes the attention gate. The 16 attention layers produce 41
physical q/k/v launches: fifteen q, one q+k, nine k, ten v and six k+v.
Merged launches are retained rather than assigning arbitrary shares of
one kernel's service to separate projections. Mixed GDN inputs produce
83 quantized launches across 48 layers, plus 96 dense a/b launches.
The 48 native gate/up calls include eight IQ3_S pairs and forty mixed
pairs; their average is not the latency of the pure IQ3_S kernel.

## Target auxiliary service

| Category | Calls/round | us/call | ms/round |
| --- | ---: | ---: | ---: |
| communication_or_reduction | 130 | 11.275 | 1.465747 |
| rms_norm | 129 | 6.057 | 0.781352 |
| layout_or_copy | 143 | 4.561 | 0.652270 |
| GDN_state_or_gating | 144 | 8.408 | 1.210778 |
| other_target | 128 | 3.946 | 0.505106 |
| attention | 32 | 32.065 | 1.026069 |
| unfused_FFN_epilogue | 16 | 3.633 | 0.058121 |

## Work after the target graph

The tail contains 210 kernels per round. Target logits live in the rejection
graph; draft candidates live in the proposal path and read the shared
target-owned head. The two U4 head GEMMs are identified by graph linkage
and execution order. Python method NVTX wrappers do not emit during
replay of previously captured rejection/proposal graphs.

| Category | Calls/round | us/call | ms/round |
| --- | ---: | ---: | ---: |
| other_tail | 139 | 5.797 | 0.805743 |
| draft_GEMM | 23 | 43.254 | 0.994842 |
| communication_or_reduction | 18 | 13.154 | 0.236773 |
| target_head | 1 | 270.351 | 0.270351 |
| sampling_and_sorting | 23 | 7.346 | 0.168959 |
| draft_attention | 5 | 104.837 | 0.524183 |
| draft_shared_head | 1 | 273.531 | 0.273531 |

Each head reads 198656000 canonical bytes per rank. The target and draft
head calls therefore give effective weight rates of 734.8 and 726.3GB/s.
Their inputs are data dependent; the trace does not establish a safe
way to merge the two calls.

The draft cache actually uses FP16, D128, eight query heads and two KV
heads per rank, page size 832, non-causal attention and a 2048-token
sliding window. The existing grouped verifier requires D256, six query
heads, one KV head and no window. Enabling its model gate alone cannot
serve these draft shapes; a D128 window-aware implementation or split-KV
path needs separate numerical and context-length checks.

## Graph-boundary idle time

Rank0 graph-boundary intervals contain 367.413us of pure idle time per
round, after removing any kernels running between graph envelopes.
The last draft-attention graph ends 219.720us before the next target graph,
but this includes later head and sampling kernels. After the last tail
kernel, only 11.857us remains before the next target begins.
These intervals are subsets of the wall-time gaps, not additional costs.

## Projection counterpart measurements

Pending the same-machine operator sweep. The NVFP4-labelled checkpoint
stores FFN weights in NVFP4 and attention/GDN weights in channel FP8;
the comparison must name those actual formats. Cold-L2 operator timings
will remain separate from traced model service.

## Next implementation boundary

Prioritize down, then joint qkvz/a/b, then GDN output and attention.
Admit only measured type/shape/M winners; other M retain canonical
outputs bitwise. Raw Q2_K/Q4_K reader regressions switch to reading
canonical streams on the small-M skeleton. Pure IQ3_XXS and IQ4_XS
pairs use the existing pair readers after their own shape comparison.
Keep FP32 accumulation and the existing scale precision.

The next full-model run and trace follow a merged projection batch.
Operator deltas times layer counts are estimates until that run closes
the emitted-token and round-time measurements.
