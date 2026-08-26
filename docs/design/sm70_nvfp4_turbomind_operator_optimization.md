# SM70 TurboMind Operator Optimization

Date: 2026-06-30

This is the shared maintenance document for SM70/V100 TurboMind operator
optimization work. It currently tracks two active routes:

- NVFP4 decode GEMM reduction for Qwen3.5-27B-NVFP4 TP2.
- AWQ uint4 verifier GEMM reduction for Qwen3.6-27B-AWQ DDTree TP2.

The file name still contains `nvfp4` for compatibility with existing local
links; do not create a second operator ledger for the AWQ verifier route unless
this document is renamed in one coordinated patch.

## NVFP4 Stage 1 Objective

Reduce the Qwen3.5-27B-NVFP4 TP2 TurboMind decode critical-path bucket:

```text
TurboMind NVFP4 GEMM: 12.877 ms/token -> < 10.000 ms/token
```

This is a CUDA-graph, non-eager, TurboMind-only target. Marlin, eager fallback,
and route-hit-only smokes do not count toward this objective.

The accepted stage-1 result must satisfy both:

- Nsight Systems graph-node breakdown shows `TurboMind NVFP4 GEMM`
  `critical-path mean <= 10.000 ms` on the same profiling method used below.
- A low-overhead benchmark on the same model, GPUs, TP size, input/output
  shape, sampling, and graph policy shows real TPOT improvement. The graph-node
  trace is composition evidence, not the accepted absolute speed baseline.

## Current Evidence

Artifacts:

- `bench_results/nsys_27b_nvfp4_turbomind_20260630/graph_baseline_i512_o64.json`
- `bench_results/nsys_27b_nvfp4_turbomind_20260630/graph_baseline_i512_o64_decode_event_trace.json`
- `bench_results/nsys_27b_nvfp4_turbomind_20260630/graph_node_generate_i512_o16_trace.nsys-rep`
- `bench_results/nsys_27b_nvfp4_turbomind_20260630/graph_node_generate_i512_o16_trace.sqlite`
- `bench_results/nsys_27b_nvfp4_turbomind_20260630/per_token_latency_breakdown_graph_node_i512_o16.md`
- `bench_results/nsys_27b_nvfp4_turbomind_20260630/per_token_latency_breakdown_graph_node_i512_o16.csv`
- `bench_results/nsys_27b_nvfp4_turbomind_20260630/ncu_tmfp4_final_skip512_full_i512_o64.ncu-rep`

Baseline:

- Low-overhead 64-token TPOT: `19.986 ms/token`.
- Host-side decode trace showed
  `AsyncGPUModelRunnerOutput.async_copy_ready_event.synchronize`
  mean `18.807 ms`; graph-node trace confirms this is GPU completion wait, not
  a simple copy.

Graph-node breakdown, middle steady tokens:

| bucket | critical-path mean | TP2 GPU-sum mean | share |
|---|---:|---:|---:|
| TurboMind NVFP4 GEMM | `12.877 ms` | `25.717 ms` | `63.5%` |
| cuBLAS/CUTLASS fp16 GEMM/GEMV | `2.293 ms` | `4.579 ms` | `11.3%` |
| TP communication/allreduce | `1.292 ms` | `2.561 ms` | `6.4%` |
| FlashAttn V100 decode | `1.075 ms` | `2.146 ms` | `5.3%` |
| RMSNorm/residual fused Triton | `0.788 ms` | `1.543 ms` | `3.9%` |
| GDN decode / causal conv | `0.529 ms` | `1.056 ms` | `2.6%` |

The TurboMind NVFP4 bucket is not one large GEMM. It is many small decode GEMMs:

- About `510` TurboMind NVFP4 GEMM kernel launches per token across TP2.
- About `255` launches per token per GPU.
- Average TurboMind FP4 GEMM duration is about `50 us`.

Representative NCU evidence for one TurboMind NVFP4 GEMM:

- Duration: `61.344 us`.
- Registers/thread: `93`.
- Dynamic shared memory/block: `12.320 KiB`.
- Memory throughput: `66.34%`.
- DRAM throughput: `39.77%`.
- SM throughput: `40.72%`.
- Achieved occupancy: `27.43%`.
- Main stalls: short scoreboard, long scoreboard, barrier, wait.
- Source counters reported excessive shared wavefronts, so shared/L1 layout is
  part of the bottleneck.

Interpretation: stage 1 is not a host/scheduler problem. The main issue is a
high launch count of M=1-ish NVFP4 decode GEMMs whose per-kernel efficiency is
limited by occupancy, shared/L1 access, scoreboard stalls, and synchronization
overhead.

## Measurement Contract

Use the Codex skill:

```text
~/.codex/skills/vllm-token-latency-breakdown
```

Required run shape unless explicitly superseded:

- model:
  `/home/ymzx/.cache/huggingface/hub/models--apolo13x--Qwen3.5-27B-NVFP4/snapshots/f48cfc832696cccd195ac305ff7d2ffaeeab4d28`
- `CUDA_VISIBLE_DEVICES=2,3`
- `tensor_parallel_size=2`
- `input_len=512`
- formal low-overhead baseline: `output_len=64`
- graph-node composition trace: short `output_len=16` is acceptable
- `VLLM_SM70_QUANT_BACKEND=turbomind`
- `VLLM_SM70_NVFP4_TURBOMIND=1`
- `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1`
- `VLLM_SM70_LM_HEAD_TOP1=0`
- no eager

Every result must record:

- Exact env and command.
- Low-overhead TPOT and pure decode timing.
- Graph-node category table from the skill parser.
- Whether token/output quality matched the current accepted route.
- If a candidate is slower or hangs, the exact failure mode and artifact path.

## NVFP4 Microbench Baseline

Use the microbench harness for operator-only candidate ranking:

```text
benchmarks/benchmark_sm70_nvfp4_gemm_micro.py
```

Default suite models one Qwen3.5-27B-NVFP4 TP2 decode token on one rank. It
uses synthetic unpacked NVFP4 weights with the same per-rank shapes as the real
model, calls `nvfp4_sm70_prepare` once, warms dispatch, then times only
`nvfp4_gemm_sm70_out`.

Default per-rank shape suite:

| projection group | calls/token/GPU | M | N | K |
|---|---:|---:|---:|---:|
| linear-attention `in_proj_qkvz` | 48 | 1 | 8192 | 5120 |
| all attention/GDN `out_proj` | 64 | 1 | 5120 | 3072 |
| full-attention `qkv_proj` | 16 | 1 | 7168 | 5120 |
| MLP `gate_up_proj` | 64 | 1 | 17408 | 5120 |
| MLP `down_proj` | 64 | 1 | 5120 | 8704 |

Current artifacts:

- `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_qwen35_27b_tp2_m1_5shape.json`
- `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_qwen35_27b_tp2_m1_5shape_cudagraph.json`
- `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_trace_5shape_specs.log`

Current 5-shape non-graph CUDA-event wall, `warmup=30`, `iters=200`:

| projection group | mean/op | weighted ms/token/GPU |
|---|---:|---:|
| linear-attention `in_proj_qkvz` | `69.05 us` | `3.315 ms` |
| all attention/GDN `out_proj` | `41.09 us` | `2.630 ms` |
| full-attention `qkv_proj` | `57.02 us` | `0.912 ms` |
| MLP `gate_up_proj` | `100.49 us` | `6.431 ms` |
| MLP `down_proj` | `67.55 us` | `4.323 ms` |
| total | - | `17.611 ms` |

Current single-op CUDA-graph replay wall:

| projection group | mean/op | weighted ms/token/GPU |
|---|---:|---:|
| linear-attention `in_proj_qkvz` | `93.11 us` | `4.469 ms` |
| all attention/GDN `out_proj` | `30.12 us` | `1.928 ms` |
| full-attention `qkv_proj` | `47.65 us` | `0.762 ms` |
| MLP `gate_up_proj` | `90.58 us` | `5.797 ms` |
| MLP `down_proj` | `57.48 us` | `3.679 ms` |
| total | - | `16.635 ms` |

Interpretation:

- The microbench event wall is not numerically identical to the Nsight
  graph-node `12.877 ms` category, because the Nsight bucket is kernel-duration
  sum while CUDA event wall can include intra-op idle/enqueue gaps. Use the
  microbench for relative candidate ranking and shape priority, then confirm
  accepted changes with the graph-node parser.
- MLP dominates the microbench. `gate_up_proj + down_proj` is roughly
  `10.75 ms` in the non-graph wall metric and `9.48 ms` in single-op graph
  replay, so the stage-1 route must materially improve MLP shapes.
- All observed NVFP4 shapes currently use the same `8x128x64` mgroup SM70
  s884 kernel family with `12.320 KiB` dynamic shared memory/block.
- Selector/split forcing is not enough. Best observed forced split tests moved
  `gate_up_proj` by about `0.1 ms/token` and `down_proj` by about
  `0.1-0.2 ms/token`; the stage target needs about `2.9 ms` on the Nsight
  kernel-sum metric.

### 2026-06-30 Raw M=1 GEMV Prototype

Purpose: test the hypothesis that the `M=1` decode path is losing mainly to
the current TurboMind `CTA_M=8` HMMA padding, and that a direct batch-1 GEMV
can materially beat the generic GEMM path on the two dominant MLP shapes.

Prototype:

- C++/CUDA op: `nvfp4_gemv_sm70_raw_out`.
- Python wrapper: `vllm/_sm70_ops.py`.
- Microbench mode:
  `benchmarks/benchmark_sm70_nvfp4_gemm_micro.py --mode raw-gemv`.
- Scope: default-off microbench path only. It uses a raw row-major packed
  `[K, N/8]` uint4 layout, not the TurboMind prepared layout.
- Kernel shape: one thread handles one packed qword, i.e. 8 output columns;
  split-K writes float partials and a second kernel reduces to fp16 output.
- Correctness sanity: small `1x32x64` reference comparison matched the Python
  dequantized fp32 reference within fp16 output error
  (`max_abs=0.0625`, `max_rel=4.4e-4`).

Non-graph MLP-only ranking, `warmup=20-30`, `iters=100-200`:

| mode | split-K | gate/up mean | gate/up weighted | down mean | down weighted | total weighted | vs GEMM |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboMind GEMM | 0 | `102.231 us` | `6.543 ms` | `67.784 us` | `4.338 ms` | `10.881 ms` | `1.00x` |
| raw GEMV | 4 | `685.425 us` | `43.867 ms` | `1114.071 us` | `71.301 ms` | `115.168 ms` | `10.58x` slower |
| raw GEMV | 8 | `361.851 us` | `23.158 ms` | `549.038 us` | `35.138 ms` | `58.297 ms` | `5.36x` slower |
| raw GEMV | 16 | `222.126 us` | `14.216 ms` | `303.514 us` | `19.425 ms` | `33.641 ms` | `3.09x` slower |
| raw GEMV | 32 | `168.520 us` | `10.785 ms` | `186.839 us` | `11.958 ms` | `22.743 ms` | `2.09x` slower |
| raw GEMV | 64 | `207.217 us` | `13.262 ms` | `163.932 us` | `10.492 ms` | `23.754 ms` | `2.18x` slower |

CUDA-graph MLP-only ranking:

| mode | split-K | gate/up mean | gate/up weighted | down mean | down weighted | total weighted | vs graph GEMM |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboMind GEMM | 0 | `129.741 us` | `8.303 ms` | `55.941 us` | `3.580 ms` | `11.884 ms` | `1.00x` |
| raw GEMV | 16 | `215.050 us` | `13.763 ms` | `288.451 us` | `18.461 ms` | `32.224 ms` | `2.71x` slower |
| raw GEMV | 32 | `158.976 us` | `10.174 ms` | `169.078 us` | `10.821 ms` | `20.995 ms` | `1.77x` slower |
| raw GEMV | 64 | `178.842 us` | `11.446 ms` | `125.870 us` | `8.056 ms` | `19.502 ms` | `1.64x` slower |
| raw GEMV | 128 | `233.615 us` | `14.951 ms` | `130.739 us` | `8.367 ms` | `23.319 ms` | `1.96x` slower |
| raw GEMV | 256 | `323.512 us` | `20.705 ms` | `183.163 us` | `11.722 ms` | `32.427 ms` | `2.73x` slower |

Artifacts:

- `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_mlp_gemm_after_rawgemv_base.json`
- `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_mlp_gemm_after_rawgemv_graph_base.json`
- `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_mlp_rawgemv_split{4,8,16,32,64}.json`
- `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_mlp_rawgemv_split{16,32,64,128,256}_graph.json`

Decision:

- Reject the raw scalar CUDA-core GEMV implementation as a stage-1 route.
  It is correct enough for benchmarking, but the best CUDA-graph point is
  `19.502 ms`, still `1.64x` slower than the same-shape graph GEMM baseline.
- Do not spend time wiring this raw layout into production decode.
- This result does **not** reject all M=1 specialization. It rejects the
  simple CUDA-core per-qword GEMV structure. The next M=1 route must keep
  Tensor Core throughput or use a more aggressive persistent/Stream-K
  scheduling design.

### 2026-06-30 GEMV Follow-Up Variants

Purpose: give GEMV a fairer microbench attempt before abandoning the
CUDA-core branch. Two additional default-off variants were added:

- `raw-gemv-warp`: one warp computes one packed qword, 32 lanes cooperate over
  K, then reduce with shuffle. This removes the split-K partial matrix.
- `raw-gemv-h2`: keeps the split-K shape but uses CUDA-core `half2` FMA and
  fp16 partials, reducing scalar float work and partial bandwidth.

Representative MLP-only results:

| mode | graph | split-K | gate/up mean | gate/up weighted | down mean | down weighted | total weighted | vs graph GEMM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TurboMind GEMM | false | 0 | `102.231 us` | `6.543 ms` | `67.784 us` | `4.338 ms` | `10.881 ms` | - |
| TurboMind GEMM | true | 0 | `129.741 us` | `8.303 ms` | `55.941 us` | `3.580 ms` | `11.884 ms` | `1.00x` |
| raw GEMV warp | false | 0 | `246.001 us` | `15.744 ms` | `316.805 us` | `20.276 ms` | `36.020 ms` | - |
| raw GEMV h2 | false | 64 | `172.268 us` | `11.025 ms` | `144.435 us` | `9.244 ms` | `20.269 ms` | - |
| raw GEMV h2 | false | 128 | `173.804 us` | `11.123 ms` | `118.856 us` | `7.607 ms` | `18.730 ms` | - |
| raw GEMV h2 | true | 64 | `164.091 us` | `10.502 ms` | `129.137 us` | `8.265 ms` | `18.767 ms` | `1.58x` slower |
| raw GEMV h2 | true | 128 | `153.897 us` | `9.849 ms` | `97.787 us` | `6.258 ms` | `16.108 ms` | `1.36x` slower |
| raw GEMV h2 | true | 256 | `175.293 us` | `11.219 ms` | `107.392 us` | `6.873 ms` | `18.092 ms` | `1.52x` slower |

Accuracy notes:

- `raw-gemv-warp` matches the float raw reference at the same fp16 output error
  level as the original scalar GEMV.
- `raw-gemv-h2` is faster than scalar raw GEMV but changes accumulation
  semantics. On full MLP shapes at split-K 128, `down` differed from float raw
  by mean relative error `0.27%`; `gate/up` had many near-zero denominators and
  showed mean absolute error `0.54` with mean relative error `51%`. It is a
  speed probe, not a production-quality candidate.

Decision:

- Reject pure CUDA-core GEMV as the stage-1 production route. The best h2 graph
  result is `16.108 ms`, an improvement over raw float GEMV but still much
  slower than TurboMind GEMM's `11.884 ms` on the same MLP-only graph metric.
- Do not continue with more scalar/half2 GEMV micro-tuning unless there is a
  new structural change that removes the main bottleneck. The evidence now
  says V100 still needs Tensor Core throughput for these MLP shapes.
- Carry forward only the useful lesson: split-K around `128` helps the down
  projection in CUDA-core GEMV, but the gate/up projection remains too slow and
  dominates the miss to the stage target.

## External Research Scan

Date: 2026-06-30. Purpose: identify non-selector innovation routes that can
move the NVFP4 GEMM bucket from `12.877 ms` to `<10 ms`.

Relevant sources:

- FlashDecoding++:
  <https://arxiv.org/html/2311.01282v4>
- Stream-K:
  <https://arxiv.org/abs/2301.03598>
- QUICK:
  <https://arxiv.org/abs/2402.10076> and
  <https://github.com/squeezebits/quick>
- FastGEMV:
  <https://github.com/wangsiping97/FastGEMV>
- Quill fused quantized GEMV:
  <https://github.com/Aminsed/quill>
- Batch-1 decode memory/launch analysis:
  <https://arxiv.org/html/2605.30571v1>
- Low-latency megakernel / persistent decode:
  <https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>
  and <https://arxiv.org/abs/2512.22219>
- QServe/QoQ:
  <https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf>
  and <https://github.com/mit-han-lab/omniserve>
- Fast NF4 dequantization:
  <https://arxiv.org/html/2604.02556v1>
- CodeGEMM:
  <https://proceedings.neurips.cc/paper_files/paper/2025/file/3151e460c41ba67dc55412861184ef35-Paper-Conference.pdf>
- tinygemm / any4:
  <https://arxiv.org/html/2507.04610v1> and
  <https://github.com/facebookresearch/any4>
- GemLite:
  <https://github.com/dropbox/gemlite> and
  <https://dropbox.github.io/gemlite_blogpost/>
- LUT-GEMM:
  <https://arxiv.org/html/2206.09557v4> and
  <https://github.com/naver-aics/lut-gemm>
- FlashInfer GEMM / grouped GEMM:
  <https://github.com/flashinfer-ai/flashinfer> and
  <https://docs.flashinfer.ai/api/gemm.html>
- DeepGEMM:
  <https://github.com/deepseek-ai/DeepGEMM>
- Megakernel / persistent decode direction:
  <https://arxiv.org/html/2605.11581v1>

Single-request decode fit:

- QUICK is **not** the primary stage-1 model for our case. Its published gains
  are mainly at larger batches, where mixed-precision GEMM becomes
  compute/dequant/shared-memory-conflict limited. Our current acceptance target
  is one-token decode with `M=1`, so the more direct references are
  FastGEMV/FlashDecoding++/GemLite-style quantized GEMV and low-latency
  megakernel work.
- For our graph-node evidence, the problem is not only CPU launch overhead.
  CUDA graph is already enabled, yet the TurboMind NVFP4 bucket still spends
  `12.877 ms/token` on about `255` FP4 GEMM kernels per GPU. This points to
  low per-kernel efficiency, M-padding waste, K-split/reduction overhead, and
  dequant/scale overhead inside the GPU work.
- Therefore stage 1 should first prove or disprove an `M=1` NVFP4 GEMV
  primitive on the two MLP shapes. If a CUDA-core GEMV path cannot beat the
  HMMA path by at least `15%` on both dominant MLP shapes, the bottleneck is
  likely not M-padding alone and we should move to persistent scheduling or MLP
  fusion.

Takeaways for this project:

1. **M=1 should not be assumed to be a Tensor Core GEMM problem.**
   FlashDecoding++ explicitly treats decode linear work as GEMV/flat-GEMM and
   switches between CUDA Core and Tensor Core implementations by shape. Our
   current TurboMind path uses SM70 HMMA with `CTA_M=8` while only one row is
   live, so a large fraction of M dimension work is structurally padded away.

2. **Work-centric split-K is more relevant than more tile candidates.**
   Stream-K is directly aligned with our shape: skinny `M=1`, huge `N/K`, and
   split counts like `7` and `15`. The current TurboMind decomposition is still
   tile/split based with a generic epilogue/reduction. A persistent K-slice or
   Stream-K-style scheduler could keep SMs full without relying on fragile split
   heuristics.

3. **Shared-memory layout is a conditional target, not the first assumption.**
   QUICK attacks shared-memory write-back bank conflicts by offline
   quantization-aware interleaving. Our NCU evidence already shows excessive
   shared wavefronts and scoreboard/barrier stalls for the NVFP4 kernel, so a
   conflict-free NVFP4 prepared-weight layout is plausible. However, QUICK's
   strongest results are large-batch results, so for our M=1 target this route
   must be gated by NCU proof that shared wavefront / barrier stalls materially
   limit the MLP NVFP4 kernels.

4. **Dequantization overhead must be redesigned, not hidden.**
   QServe/QoQ focuses on progressive quantization, register-level parallelism,
   and compute-aware weight reordering to reduce low-throughput CUDA-core
   dequant overhead. For SM70 NVFP4, the analogous target is to co-design FP4
   value expansion, scale application, and HMMA/CUDA-core accumulation instead
   of treating dequant as a generic transform feeding a generic GEMM.

5. **Small-batch low-bit GEMV libraries are useful references, not drop-ins.**
   FastGEMV, Quill, tinygemm/any4, and GemLite all reinforce the same point:
   batch-1 decode should be treated as GEMV. They are not drop-ins for V100
   TurboMind NVFP4, but their design targets are directly relevant:
   coalesced packed-weight loads, register dequant, warp/block reductions,
   split-K only when it increases occupancy, and metadata reuse across adjacent
   blocks.

6. **LUT-GEMM is interesting but lower confidence for NVFP4.**
   NVFP4 has only 16 FP4 values, so a token-local LUT or CodeGEMM-like
   psumbook for activation times FP4 code value is tempting. However, our
   scales are per group and output column, so the scale dimension may destroy
   the simple LUT reuse. This is a research branch only after the direct M=1
   kernel and layout work.

7. **Megakernel/persistent decode is real, but too broad as the first cut.**
   Current megakernel work targets low-latency batch-1 inference by
   eliminating kernel boundaries, straggler bubbles, and memory-load bubbles.
   For this project, the tractable stage-1 version is not a whole-model
   megakernel; it is a persistent MLP or task-list kernel for the two dominant
   NVFP4 projections.

External-research-driven experiment queue:

| priority | route | first proof | accept / reject rule |
|---|---|---|---|
| P0a | raw M=1 CUDA-core/GEMV kernel for MLP gate/up and down | one default-off kernel family for `1x17408x5120` and `1x5120x8704` | rejected on 2026-06-30; best graph point `19.502 ms` vs graph GEMM `11.884 ms` |
| P0b | M=1 HMMA/Stream-K decode-specialized mainloop | one MLP shape using Tensor Core fragments or persistent K-slices without generic GEMM overhead | continue only if one MLP shape improves by `>=10%`, then require both MLP shapes `>=15%` |
| P1 | QUICK-style conflict-free NVFP4 prepared layout | add optional prepared layout for one MLP shape and compare NCU shared wavefronts/stalls | continue only if NCU shared wavefront / barrier stalls drop and wall improves |
| P2 | projection-level fusion | group same-input projections or fuse gate/up epilogue work where scale semantics allow it | continue only if graph-node FP4 bucket drops and low-overhead TPOT improves |
| P3 | persistent task-list decode engine | one narrow layer or projection-cluster task list, not whole-model persistence | only start after P0b/P2 fail to reach the stage target |
| P4 | LUT/codebook path | prototype FP4-code LUT for one small shape | low priority; reject if scale loads dominate |

## Operator Route Plan

### Route A: Shape and Layer Attribution

Status: required next step.

Current graph-node evidence gives the bucket and kernel count, but not enough
shape/layer attribution to choose the best low-level rewrite. Add an env-gated
NVFP4 attribution mode that records, for each `nvfp4_gemm_sm70_out` call:

- layer/module name if available from Python wrapper,
- logical projection name if available,
- `m`, `n`, `k`, group size, output dtype,
- selected dispatch policy,
- CUDA graph capture/replay mode,
- call count during steady decode.

The first output should be a per-token weighted shape table:

```text
shape/projection, calls/token/GPU, mean kernel us, total ms/token/GPU
```

Do not start a broad kernel rewrite until this table exists. It prevents
optimizing an impressive microbench shape that does not dominate the real
decode bucket.

### Route B: Decode-Specialized M=1 NVFP4 Kernel Family

Status: primary implementation route, but the raw scalar CUDA-core GEMV branch
is rejected.

The current TurboMind kernel is still a general GEMM family. For decode, the
hot path is single-token or very small-M. Build a separate NVFP4 decode kernel
family for the dominant shapes from Route A, with goals:

- reduce per-kernel average from about `50 us` toward `35-40 us`;
- reduce register pressure and shared-memory pressure;
- remove or simplify split/general epilogue paths that are unnecessary for
  `M=1`;
- improve shared-memory layout/coalescing for FP4 packed B and scale loads;
- keep output layout and quality identical to the current TurboMind route.

Estimated stage impact: if the dominant FP4 kernels move from `~50 us` to
`~40 us`, the TurboMind bucket can drop by roughly `2.5 ms/token`, which is
enough to cross the `<10 ms` stage target.

Candidate source areas:

- `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu`
- `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/mainloop_sm70.h`
- `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/epilogue.h`
- `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel_impl.h`
- `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu`

Post-raw-GEMV requirements:

- Do not build another scalar per-qword CUDA-core GEMV without a new design
  reason. The simple split-K version is slower than HMMA.
- Prefer a Tensor Core preserving M=1 design: specialize the current SM70 s884
  mainloop/epilogue for one live row, or build a Stream-K-style persistent
  K-slice scheduler that keeps HMMA utilization while removing generic
  split/reduction overhead.
- If CUDA-core is revisited, the prototype must use a materially different
  structure, e.g. half2/vectorized accumulation, warp-cooperative output
  tiles, and no large float partial matrix.

Concrete next implementation target:

- Add a new env-gated TurboMind dispatch policy for only
  `M=1, group_size=16, N/K in {17408x5120, 5120x8704}`.
- Keep the existing SM70 s884 operand loaders and HMMA path, but specialize the
  scheduler/epilogue for one live row and remove work that only exists for
  generic `M>=8` GEMM.
- First proof is one MLP shape beating the current graph GEMM mean by at least
  `10%` without changing output semantics; only then wire both MLP shapes and
  run the graph-node decode parser.

### Route C: Reduce GEMM Launch Count with Projection Fusion

Status: parallel design route; implement only after Route A proves the target
projection groups.

The bucket has about `255` FP4 GEMM launches/token/GPU. Even perfect kernel
micro-tuning leaves launch and per-kernel fixed overhead. Investigate fusing
or grouping projections that share the same activation input and can preserve
NVFP4 scale semantics:

- Q/K/V-style projections where the model path has separate linear modules.
- gate/up-style projections before SiLU when the checkpoint scale layout allows
  a combined output.
- GDN input/output projection groups if shape attribution shows they dominate.

This route is allowed to add a new prepared-weight layout, but only behind an
explicit env gate until quality and speed are proven.

Hard constraints:

- Do not merge projections if their NVFP4 global/per-group scales make the
  result numerically different beyond the classified quality gate.
- Do not count a route-hit as speed evidence. It must reduce the graph-node FP4
  bucket and the low-overhead TPOT.

### Route D: Persistent or Layer-Level Decode Linear Engine

Status: research route, not first implementation.

If Route B and Route C are insufficient, consider a narrow persistent/layer
engine that consumes a small task list of same-token NVFP4 linear work. This is
the lower-level innovation path analogous to TileRT-style execution, but scoped
to NVFP4 linear decode rather than the whole model.

Start with one layer or one projection cluster. Requirements:

- CUDA graph compatible.
- No cross-layer semantic reordering.
- Explicit task descriptor and output ownership.
- Measured win over normal graph launch sequence.

Do not start with a whole-model persistent engine; it is too broad for the
stage-1 target.

### Route E: Selector and Generic Kernel Micro-Tuning

Status: secondary. Useful, but not sufficient alone.

Selector tuning is still worth doing after shape attribution, but it should not
be the main plan. Current evidence already shows NVFP4 small-shape tuning is
enabled by default and the gap is too large for selector-only changes to
reliably move `12.877 ms -> <10 ms`.

Valid micro-tuning targets:

- CTA shape and split policy for the dominant Route-A shapes.
- Shared-memory swizzle/padding to reduce excessive wavefronts.
- Register-pressure reduction that raises active warps without reducing tensor
  core efficiency.
- Epilogue simplification for `M=1`.

## Known Non-Goals and Dead Ends

- Do not switch to eager to make profiling easier; this invalidates the target.
- Do not switch to marlin; the target route is TurboMind.
- Do not treat graph-node trace TPOT as the accepted speed baseline; graph-node
  tracing adds overhead.
- Do not spend another cycle proving the 18.8 ms host wait is a copy. It is GPU
  completion wait; graph-node trace already resolved this.
- Do not claim success from allreduce or attention wins alone. Stage 1 is
  specifically the TurboMind NVFP4 GEMM bucket.
- Do not revive broad side-stream worker designs that previously hung in real
  decode without a new minimal proof. Any persistent-engine route must start
  with one narrow task list and a timeout-safe microbench.
- Do not optimize only synthetic shapes. Every candidate must map back to the
  Route-A real shape table.

## Experiment Ledger

Add every significant attempt here before moving on.

| date | route | artifact | result | decision |
|---|---|---|---|---|
| 2026-06-30 | baseline graph-node breakdown | `bench_results/nsys_27b_nvfp4_turbomind_20260630/per_token_latency_breakdown_graph_node_i512_o16.md` | TM NVFP4 GEMM critical path `12.877 ms`, about `255` launches/token/GPU | stage-1 target is valid; optimize TM FP4 bucket first |
| 2026-06-30 | NVFP4 5-shape microbench baseline | `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_qwen35_27b_tp2_m1_5shape.json` | non-graph event wall `17.611 ms`; MLP gate/up `6.431 ms`, MLP down `4.323 ms` | use for relative ranking only; primary hot shapes are MLP |
| 2026-06-30 | NVFP4 single-op CUDA graph microbench | `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_qwen35_27b_tp2_m1_5shape_cudagraph.json` | graph replay event wall `16.635 ms`; MLP gate/up+down `9.476 ms` | confirms launch gap is not the only issue; optimize kernel body/split strategy |
| 2026-06-30 | forced split/swizzle selector probes | `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_force_gate_s2w4.json`, `...force_down_s7w1.json` | best observed gate/up `6.247 ms`, down `4.215 ms` weighted; only `0.1-0.2 ms` class gain | selector forcing alone rejected for stage 1 |
| 2026-06-30 | NVFP4 `8x256x64` registry candidate | `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_force_gate_8x256_s2w4.json`, `...force_down_8x256_s7w1.json` | gate/up regressed; down improved only to `4.139 ms`; full default total stayed in noise band | reverted; do not repeat unless paired with a new M=1 mainloop |
| 2026-06-30 | raw M=1 NVFP4 CUDA-core GEMV prototype | `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_mlp_rawgemv_split64_graph.json` | best graph point `19.502 ms` for MLP gate/up+down vs graph GEMM `11.884 ms`; best non-graph point `22.743 ms` vs non-graph GEMM `10.881 ms` | reject raw scalar GEMV; next route must preserve HMMA throughput or use persistent/Stream-K scheduling |
| 2026-06-30 | warp-cooperative raw GEMV | `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_mlp_rawgemv_warp.json` | non-graph MLP total `36.020 ms`, worse than scalar split-K GEMV | reject; shuffle/K-cooperation without Tensor Core throughput is not enough |
| 2026-06-30 | half2 raw GEMV with fp16 partials | `bench_results/nsys_27b_nvfp4_turbomind_20260630/nvfp4_gemm_micro_mlp_rawgemv_h2_split128_graph.json` | best graph MLP total `16.108 ms`; faster than scalar raw GEMV but `1.36x` slower than graph GEMM and less accurate | reject as production route; use result to stop pure CUDA-core GEMV tuning |

## Stage Exit Criteria

Stage 1 is complete only when all are true:

- TurboMind NVFP4 GEMM critical-path mean is `< 10.000 ms/token` using the
  graph-node parser.
- Low-overhead TPOT improves on the same accepted route and benchmark shape.
- Token/output quality passes the current classified quality gate.
- The experiment ledger records the successful route and any rejected branches.
- The migration control document points to this plan and the final artifact.

## AWQ DDTree Verifier Track

### Objective

Reduce the Qwen3.6-27B-AWQ DDTree target verifier plus accepted-state update
wall time:

```text
target verifier + state update: 51.86 ms -> <= 40 ms
```

This supersedes the narrower 2026-06-30 single-kernel target of forcing the
rank-local TurboMind AWQ uint4 GEMM bucket from about `18.6 ms/rank` to
`10.0 ms/rank`. AWQ GEMM is still the largest bucket, but it is no longer the
only accepted source of improvement. A valid `~10 ms` recovery may combine AWQ
GEMM, DDTree attention, Torch small-kernel cleanup, TP all-reduce imbalance,
GDN delta/gating, sampling, and accepted-state update reductions.

The route remains microbench-first for individual candidates, but the accepted
near-term gate is the total profile line. Do not rerun expensive end-to-end
DDTree serving benchmarks until a candidate plausibly moves either
`target_forward` or `state_update_wall_cpu`. The end-to-end target still
requires output quality parity and the existing DDTree acceptance metrics;
operator-only speedups are not sufficient by themselves.

### Measurement Contract

Use the real-weight harness:

```text
benchmarks/benchmark_sm70_awq_verifier_micro.py
```

Required run shape unless a note says otherwise:

- model: `/home/ymzx/models/Qwen3.6-27B-AWQ`
- CUDA device: one V100/SM70 rank
- verifier shape: `m=17`, matching `budget=16` plus root
- weighted verifier suite:
  - `63` MLP gate/up calls
  - `63` MLP down calls
  - `47` linear-attention qkv calls
  - `47` linear-attention z calls
  - `16` full-attention o-proj calls
- time only `awq_gemm_sm70_out` after `awq_sm70_prepare` and warmup
- record exact env, JSON artifact, selected kernel when tracing is enabled, and
  whether the candidate is default behavior or env-forced only

Current accepted comparison artifacts:

| artifact | weighted AWQ GEMM time | note |
|---|---:|---|
| `bench_results/ddtree_awq_micro_20260630/preserve_m17.json` | `34.04 ms` | default split preservation is too conservative for verifier small-M |
| `bench_results/ddtree_awq_micro_20260630/nopreserve_m17.json` | `25.03 ms` | earlier best dynamic-selector total |
| `bench_results/ddtree_awq_micro_20260630/all_shapes_m17_dynamic_nopreserve_recheck.json` | `24.80 ms` | current best microbench total for candidate ranking |
| `bench_results/ddtree_awq_micro_20260630/restored_default_recheck_200.json` | `27.97 ms` | restored non-forced default after negative experiments were removed |
| `bench_results/ddtree_awq_micro_20260630/all_shapes_m17_forced_mgroup_c32x128_recheck.json` | `28.14 ms` | `c32x128` mgroup candidate; tested and reverted |

Current best dynamic breakdown from
`all_shapes_m17_dynamic_nopreserve_recheck.json`:

| bucket | mean kernel time | weighted time |
|---|---:|---:|
| MLP gate/up `17x17408x5120` | `132.20 us` | `8.33 ms` |
| MLP down `17x5120x17408` | `129.22 us` | `8.14 ms` |
| linear-attention qkv `17x10240x5120` | `88.28 us` | `4.15 ms` |
| linear-attention z `17x6144x5120` | `65.32 us` | `3.07 ms` |
| full-attention o-proj `17x5120x6144` | `69.25 us` | `1.11 ms` |

Total-path control point from the realistic code profile:

| component | mean |
|---|---:|
| `target_forward` | `40.540 ms` |
| `target_logits` | `2.848 ms` |
| `target_rejection_sample` | `2.394 ms` |
| target verifier subtotal | `45.78 ms` |
| `state_update_wall_cpu` | `6.082 ms` |
| target verifier + state update | `51.86 ms` |
| `draft_wall_cpu` | `17.450 ms` |

Nsight hierarchy for the selected target-forward window:

| bucket | per-rank / wall-scale read | note |
|---|---:|---|
| AWQ uint4 TurboMind GEMM | `18.57-18.58 ms/rank`, `46.3%` wall share | largest bucket, but not the only target |
| DDTree paged attention | `4.65 ms/rank`, `11.6%` wall share | 16 verifier launches |
| Torch native elementwise/scatter | `3.88-3.89 ms/rank`, `~9.7%` wall share | many small native kernels |
| TP custom all-reduce | rank0 `3.86 ms`, rank1 `1.90 ms` | rank imbalance extends the wall |
| GDN delta/gating update | `3.30 ms/rank`, `8.2%` wall share | plus separate accepted-state update cost |

### Nsight Compute Findings

Representative gate/up shape, `m=17,n=17408,k=5120`, V100:

| route | duration | grid / waves per SM | regs/thread | smem/block | SM | DRAM | L1TEX | L2 | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| default non-mgroup | `167.168 us` | `68` CTAs / `0.47` | `192` | `16.4 KiB` | `44.31%` | `25.10%` | `48.20%` | `14.08%` | too little grid parallelism |
| mgroup `c32x256` | `126.240 us` | `136` CTAs / `0.94` | `187` allocated `192` | `32.8 KiB` | `58.42%` | `33.27%` | `64.77%` | `19.33%` | better utilization, register/smem limited |
| mgroup `c32x256` dense-fp16-output epilogue | `124.990 us` | `136` CTAs / `0.94` | `190` | `32.8 KiB` | `59.12%` | `33.91%` | `67.51%` | `19.57%` | diagnostic only; no material microbench win |
| mgroup `c32x128` | `126.464 us` | `136` CTAs / `0.94` | `187` allocated `192` | `16.4 KiB` | `58.42%` | `33.21%` | `64.90%` | `19.30%` | smem lower, still register-limited |

Interpretation:

- The hot gate/up kernel is not a pure DRAM-bandwidth bottleneck. It is a
  mixed Tensor Core / LSU / ALU path with low occupancy and small-GEMM
  parallelism limits.
- The mgroup path fixes the biggest grid underfill problem but still allocates
  `192` registers/thread. That caps the kernel at two active blocks per SM
  even when shared memory is reduced.
- Reducing shared memory alone does not recover the target. The new
  `c32x128` candidate halves shared memory versus `c32x256` but keeps the same
  register cap and does not improve full weighted time.
- Simplifying the dense verifier epilogue alone does not recover the target.
  The default-off dense-fp16-output diagnostic still used about `190`
  registers/thread, measured only a `~1%` NCU duration change, and did not
  improve the CUDA-event microbench enough to matter.
- `__launch_bounds__(..., 3)` is not a valid shortcut. It reduced register
  allocation pressure syntactically but slowed the gate/up shape to about
  `187.48 us`, likely by hurting scheduling/ILP or causing worse local-memory
  behavior.

### AWQ Experiment Ledger

| date | route | artifact | result | decision |
|---|---|---|---|---|
| 2026-06-30 | dynamic no-preserve baseline | `bench_results/ddtree_awq_micro_20260630/all_shapes_m17_dynamic_nopreserve_recheck.json` | weighted `24.80 ms`; gate/up `132.20 us`, down `129.22 us` | current microbench comparison point |
| 2026-06-30 | 24-row CTA-M candidates | local reverted experiment | gate/up slower than current `32x256x32`; no kept artifact is accepted | reverted; do not repeat without a new register model |
| 2026-06-30 | direct verifier epilogue/launch branch | `bench_results/ddtree_awq_micro_20260630/m17_direct_dense_verifier_probe.json` | weighted `40.09 ms`; down path regressed to `293.79 us` | reverted; direct wrapper without mainloop change is wrong direction |
| 2026-06-30 | mgroup `c32x128` candidate | `bench_results/ddtree_awq_micro_20260630/all_shapes_m17_forced_mgroup_c32x128_recheck.json` | weighted `28.14 ms`; gate/up `137.52 us`; down `144.96 us` | reverted; shared-memory reduction alone did not help |
| 2026-06-30 | forced launch bounds for 3 blocks/SM | `bench_results/ddtree_awq_micro_20260630/gate_up_m17_mgroup_c32x128_launchbounds3.json` | gate/up `187.48 us` | reverted; source-level low-register mainloop is needed |
| 2026-06-30 | no-prefetch low-live-range mainloop | `bench_results/ddtree_awq_micro_20260630/gate_up_m17_mgroup_c32x128_noprefetch_probe.json` | gate/up `189.07 us` | reverted; reducing live range by removing prefetch loses too much ILP/latency hiding |
| 2026-06-30 | dense-fp16-output epilogue diagnostic | `bench_results/ddtree_awq_micro_20260630/gate_up_m17_mgroup_denseout_probe.json`; NCU `bench_results/ddtree_awq_micro_20260630/ncu/gate_up_m17_mgroup_denseout_gemm_full_sudo.ncu-rep` | gate/up CUDA-event mean `140.65 us`; NCU duration `124.99 us`, registers/thread `190`, SM `59.12%`, DRAM `33.91%` | reverted; epilogue-only simplification did not lower the register/occupancy limiter |
| 2026-06-30 | two-slot K-buffer `lowreg2` mainloop | `bench_results/ddtree_awq_micro_20260630/gate_up_m17_mgroup_lowreg2_200.json`; NCU `bench_results/ddtree_awq_micro_20260630/ncu/gate_up_m17_mgroup_lowreg2_metrics_sudo.ncu-rep` | bitwise exact; gate/up `143.64 us`; registers/thread `184`; SM `58.18%` | reverted; lowered registers slightly but did not cross an occupancy tier and added control/ALU overhead |
| 2026-06-30 | mgroup `streamdense` epilogue | `bench_results/ddtree_awq_micro_20260630/gate_up_m17_mgroup_streamdense_200.json`; NCU `bench_results/ddtree_awq_micro_20260630/ncu/gate_up_m17_mgroup_streamdense_metrics_sudo.ncu-rep` | bitwise exact; gate/up `139.44 us`; registers/thread `192`; no stable win over forced mgroup | reverted; epilogue streaming did not remove the register limiter |
| 2026-06-30 | mgroup `lowreg2+streamdense` | `bench_results/ddtree_awq_micro_20260630/gate_up_m17_mgroup_lowreg2_streamdense_200.json`; NCU `bench_results/ddtree_awq_micro_20260630/ncu/gate_up_m17_mgroup_lowreg2_streamdense_metrics_sudo.ncu-rep` | gate/up `140.06 us`; NCU `124.03 us`, registers/thread `184`, SM `60.15%` | reverted; not a CUDA-event win and still far from the total-path target |
| 2026-06-30 | current-best-family `fff c32x128 streamdense` | `bench_results/ddtree_awq_micro_20260630/gate_up_m17_fff_c32x128_streamdense_200.json`; exactness `bench_results/ddtree_awq_micro_20260630/exactness/gate_up_m17_fff_c32x128_streamdense_compare.json` | bitwise exact; streamdense `183.10 us` vs same fast-selector default `166.46 us` | reverted; confirms stream-store epilogue is negative on the active best family |

### Next AWQ Implementation Route

Selector/config tuning and epilogue-only changes are no longer the main path.
The new main target is total `target verifier + state update <= 40 ms`. A
successful change may come from any bucket, but each candidate must show a
credible total-path contribution before end-to-end serving is repeated.

For the AWQ sub-bucket, the next useful implementation should specialize the
SM70 U4 verifier mainloop for `m=17` while preserving the TurboMind AWQ
packed-weight layout and exact output semantics.

Start from:

- `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/mainloop_sm70.h`
- `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel_impl.h`
- `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/epilogue.h`
- `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu`

Current AWQ-specific hypotheses:

- Keep the useful `CTA_N=256`/mgroup parallelism for the MLP shapes.
- Do not simply disable the current prefetch/transform overlap. The no-prefetch
  probe reduced the intended live range but regressed gate/up to about
  `189 us`.
- Do not start with another epilogue-only simplification. Dense-output,
  streamdense, and `c32x128` shared-memory reduction did not reduce the
  register/occupancy limiter enough to matter.
- The next AWQ candidate should attack mainloop fixed cost and scheduler
  underutilization, such as verifier-specific V-scale reuse for
  `GroupSizeV=128` and/or a carefully validated `CTA_K=64` candidate, before
  any more epilogue work.

Total-path hypotheses for the new `51.86 -> <=40 ms` target:

- Reduce `state_update_wall_cpu=6.082 ms` first if it contains avoidable CPU
  synchronization or per-step small kernels; this is the cleanest non-AWQ
  10ms contributor.
- Reduce Torch native elementwise/scatter kernel count in the verifier graph;
  the selected target-forward window spends about `3.9 ms/rank` there.
- Investigate the rank-0 TP all-reduce imbalance (`3.86 ms` vs `1.90 ms`),
  because the slower rank extends the wall by roughly `2 ms`.
- Keep AWQ mainloop work active, but no longer require one AWQ kernel family to
  supply the entire `~10 ms` reduction.

State-update experiment notes, 2026-06-30:

- Runner profile now splits `state_update_wall_cpu` into validation, attention
  compact, Mamba compact, input-batch update, and drafter-context compact.
- Artifact
  `bench_results/ddtree_total_target_20260630/realcode2_o128_mamba_fused_cache_profile.json`
  kept official sampling (`temperature=1.0/top_p=0.95/top_k=20`), `budget=16`,
  text-only `limit_mm_per_prompt={"image":0,"video":0}`, and measured
  `mean_acceptance_length=12.86`, `draft_acceptance_rate=0.741`,
  `generate_seconds=3.48` for `256` output tokens. The steady profile was
  still above target: `target_forward=56.583 ms`, `target_logits=3.120 ms`,
  `target_rejection_sample=4.031 ms`, and `state_update_wall_cpu=5.941 ms`.
- Current 27B-AWQ runs select `mamba_cache_mode=none`, so align-mode fused
  postprocess cannot replace explicit DDTree Mamba state compaction. The
  align-mode deferral path remains covered by tests, but it is not the active
  serving path for this artifact.
- A DDTree Mamba compact experiment that reused `MambaCopyBuffers` and
  `batch_memcpy` for whole-block branch copies was tested and reverted. Artifact
  `bench_results/ddtree_total_target_20260630/realcode2_o128_mamba_batchcopy_cache_profile.json`
  kept the same setup and measured `mean_acceptance_length=12.85`, but profile
  cost worsened: at `calls=20`, `state_update_wall_cpu=16.311 ms`,
  `state_update_mamba_compact_cpu=6.696 ms`, and
  `state_update_drafter_context_cpu=7.350 ms`. Do not repeat this batch-memcpy
  design without a different state-layout/copy-size hypothesis.
- The remaining high-value target is now the target-forward wait itself: recent
  worker-profile lines show `full_logits_split pre_sync_ms` around
  `31-36 ms` for the `rows=17` verifier logits path. That wait corresponds to
  GPU work completing before logits and dominates any state-update-only saving.

Validation order:

1. Update runner profiling so `state_update_wall_cpu` is split into CPU wait,
   GPU kernels, accepted-state copy/commit, and metadata work.
   `gpu_model_runner.py` now reports the first split in the existing
   `SM70 spec runner profile avg_ms` line:
   `state_update_validate_cpu`, `state_update_attn_compact_cpu`,
   `state_update_mamba_compact_cpu`, `state_update_input_batch_cpu`, and
   `state_update_drafter_context_cpu`.
2. For AWQ candidates, prove bitwise exactness and beat the current best
   gate/up/down microbench before counting the change.
3. For graph cleanup candidates, use Nsight or runner profile to show a
   reduction in the selected target-forward or state-update window.
4. Only after the total target verifier + state update estimate moves toward
   `40 ms`, rerun the `budget=16`, official-sampling DDTree coding benchmark.

DDTree total-path result, 2026-06-30:

- The `<=40 ms` target was reached by reducing metadata and state-update cost,
  not by another AWQ mainloop candidate. The accepted artifact is
  `bench_results/ddtree_total_target_20260630/realcode2_o256_slotsidecar_align_16k.*`.
- The runner now uses CPU sidecars for DDTree accepted rows, sampled-token
  counts, current positions, and current request indices. These sidecars feed
  accepted-state update, no-clamp detection, and attention KV slot-pair
  calculation.
- The attention KV slot-pair fast path avoids the previous hot
  `slot_mapping.detach().cpu().tolist()` synchronization when CP/DCP=1 and
  block size matches the active cache layout. Fallback remains unchanged for
  unsupported layouts.
- Last steady hot interval in the align benchmark:

  | bucket | ms |
  | --- | ---: |
  | target forward | 33.270 |
  | target logits | 2.890 |
  | target rejection sample | 1.290 |
  | state update total | 1.938 |
  | total verifier+state | 39.388 |

- State-update sub-buckets in that interval:

  | state-update bucket | ms |
  | --- | ---: |
  | validate | 0.029 |
  | attention compact | 0.797 |
  | Mamba compact | 0.013 |
  | input batch | 0.772 |
  | drafter context | 0.323 |

- Negative A/Bs to avoid repeating: direct Mamba compact copy
  (`realcode2_o256_slotsidecar_mambadirect_16k.*`, last hot `41.422 ms`) and
  the earlier whole-block `batch_memcpy` design
  (`realcode2_o128_mamba_batchcopy_cache_profile.json`,
  `state_update_wall_cpu=16.311 ms`).
- AWQ operator work remains useful for future end-to-end throughput, but this
  milestone shows the nearest `51.86 -> <=40 ms` reduction came from removing
  CPU/GPU metadata synchronization and aligning state update, rather than from
  forcing another TurboMind U4 kernel variant.
