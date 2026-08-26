# Qwen3.8 FP8 exact-8K prefill 5500 plan on V100

## Scope and acceptance contract

This work targets one narrow, reproducible contract:

- model: `/home/ymzx/models/Qwen3.8-27B-FP8`;
- hardware: tensor parallel four on four fully connected V100-SXM2-32GB GPUs;
- prompt/output: exactly 8000 prompt tokens and one sampled token;
- dtype/quantization: FP16 activations, FP8 checkpoint, FP8-E5M2 KV cache;
- attention: Flash-V100 full attention and FlashQLA linear attention;
- scheduler: chunked prefill and prefix caching enabled, one sequence, 8096
  maximum batched tokens;
- graph: `VLLM_COMPILE`, `FULL_AND_PIECEWISE`, compile size 8000, capture
  sizes 1/2/8000;
- sampling: temperature 1.0, top-p 0.95, top-k 20, seed 20260823;
- communication: TP custom all-reduce enabled and the accepted exact-8K
  reduce-scatter/Gemma RMSNorm/all-gather fusion active.

The acceptance metric is request-wall prompt throughput, not an isolated
kernel rate. The goal is at least **5500 prompt tok/s**, which requires the
steady request wall time to be at most `8000 / 5500 = 1.454545 s`. Pure
prefill time is reported separately. Correctness requires seven official
sampling repeats, unchanged route admission outside this exact contract, and
an explicit rollback switch.

## Accepted baseline and remaining budget

The source-matching accepted baseline produced six steady request-wall samples:

`1.559085 / 1.561460 / 1.560481 / 1.562664 / 1.564697 / 1.564643 s`.

The median is `1.562062 s`, or **5121.44 prompt tok/s**. The pure prefill
median is `1.547100 s`, or **5170.96 tok/s**. Seven of seven official samples
returned token `[1061]`, text `" This"`.

Reaching 5500 therefore requires removing about **107.5 ms** from the steady
request critical path, or 6.88% of the current request wall time. With the
current roughly 15 ms request-minus-prefill overhead, the corresponding pure
prefill target is about `1.4396 s`.

Baseline artifact:

`/data/minimax-h3/task-cache/qwen38-fp8-prefill-utilization-20260823/results/full_model_cutlass_all_projections_final_source_tp4_fused_graph_i8000.raw.json`

## Trace localization

The retained Nsight Systems trace is:

`/data/minimax-h3/task-cache/qwen38-fp8-prefill-utilization-20260823/profiles/full_model_cutlass_final_source_nsys_node_i8000_gpus4567.sqlite`

Each rank executed 1757 kernels on one CUDA stream. After the initial request
setup gap, the model region is nearly continuously occupied, so launch-gap
tuning cannot provide the required 107.5 ms. The following values are mean
per-rank kernel service; the range shows rank skew. Kernel service is not
summed across TP ranks.

| Category | Mean service | Share | Rank range |
| --- | ---: | ---: | ---: |
| gate/up exact CUTLASS | 468.787 ms | 30.27% | 457.594-476.271 ms |
| QKV/QKVZ exact CUTLASS | 325.594 ms | 21.02% | 318.374-330.700 ms |
| fused collective + Gemma norm | 274.364 ms | 17.72% | 254.165-301.994 ms |
| down exact CUTLASS | 218.743 ms | 14.12% | 214.052-221.915 ms |
| FlashQLA GDN attention core | 78.876 ms | 5.09% | 77.606-80.162 ms |
| full attention | 76.801 ms | 4.96% | 75.609-77.921 ms |
| FP8 weight dequantization | 31.126 ms | 2.01% | 30.264-32.157 ms |
| interleaved SiLU/mul | 20.101 ms | 1.30% | 19.421-20.800 ms |
| other kernels | 54.302 ms | 3.51% | 53.293-55.450 ms |

This rules out treating attention or QKV as the only target. Gate/up, QKV,
down, and the collective boundary account for about 83% of per-rank GPU
service. The first architecture experiment therefore joins communication and
the following projection instead of tuning either in isolation.

## Architecture candidate: token-shard consumer projection

The accepted fused boundary currently reduce-scatters token rows, performs the
mixed-dtype Gemma RMSNorm on the owned 2000-row shard, and writes that shard to
all four peer output tensors. The next column-parallel projection waits for the
full `[8000, 5120]` normalized tensor before launching one exact CUTLASS GEMM.

For exact `M=8000`, the candidate changes this boundary contract:

1. Reduce-scatter and Gemma RMSNorm produce only the rank-owned
   `[2000, 5120]` normalized shard, retaining the accepted FP16 reduction and
   FP32 residual order.
2. The following QKV/QKVZ or gate/up projection consumes the local shard first
   and then the three peer shards as they become visible.
3. Four `M=2000` CUTLASS subproblems write directly into their corresponding
   row slices of the normal full projection output. The K reduction and output
   row order are unchanged.
4. A replay-safe TP barrier protects peer visibility. A later iteration may
   replace the four launches with a pointer-array/grouped persistent launch,
   but only if the simple version proves that avoiding the materialized
   all-gather is profitable on V100 NVLink.

This is the SM70 analogue of an all-gather/matmul consumer pipeline. Generic
vLLM Sequence Parallelism and AsyncTP cannot be enabled unchanged here:

- SM70 has no generic SP admission threshold;
- the accepted projection is a custom block-FP8 dequant + CUTLASS operation,
  while the generic fusion pattern matches `aten.mm` (or newer scaled-mm
  routes);
- the current exact-8K graph uses SM70-specific mixed-dtype residual semantics
  that must not be replaced by a generic norm.

The prototype therefore reuses the generic topology, but implements a bounded
SM70 custom consumer. It does not use Ampere/Hopper-only `cp.async`, TMA, or
native FP8 Tensor Core instructions.

## Rejected paths that must not be repeated

- Side-stream two-workspace FP8 weight dequantization across RMSNorm was
  bitwise correct but regressed `7.98956 -> 8.00819 ms`.
- cuBLAS algorithm 106 lost to the accepted CUTLASS operation.
- Tile-ready down/output projection overlap failed to beat the full-model
  baseline. Its acceptance runs were around 1.555-1.559 s pure prefill.
- CUDA graph wait rewrites regressed to roughly 1.565-1.572 s pure prefill;
  earlier variants also errored or deadlocked.
- A prior generic symmetric-memory down+norm pipeline regressed its isolated
  boundary. It is evidence against copying AsyncTP literally, not against the
  new bounded consumer that removes the normalized all-gather.

## Validation ladder and rollback

1. Reproduce the accepted baseline on current main with the exact official
   contract and record wall/pure-prefill samples, route hits, binary hash, and
   GPU ownership.
2. Add an operator benchmark for fused norm plus the following exact QKV and
   gate/up projections. Compare the complete boundary, not GEMM alone.
3. Require finite output and bitwise equality where the unchanged per-row
   CUTLASS reduction permits it. Any numerical relaxation requires a separate
   model-quality gate and cannot be accepted from a speed smoke.
4. Integrate only after the complete operator boundary beats the accepted
   route by enough to plausibly save 107.5 ms over 127 boundaries.
5. Run a full candidate/control/candidate sandwich, seven official sampling
   repeats, and a source-matching Nsight trace. Report request-wall and pure
   prefill throughput independently.

The candidate is exact-shape and default-off until acceptance. Its rollback
gate will be `VLLM_SM70_FP8_PREFILL_SHARD_CONSUMER=0`. The existing exact
CUTLASS rollback remains `VLLM_SM70_FP8_PREFILL_CUTLASS=0`.
