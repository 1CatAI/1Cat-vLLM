# SM70 Q8000-core long prefill

SM70 FlashAttention builds include this kernel by default. Build with
`-DVLLM_FLASH_ATTN_SM70=ON`; pass `-DVLLM_SM70_79T_PREFILL=OFF` to omit it.
This replaces the legacy architecture implementation inside the normal
`vllm.vllm_flash_attn._vllm_fa2_C` extension. It does not load a private DSO.
The existing v37 implementation remains available in the same extension.

The Q8000-core route is selected by default for its admitted shapes. These are
the corresponding explicit runtime settings; set V37 to 1 for rollback:

```bash
export VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL=1
export VLLM_FLASH_V100_PREFILL_D256_GQA_V37=0
export VLLM_FLASH_V100_ROUTE_SUMMARY=1
```

The admitted shape is FP16 Q8000 through Q8192/Hq6/Hkv1/D256, causal, scale
1/16, with KV at least Q through 262144 and aligned to 32 tokens. The 32-token
alignment is required by the prefix PV Tensor Core K tile; other KV lengths
stay on the general path. Q8000 runs through its qualified 320-token
tail specialization. Q8192 has a native 256-token tail specialization. Q8001
through Q8191 are leading-padded to Q8192, which preserves bottom-right causal
alignment while keeping the added work below 2.4%. Other shapes retain existing
routes. For one full chunk, use a per-request long-prefill threshold of 8192.
The total `max_num_batched_tokens` can be larger: for example, use 16384 plus
`long_prefill_token_threshold=8192` to schedule two full Q8192 chunks in the
same step. Non-contiguous paged KV is gathered into a reusable workspace for
each request, including multi-request batches. E4M3 storage uses
the separately resolved `sm70_v37_e4m3_bridge`; selecting the architecture
kernel must not disable that storage conversion. Both architecture and
E4M3 bridge route counters must appear on every rank in an E4M3 cold run.
The `prefill_dense_d256_gqa_79t_fp32` counter identifies this compute family;
`prefill_dense_d256_gqa_79t_fp32_q8192` identifies the native Q8192 kernel and
`prefill_dense_d256_gqa_79t_fp32_q8192_pad` identifies padded mixed chunks.

The historical 79T recipe uses zero-shift exponentials and unguarded FP16
accumulation. Real model inputs overflow that recipe, although zero-mean
random-input tests pass. The qualified integration samples score maxima at
stride 8 with a fixed margin and exponent cap, centers biased values, and
scales value residuals by an exact power of two with 64x headroom. PV uses
FP16 Tensor Core operands with FP32 MMA accumulation, writes each 24K prefix
block in FP32, and performs the online prefix/tail merge in FP32 before the
value center is restored. The 128x256 threadblock and 64x64 warp shape avoid
the register spills that made the earlier FP32 path slow. This remains an
approximate attention path because scores and probabilities are stored in
FP16 and score maxima are sampled. It must pass both sampled FP32-oracle
checks and cold model requests. Current qualified medians exceed 75 TFLOPS
at KV128K and KV256K; see `VALIDATION.md` for the exact contract and quality
limits. The build option defaults to ON for SM70 FA2 builds, and the runtime
route defaults to this FP32-accumulated kernel.

The private `MmaPipelined79T` template retains transform `set_valid()` and
`finalize()` hooks. Without finalization the tail row masses remain zero and
normalization produces non-finite outputs. The template has a distinct name
so the hooks cannot alter other CUTLASS translation units. The tail adapter
preserves the historical normalized-output contract (`unnormalized=0`).

`benchmarks/benchmark_sm70_79t_operator.py` reports CUDA-event operator time
and sampled FP32-oracle error. `benchmark_sm70_79t_cold.py` disables prefix
caching, excludes engine load and a short warmup, and records client TTFT,
request wall time, subsequent decode rate, and output token IDs separately.
The deterministic sampling override is explicit; EOS is respected.
Route snapshots use a benchmark-owned callable through local worker RPC;
set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` for this local benchmark only.

Source lineage: the 2026-09-12/14 79T torch-port experiment based on the
historical 7787-line architecture source, moved into the parent repository.
The CMake recipe is the source of truth for tile shapes and flags. No
operator TFLOPS result should be reported as model prompt tokens/s.
