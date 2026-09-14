# SM70 Q8000 batched-tail prefill (experimental)

Build with `-DVLLM_FLASH_ATTN_SM70=ON -DVLLM_SM70_79T_PREFILL=ON`.
This replaces the legacy architecture implementation inside the normal
`vllm.vllm_flash_attn._vllm_fa2_C` extension. It does not load a private DSO.
The existing v37 implementation remains available in the same extension.

Runtime selection:

```bash
export VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL=1
export VLLM_FLASH_V100_PREFILL_D256_GQA_V37=0
export VLLM_FLASH_V100_ROUTE_SUMMARY=1
```

The admitted shape is FP16 Q8000/Hq6/Hkv1/D256, causal, scale 1/16,
KV16000 through KV256000 in steps of 8000. Other shapes retain existing
routes. For Qwen TP4 use `max_num_batched_tokens=8000`. E4M3 storage uses
the separately resolved `sm70_v37_e4m3_bridge`; selecting the architecture
kernel must not disable that storage conversion. Both architecture and
E4M3 bridge route counters must appear on every rank in an E4M3 cold run.

The historical 79T recipe uses zero-shift exponentials and unguarded FP16
PV accumulation. Real model inputs overflow that recipe, although zero-mean
random-input tests pass. The qualified integration samples score maxima at
stride 8 with a fixed margin and exponent cap, centers biased values, and
scales value residuals by an exact power of two with 64x headroom. FP16
tensor-core PV is retained for throughput; row masses and the online block
merge use FP32 before the value center is restored. This is an approximate
attention path and must pass both sampled FP32-oracle checks and cold model
requests. Current qualified medians exceed 75 TFLOPS at KV128K and KV256K;
see `VALIDATION.md` for the exact contract and quality limits. The build
option is OFF by default.

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
