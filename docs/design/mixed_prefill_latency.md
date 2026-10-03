# Mixed prefill latency control

When resident requests are decoding, an incoming long prompt can occupy the GPU
for a large prefill chunk. Its decode/verification rows may use fast kernels
and still wait seconds for the rest of the step to finish.

The scheduler now reserves tokens for eligible resident decoders and limits
the aggregate prefill work sharing their step. The default GPU step latency
target is 250 ms. A conservative initial budget is adjusted using completed
GPU measurements, so the policy follows the actual model, device, parallelism,
and context instead of an exact model-shape contract. This is a soft target,
not a deadline guarantee. No resident decode means the normal large prefill
budget is retained. Pure decode retains its existing FULL CUDA graph route.

`--mixed-prefill-step-latency-ms` changes the target; zero disables the policy
for comparison. There is no extra environment variable. GPU timing creates
events only for controlled mixed steps, checks their readiness without waiting,
and returns plain completed timing data to the scheduler. Pipeline parallel
and indivisible multimodal chunks retain their existing scheduling behavior.

## Small chunks and recurrent state

Mamba align mode allows chunks shorter than a state block. The worker continues
updating the current running state in place. At a retained boundary the
scheduler ends the chunk, and the next chunk copies that state into a fresh
running block. Prefix lookup admits completed checkpoint states only.
Regression tests exercise allocation, worker state movement, boundary snapshot
immutability, and prefix hits with 560/1024 and 512/8192 token chunks/blocks.

## Prefill kernels

The mixed attention/GDN routes incorporate the work in #841. Mixed-row plans
copy pinned host metadata asynchronously rather than creating device tensors
from pageable host lists. The NVFP4 small-prefill dequantization candidate uses
shared memory to transpose the original FP16 values into aligned vector stores;
it keeps the existing scale arithmetic and cuBLAS GEMM. Large prefill chunks
and the small-row decode kernels retain their previous dispatch.

A first fused WMMA dequantization/GEMM candidate was rejected: on the screened
512–8192-row shapes it was approximately four times slower than the existing
dequantization plus cuBLAS path. It is not included in the implementation.

## Validation

Use `benchmarks/benchmark_mixed_prefill_latency.py` with four fixed 32K token-ID
prompts. Two resident requests reach steady decode before two additional
prompts arrive. The client reports resident token throughput during the
incoming prefill window, the longest gap between resident stream updates,
and each new request's TTFT. It counts returned token IDs, not stream chunks.
Resident EOS is disabled to sustain the load; output quality must be checked
separately with natural completion behavior. There is no prefill/decode barrier
in this mixed-load measurement.

Matched pure-decode C1/C4/C8 measurements and natural-output checks remain
required before promotion. Online A/B results and the final default tradeoff
are pending GPU validation.
