# SM70 v37 long-prefill attention

This is a prefill-only, FP32-accumulating implementation for one request with
FP16 activations, six query heads, one KV head and head dimension 256. It does
not change decode, speculative decoding, sampling, or the global KV format.

## Dataflow and numerical contract

We pack the six query heads into the GEMM row dimension. For a causal chunk,
the preceding keys form a non-causal prefix and the current chunk forms an
exactly masked tail. The prefix uses M128/N256/K32 SM70 Tensor Core GEMMs.
Its QK epilogue forms tile-local probabilities from FP32 logits and FP32
max/sum statistics, before rounding probabilities to FP16 tensor operands.
PV rescales those probabilities and accumulates the numerator and online
partial state in FP32. The tail also retains its unnormalized output in FP32.
We combine both states before the single final FP16 output conversion.

“FP32 accumulation” is not a claim of FP32 inputs or exact real arithmetic:
probability operands and the final output are still FP16. The explicit E4M3
bridge expands stored bytes with their per-layer scales; it does not undo
the original KV quantization loss.

## Admission and fallback

- SM70 only; contiguous, 16-byte-aligned FP16 Q/K/V/output on one device.
- Batch 1, Hq 6, Hkv 1, D256, causal, scale 1/16.
- Q is 64–8192 in multiples of 64; Q < KV <= 262144; KV is a multiple of 32.
- The bridge can prepend zero query rows to reach a 64-row boundary. It
  keeps the original KV length and slices away the leading outputs, so the
  causal offset of every real query is unchanged.
- A partial K32 tile is deliberately not admitted. A 16-aligned but
  non-32-aligned KV probe failed the FP64 gate during integration.
- CUDA Graph capture, unsupported shapes and insufficient workspace retain
  the existing exact fallback. Decode never enters this prefill operator.

The score cache reserves 768 MiB per used device (8192 queries times six
heads times 8192 prefix columns times two bytes). FP32 partials, statistics
and tail output use a transient slab. The first allocation requires this
workspace plus 128 MiB of downstream headroom. A per-device lock and CUDA
completion event serialize shared score/metadata use across caller streams.
An input-ready event starts the tail on its private stream; a completion
event joins it before the final merge.

## Build and identify the route

The normal SM70 FA2 CMake target includes these translation units. The
dedicated operator name, `sm70_d256_gqa_v37_fwd`, prevents an old FA2 library
from being mistaken for this implementation. If the operator is absent, the
backend warns and falls back instead of selecting the legacy architecture.

`VLLM_FLASH_V100_PREFILL_D256_GQA_V37=1` explicitly selects this v37 operator;
the default value of 0 selects the qualified Q8000 architecture operator on its
narrow shape family. The E4M3 bridge is resolved independently of the compute
operator. The existing long-prefill architecture switch remains an additional
gate. Explicit `fp8_e4m3` and `fp8_e5m2` retain their respective byte encodings;
this change does not reinterpret `fp8`.

The `prefill_dense_d256_gqa_v37` and `prefill_prefix_fp8_e4m3_bridge` route
counters identify actual execution. A requested environment variable alone
is not route-hit evidence.

## Focused validation

Run from an SM70-built environment:

```bash
.venv/bin/python -m pytest tests/v1/attention/test_sm70_v37_prefill.py tests/v1/attention/test_sm70_flash_v100_policy.py -q
.venv/bin/python -m pytest tests/kernels/attention/test_sm70_v37_prefill.py -q
```

The GPU tests compare causal attention against a PyTorch FP64 oracle, check
leading-query padding and unsupported alignment, and cover every E4M3 byte
at unit/non-unit scales, including graph replay with changing live lengths.
Long-context model acceptance additionally requires a matched no-MTP TP4
run with real route counters, finite logits, output checks, and separate
prefill/TTFT/decode timing. Operator tests alone do not establish that gate.
