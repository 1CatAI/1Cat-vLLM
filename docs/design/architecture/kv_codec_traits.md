# SM70 CUDA KV codec traits

`flash-attention-v100/kernel/kv_codec_traits.cuh` is the format contract for
FP16, FP8 E4M3 and FP8 E5M2 payloads. `KVCodecTraits<KV_DTYPE, E4M3_BITS>`
provides `storage_type`, `quantized`, `element_bytes`, `vector_elements` and
`vector_bytes`, together with the shared reader operations:

| Operation | Contract |
| --- | --- |
| `unscaled(cache, index)` | Scalar conversion to float; index is in payload elements. |
| `scaled(cache, index, scale)` | Existing FP32 scalar multiply. |
| `half(cache, index, scale)` | Existing conversion and rounding to half. |
| `load_half8<SHARED_LUT>(cache, offset, column, lut)` | Eight unscaled half values; base offset in elements, column in eight-element vectors. |
| `half8_from_packed<FAST, SHARED_LUT>(raw, lut)` | Eight FP8 bytes converted to half; FP16 is rejected at compile time. |

The reader implementation in `kv_codec.cuh` is reused from
[PR #1048](https://github.com/1CatAI/1Cat-vLLM/pull/1048), source revision
`c0737e46210ac5b1273b78d622de1eefa71fd4de`. This scope adds storage traits and
adopts them in the existing XQA/grouped family; it does not introduce a second
reader algorithm or duplicate #1048's storage accounting and Triton writer
work. Existing `KVReader`, scalar helper names and both `fp8_kv_utils.cuh`
include paths remain available. Internal native callers use the owner headers.
Both source distributions include the shared headers.

Scheduling, page addressing, scale placement, accumulator type and reduction
remain operator responsibilities. In particular, FP8 uses separate scalar K/V
scales; this interface does not yet provide INT8 group sidecars or a new launch
ABI. BF16 payloads are not a native reader specialization.

`paged_to_contiguous_old.cu` and `paged_to_contiguous_fixed.cu` are deprecated
experimental variants retained for reproduction. Production builds use
`paged_to_contiguous.cu`.

## Native prefill and bridge fallback

The packaged general paged-prefill function advertises `native_kv_codecs`,
copied from the loaded extension's `prefill_paged_native_kv_codecs` declaration.
The declaration covers FP16/E4M3/E5M2 in that entry point only; FA2 and split-KV
have narrower contracts. Backend initialization captures this capability
before feature hooks wrap the callable in `functools.partial`.

When the installed general operator supports the selected codec, prefix
prefill reads the original compressed cache and bypasses the bridge. Its
selection is visible as `prefill_prefix_paged`. The existing FP8 bridge is
eligible only without a declared native reader and within its original
shape/layout/policy contract. Older extensions lacking the declaration retain
their previous selection. The two existing outer bridge route names are now
declared fallbacks, logged once and counted even without debug routing enabled;
the nested exact-route details do not double-count the fallback. Direct helper
calls also reject bridge execution when native support has been captured.

This is an intentional routing change for rebuilt extensions, separate from
the reader extraction. It can select the general paged kernel instead of
bridge plus FA2. Output error and performance relative to the bridge are
unmeasured; no speedup or latency acceptance is claimed. The legacy environment
adapter still disables bridge eligibility when set to zero, and does not
override the native-reader preference when set to one.

## Static validation

`tests/tools/test_sm70_kv_codec_traits.py` compiles a frozen A4b reader and the
current traits with NVCC 12.0, C++17, `sm_70` and `--use_fast_math`. Their PTX
instructions match for scalar conversion/scaling/rounding, regular/fast/LUT
packed conversion and vector cache addressing. Runtime inputs retain NaN,
signed-zero, subnormal and arbitrary-scale cases in the compiled branches.
The test skips explicitly when NVCC is unavailable.

This checks compiler transformations without GPU execution. It does not measure
GPU max-abs error, CUDA graph replay, route traces or latency, and it does not
build a linked Torch extension or qualify a wheel. V100 validation is omitted
at the user's request.
