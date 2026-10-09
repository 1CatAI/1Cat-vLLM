# Small-batch SM70 router operators

Flash-Next verification routes five rows through a replicated FP16
`[512, 2560]` router on every TP rank. The retained target trace attributes
about 0.714 ms to projection and 0.383 ms to selection across 48 layers.
These are profiled service sums with shared-expert concurrency, rather than
an isolated router benchmark or an additive part of the 17.4 ms endpoint.

The standalone 48-layer graph rotates all 48 actual GGUF router weights
(125,829,120 bytes in FP16), rather than repeatedly serving one L2-resident
matrix. Activations are synthetic FP16. Expert-chain fixtures use rank-0
TP4 shards from three actual layers, covering IQ3_S, IQ3_XXS and IQ2_S.

## Implementations

`vllm._sm70_router_C` is a normal CMake/setup extension. It exposes these
out-parameter CUDA operators without changing model dispatch:

* `vllm_sm70_router.top10`: sort sixteen packed FP16/expert-ID keys locally
  per lane, then merge only the ten winning list heads across a warp.
  Exponentiate and normalize only those winners. Support M1..20 and E512/K10.
* `vllm_sm70_router.select_quantize`: retain the quantizer's independent
  256-column blocks. One warp in the first block of each row also produces
  top-10 routing. Routing and Q8_1 activations become ready at one kernel
  boundary, without cross-CTA communication. Support M1..20 and K2560/E512.

The existing Q8_1 layout and warp quantizer are shared through
`gguf_q8_1.cuh`. Their formulas and reduction order are unchanged. This does
not introduce a new activation precision: the existing GGUF expert path
already consumes these Q8_1 packets. `select_quantize` must produce identical
packet bytes to that quantizer.

The selector preserves lower-expert-ID ties, signed-zero equality, rank-major
source indices and zero-weight degenerate rows (NaN, positive infinity, or
all negative infinity). FP32 normalization may differ by a few ulps. The
retained operators do not change the router projection or its reduction
order. The SIMT fused projection remains an unselected research result:
it reorders the K reduction and is slower than row-local selection plus
quantization in all three complete MoE fixtures. Both M5 and M20 keep the
control projection.

## Structural screens

The following are research-only graph measurements on V100-SXM2-32GB,
CUDA 12.8 and Torch 2.10.0+cu128. Each pair runs in one process with five
alternating ABBA/BAAB groups and ten observations per arm. The full router
chain interleaves projection and selection per layer. These early screens
disabled the existing top-16 selector and used Triton at M20. Their positive
deltas do not measure improvement over the current default dispatch.

| M5, 48 actual router matrices | Control ms | Candidate ms |
| --- | ---: | ---: |
| Register-local top-10, selection only | 0.206 | 0.160 |
| Exact candidate-threshold selection, selection only | 0.206 | 0.159 |
| SIMT projection, projection only | 0.351 | 0.338 |
| Original projection plus register-local selector | 0.556 | 0.524 |
| SIMT projection plus register-local selector | 0.557 | 0.514 |
| Single-launch SIMT projection and selector | 0.556 | 0.533 |
| Original HMMA plus selector and Q8, centralized completion | 0.650 | 0.776 |
| Original HMMA plus selector and Q8, tagged consumer | 0.656 | 1.067 |

Selection-only M20 improves 0.257 to 0.160 ms. The SIMT M20 projection
regresses 0.510 to 1.026 ms and changes some route IDs relative to the
control's reduction; it is rejected. Unsorted repeated winner scans save
only about 0.001 ms at M5, so they are also rejected. Threshold filtering
does not materially improve on the simpler register-list selector.

The single-launch SIMT projection also saves approximately 2.4–2.6 us per
complete MoE call in the normal-extension screen. Row-local selection plus
quantization is faster in every tested expert type and avoids changing the
projection reduction, so the SIMT entry is not shipped. Neither polling
prototype is shipped or retried through launch-parameter sweeps.

## Community references

[FlashInfer/ TensorRT-LLM top-k primitives](https://github.com/flashinfer-ai/flashinfer/blob/c1af49ecbad5433d4332aaec986d1a3bd1e1e610/csrc/fused_moe/moeTopKFuncs.cuh)
use register-local candidate lists and warp winner merging. This Apache-2.0
implementation informs the selector; SM70 uses shuffle reductions rather
than newer warp-reduction instructions.

[MonoMoE routing](https://github.com/flashinfer-ai/flashinfer/blob/c1af49ecbad5433d4332aaec986d1a3bd1e1e610/csrc/fused_moe/monomoe/src/moe_routing.cuh)
selects logits before evaluating winner exponentials. With renormalized
softmax routing, the full softmax denominator cancels, so only the ten
selected exponentials are needed.

[SGLang routing](https://github.com/sgl-project/sglang/blob/5cbf949839b1fc73802d20670943ccbeb805d3f2/python/sglang/kernels/ops/moe/router.py)
includes full-K CUDA-core routing and tensor-core fusion. Its tensor-core
entry supports top-k at most two; its one-program-per-token full-K schedule
does not directly supply E512/K10 parallelism on V100. The current work
retains the existing projection and measures complete dependency chains.

## Normal extension qualification

The normal CMake target was installed in a complete source runtime, without a
wheel rebuild, preload or private research library. The baseline core SHA256
is `815a458485194820bc4671295c924f26001911f74620d131492387c315cd10f2`.
The generic MoE module SHA256 is
`d61cc05c30b01009b5bbe95e8b16add5edf5ba2595c1bbc5caa3a0a673765170`.
The final two-operator module SHA256 is
`b15bf0ce1b4e047cfad23b7e36a765938d2c02dde1cfd565aa41cf986c3098b9`.
The control source is `86e9675964`; the normal-extension screening module
also contained the subsequently unselected fused-projection entry.
The final control enables the existing top-16 selector for M5 and uses
the dispatched generic CUDA `topk_softmax` at M20. Projection remains the
native packed FP16 operator at M5 and `torch.mm` at M20 in both arms.
Qualification uses one V100-SXM2-32GB at 1530 MHz SM / 877 MHz memory,
driver 580.173.02, CUDA 12.8 and Torch 2.10.0+cu128. The expert fixtures
represent TP4 rank-0 slices; this is not a simultaneous four-rank model run.

| Complete 48-layer graph chain | Control ms | Candidate ms | Saving ms |
| --- | ---: | ---: | ---: |
| M5 projection plus selection | 0.50060 | 0.52600 | -0.02541 |
| M5 projection, selection and Q8 | 0.60500 | 0.55629 | 0.04872 |
| M20 projection plus selection | 1.16617 | 0.71096 | 0.45521 |
| M20 projection, selection and Q8 | 1.30147 | 0.74701 | 0.55446 |

The selected row-local fusion saves 8.1% at M5 and 42.6% at M20. It
preserves all actual-weight expert/source IDs and every Q8_1 packet byte;
maximum normalized-weight difference is 1.79e-7. No projection value changes.
Standalone native selection regresses 5.1% against default top-16 at M5,
so keep that existing selector when Q8_1 is not needed. M20 standalone
selection improves 39.0% against the generic CUDA path; this covers the
prefix of expert paths that retain FP16 activations instead of Q8_1.

| Complete eight-call MoE chain | Control ms | Selection plus Q8 ms | Saving per call us |
| --- | ---: | ---: | ---: |
| IQ3_S | 0.66437 | 0.65078 | 1.70 |
| IQ3_XXS | 0.62003 | 0.60252 | 2.19 |
| IQ2_S | 0.60175 | 0.58798 | 1.72 |

Expert IDs and shared outputs remain exact; maximum routed-output relative
L2 difference is 9.03e-6. Across 48 layers this implies approximately
0.08–0.11 ms of service savings under these three fixtures. That estimate
is not a measured reduction of the historical 17.4 ms endpoint.
Standalone selection also regresses all three M5 MoE fixtures by
0.18–0.41 us per call. Only the fused selector/quantizer is recommended
for these M5 expert chains. Complete M20 MoE chains are not measured here.

The early fused-projection benchmark applied the selector's 3e-7 tolerance
to the projection-plus-selector comparison and stopped. Separating these
errors gives projection FP64 relative L2 at most 2.15e-4, almost identical
to the control, but normalized weights can differ by 2.85e-4 due to FP16
logit rounding after reassociation. The final implementation retains the
original projection and avoids this additional error source.

The generic M20 CUDA selector ranks rounded full-softmax scores. At FP16
subnormal logits around 5e-8, different logits can round to the same score
and change the lower-ID tie order. The native selector retains exact
raw-logit ordering, like the existing small-M Triton path. Changed-input
tests use an independent stable raw-logit sort and FP64 selected softmax;
they also compare default dispatch outside this rounding case. Real-weight
microbenchmarks still require identical default-dispatch expert/source IDs,
Q8 bytes and normalized weights within 3e-7 before timing.

## Reproduction

Build and install only `_sm70_router_C` from the normal CMake build. A wheel
rebuild is unnecessary when running a complete source runtime. Verify that
the extension depends only on standard Torch, CUDA and system libraries.

```bash
cmake --build "$BUILD_DIR" --target _sm70_router_C -j 2
cmake --install "$BUILD_DIR" --prefix "$SOURCE_RUNTIME" --component _sm70_router_C
flock /tmp/gpu0-3.lock .venv/bin/python -m pytest \
  tests/kernels/moe/test_sm70_router_native.py -q
flock /tmp/gpu0-3.lock .venv/bin/python benchmarks/kernels/benchmark_sm70_router_native.py \
  --model "$GGUF_MODEL" --out router-native.json
flock /tmp/gpu0-3.lock .venv/bin/python benchmarks/kernels/benchmark_sm70_router_moe_chain.py \
  --model-dir "$GGUF_DIRECTORY" --out router-moe.json
```

The test suite passes all eleven tests, covering 90 changed-input graph
cases, exact Q8_1 packets, independent FP64 selector semantics and storage
guards. The same 3e-7 absolute weight tolerance applies throughout.
The benchmarks require the shared GPU lock and idle selected GPUs. They do
not initialize the model, KV cache, PLE or MTP engine.
