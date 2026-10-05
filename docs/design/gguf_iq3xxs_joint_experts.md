# IQ3_XXS joint expert gate/up on SM70

Flash-Next IQ3_S checkpoints contain 17 IQ3_XXS gate/up expert layers alongside
IQ3_S, IQ2_S and IQ4_XS. The existing original-block grouped gate/up operator
supports IQ3_XXS, but the model preparation factory retained original rows only
for IQ3_S and IQ2_S. Add IQ3_XXS to the same retention policy; no new decoder or
operator is introduced.

Local TP4 expert rows have N160/K2560/E512. Gate/up partition along N preserves
complete source blocks along K. Each IQ3_XXS row contains 980 payload bytes plus
four alignment bytes. Retaining both matrices for 17 layers costs
2,740,715,520 additional bytes per rank (2.553 GiB), excluding allocator overhead.
Canonical banks remain available for unsupported batches. TP4 is unchanged.

## Operator policy

The existing capability framework admits original M1/M5/M20 for IQ3_XXS and
IQ3_S. IQ2_S M20 remains on canonical grouped GEMM. The opaque dispatch evaluates
original token count before top-k expansion, preserving behavior across range
compilation and full CUDA graphs. Missing retained storage and unsupported
original batches continue to report their fallback reasons.

## Validation

An ordinary source-containing SM70 wheel with CUDA12.8, Torch2.10.0+cu128 and
Python3.12 passes 33 CPU dispatch/bank/capability checks. Six GPU graph cases
cover IQ3_XXS/IQ3_S/IQ2_S at M5/M20 with changed inputs and official GGUF
dequantization as the numerical oracle. Tests preserve FP16 operands and FP32
accumulation. No private shared library or preload is needed.

Previously recorded independent actual-weight, cache-exceeding M5 pair timings
are 75.269 µs canonical and 61.417 µs original blocks for IQ3_XXS. These measure
one gate/up pair, not complete model rounds. Additional retention memory and
request acceptance must be assessed separately in the TP4 model run. The matched
model comparison uses the same memory budget for GGUF and NVFP4.
